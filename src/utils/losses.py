import torch.nn.functional as F
import torch
import torch.nn as nn



def loss_factory(args, criterion_POE=None, criterion_MV=None):
    """
    Returns a loss function for a given model.
    """
    if args.model == "OA-BreaCR":

        # Use their exact BCE loss
        criterion_BCE = risk_BCE_loss(
            weight_loss=2,
            batch_size=args.batch_size,
            num_pred_years=getattr(args, "max_followup", 6),
        )

        def _compute_head_loss(risk, risk_label, years_last_followup, emb, log_var):
            risk_label          = risk_label.clone()           
            years_last_followup = years_last_followup.clone()
            is_sto = risk.dim() == 3

            if is_sto and emb is not None:
                sample_size, _, out_dim = risk.shape
                risk_flat  = risk.view(-1, out_dim)
                label_flat = risk_label.repeat(sample_size)
                years_flat = years_last_followup.repeat(sample_size)
            else:
                risk_flat  = risk
                label_flat = risk_label
                years_flat = years_last_followup

            # BCE — matches their compute_losses exactly
            loss = criterion_BCE(
                risk_flat, label_flat, years_flat,
                weights=getattr(args, "time_to_events_weights", None)
            )

            # MV
            if criterion_MV is not None:
                loss_MV = criterion_MV(
                    risk_flat, label_flat, years_flat,
                    weights=getattr(args, "time_to_events_weights", None)
                )
                loss = loss + loss_MV

            # POE
            if emb is not None and log_var is not None and criterion_POE is not None:
                _, _, _, loss_POE = criterion_POE(
                    risk, emb, log_var,
                    risk_label, years_last_followup, None,
                    use_sto=is_sto,
                    weights=getattr(args, "time_to_events_weights", None)
                )
                loss = loss + loss_POE

            return loss

        def oa_breacr_loss(outputs, batch, model_risk):
            device     = next(model_risk.parameters()).device
            total_loss = torch.tensor(0.0, device=device)

            if outputs.get("loss") is not None:
                total_loss += outputs["loss"]

            # ── BCE + MV + POE per head ───────────────────────────────
            # Note: BCE  handles target/mask internally like their code
            head_configs = {
                "final":      (outputs.get("final"),      batch["years_to_cancer"],       batch["years_to_last_followup"],       outputs.get("emb_final"),      outputs.get("log_var_final"),      1.0),
                "current":    (outputs.get("current"),    batch["years_to_cancer"],       batch["years_to_last_followup"],       outputs.get("emb_current"),    outputs.get("log_var_current"),    0.2),
                "difference": (outputs.get("difference"), batch["years_to_cancer"],       batch["years_to_last_followup"],       outputs.get("emb_difference"), outputs.get("log_var_difference"), 0.2),
                "prior":      (outputs.get("prior"),      batch["years_to_cancer_prior"], batch["years_to_last_followup_prior"], outputs.get("emb_prior"),      outputs.get("log_var_prior"),      0.2),
            }

            for head_name, (risk, risk_label, years_lfu, emb, log_var, weight) in head_configs.items():
                if risk is None:
                    continue
                total_loss += weight * _compute_head_loss(
                    risk, risk_label, years_lfu, emb, log_var
                )

            return total_loss

        return oa_breacr_loss

    else:
        # Default BCE-only models for e.g. Mirai, ImgFeatAlign, LMV-Net, VMRA-MaR, etc.
        def default_loss(outputs, batch, model_risk):
            risk_heads = model_risk.get_risk_heads(outputs, batch)  # Use helper
            total_loss = 0.0
            for head_name, (logits, target, mask) in risk_heads.items():
                total_loss += get_risk_loss_BCE(
                    logits , target , mask
                )
            return total_loss
        return default_loss


#########################################################################
# ------------------ Risk loss ----------------------
#########################################################################


def get_risk_loss_BCE(pred, y_true, y_mask):

    y_mask = y_mask.to(pred.device)
    y_true = y_true.to(pred.device)

    mask_sum = torch.sum(y_mask.float())

    if mask_sum == 0:
        return torch.tensor(0.0, device=pred.device)

    loss = F.binary_cross_entropy_with_logits(
        pred,
        y_true.float(),
        weight=y_mask.float(),
        reduction='sum'
    )

    return loss / mask_sum



class risk_BCE_loss(nn.Module):
    """
    Defines for computing the risk prediction loss for time-to-event data.

    This class calculates a custom loss function based on binary
    cross-entropy. It utilizes masking to handle censored data during follow-up periods and allows
    for weighted contributions to the loss computation to reflect varying levels of significance
    across events and time periods. The class is parameterized for flexibility in accounting for
    different datasets and settings.
    """
    def __init__(self, weight_loss=2, batch_size=1, num_pred_years=6):
        super(risk_BCE_loss, self).__init__()
        self.weight_loss = weight_loss
        self.batch_size = batch_size
        self.num_pred_years = num_pred_years
        self.max_followup = num_pred_years

    def forward(self, pred, risk_label, years_last_followup, weights=None):
        pred = F.softmax(pred, dim=1)
        batch_size, num_pred_years = pred.shape
        followup = num_pred_years - 1
        print("pred", pred)
        print("risk label", risk_label)
        print("years last followup", years_last_followup)
        risk_label = risk_label.cpu().detach().numpy()
        years_last_followup = years_last_followup.cpu().detach().numpy()
        device = pred.device  # or self.device
        risk_mask = torch.ones((batch_size, self.num_pred_years), device=device)
        y_seq = torch.zeros((batch_size, self.num_pred_years), device=device)

        risk_label[risk_label > (num_pred_years - 1)] = num_pred_years - 1
        for i in range(batch_size):
            y_seq[i, risk_label[i]] = 1
            # ra = risk_label[i]
            # fa = years_last_followup[i]
            if risk_label[i] == followup and years_last_followup[i] < followup:
                risk_mask[i, years_last_followup[i] + 1:] = 0

        if torch.sum(risk_mask.float()) == 0:
            print('wrong!!!!!!!!!', torch.sum(risk_mask.float()))

        if weights is not None:
            weights_ = torch.tensor(weights, dtype=torch.float, device=pred.device).view(1, -1)
            risk_mask = risk_mask * weights_

        loss = F.binary_cross_entropy(
            pred, y_seq.float(),
            weight=risk_mask.float(),
            reduction='sum'
        ) / torch.sum(risk_mask.float()) * self.weight_loss


        return loss 

#########################################################################
# ------------------ Mean Variance loss ------------------
#########################################################################


class MeanVarianceLoss(nn.Module):

    def __init__(self, lambda_1=0.2, lambda_2=0.05, cumpet_ce_loss=False, start_label=0):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cumpet_ce_loss = cumpet_ce_loss
        self.start_label = start_label

    def forward(self, input, target_label, years_last_followup, weights=None):
        class_dim = input.shape[-1]
        batch_size = input.shape[0]

        target_ = target_label.detach()
        target_[target_ > (class_dim - 1)] = class_dim - 1
        mask = 1 - ((target_.cpu() == (class_dim - 1)) & (years_last_followup.cpu() < (class_dim - 1))).int()
        count = sum(mask)
        device = input.device
        if sum(mask) > 0:
            input = input[mask == 1, ...]
            target_ = target_[mask == 1, ...]
            years_last_followup = years_last_followup[mask == 1, ...]

            p = F.softmax(input, dim=-1)
            # mean loss
            a = torch.arange(class_dim, dtype=torch.float32, device=device)
            target = target_.to(device)
            mean = torch.squeeze((p * (a + self.start_label)).sum(1, keepdim=True), dim=1)
            mse = (mean - target)**2
            # new_weights = torch.zeros_like(target)
            # weights = None
            if weights is not None:
                weights_ = torch.tensor(weights, dtype=torch.float).view(1, -1)
                weights_ = weights_.repeat([count,1])
                weights_ = weights_[range(count), target_.cpu()]
                weights_ = weights_.to(device)  # move weights to device
                mean_loss = sum(mse * weights_) / sum(weights_) / 2.0
                # mean_loss = (mse * weights_).mean() / 2.0

                b = (a[None, :] - mean[:, None]) ** 2
                # variance_loss = ((p * b).sum(1, keepdim=False) * weights_).mean()
                variance_loss = sum((p * b).sum(1, keepdim=False) * weights_) / sum(weights_)
            else:
                mean_loss = mse.mean() / 2.0
                # variance loss
                b = (a[None, :] - mean[:, None]) ** 2
                variance_loss = (p * b).sum(1, keepdim=True).mean()
        else:
            mean_loss = torch.tensor(0.0,  device=device)
            variance_loss = torch.tensor(0.0,  device=device)

        loss = (self.lambda_1 * mean_loss) + (self.lambda_2 * variance_loss)
        return loss


#########################################################################
# ------ POE loss adapted from https://github.com/Li-Wanhua/POEs --------
#########################################################################


def BhattacharyyaDistance(u1, sigma1, u2, sigma2):
    sigma_mean = (sigma1 + sigma2) / 2.0
    sigma_inv = 1.0 / (sigma_mean)
    dis1 = torch.sum(torch.pow(u1 - u2, 2) * sigma_inv, dim=1) / 8.0
    dis2 = 0.5 * (torch.sum(torch.log(sigma_mean), dim=1) -
                  0.5 * (torch.sum(torch.log(sigma1), dim=1) + torch.sum(torch.log(sigma2), dim=1)))
    return dis1 + dis2


def HellingerDistance(u1, sigma1, u2, sigma2):
    return torch.pow(1.0 - torch.exp(-BhattacharyyaDistance(u1, sigma1, u2, sigma2)), 0.5)


def WassersteinDistance(u1, sigma1, u2, sigma2):
    dis1 = torch.sum(torch.pow(u1 - u2, 2), dim=1)
    dis2 = torch.sum(torch.pow(torch.pow(sigma1, 0.5) -
                               torch.pow(sigma2, 0.5), 2), dim=1)
    return torch.pow(dis1 + dis2, 0.5)


def GeodesicDistance(u1, sigma1, u2, sigma2):
    u_dis = torch.pow(u1 - u2, 2)
    std1 = sigma1.sqrt()
    std2 = sigma2.sqrt()

    sig_dis = torch.pow(std1 - std2, 2)
    sig_sum = torch.pow(std1 + std2, 2)
    delta = torch.div(u_dis + 2 * sig_dis, u_dis + 2 * sig_sum).sqrt()
    return torch.sum(torch.pow(torch.log((1.0 + delta) / (1.0 - delta)), 2) * 2, dim=1).sqrt()


def ForwardKLDistance(u1, sigma1, u2, sigma2):
    return -0.5 * torch.sum(torch.log(sigma1) - torch.log(sigma2) - torch.div(sigma1, sigma2)
                            - torch.div(torch.pow(u1 - u2, 2), sigma2) + 1, dim=1)


def ReverseKLDistance(u2, sigma2, u1, sigma1):
    return -0.5 * torch.sum(torch.log(sigma1) - torch.log(sigma2) - torch.div(sigma1, sigma2)
                            - torch.div(torch.pow(u1 - u2, 2), sigma2) + 1, dim=1)


def JDistance(u1, sigma1, u2, sigma2):
    return ForwardKLDistance(u1, sigma1, u2, sigma2) + ForwardKLDistance(u2, sigma2, u1, sigma1)


class ProbOrdiLoss(nn.Module):
    def __init__(self, distance='Bhattacharyya', alpha_coeff=0, beta_coeff=0, margin=0, main_loss_type='cls',
                 criterion='l1', start_label=0):
        super(ProbOrdiLoss, self).__init__()
        self.alpha_coeff = alpha_coeff
        self.beta_coeff = beta_coeff
        self.margin = margin
        self.start_label = start_label

        assert main_loss_type in ['cls', 'reg', 'rank'], \
            "main_loss_type not in ['cls', 'reg', 'rank'], loss type {%s}" % (
                main_loss_type)
        self.main_loss_type = main_loss_type
        self.criterion = criterion

        if distance == 'Bhattacharyya':
            self.distrance_f = BhattacharyyaDistance
        elif distance == 'Wasserstein':
            self.distrance_f = WassersteinDistance
        elif distance == 'JDistance':
            self.distrance_f = JDistance
        elif distance == 'ForwardKLDistance':
            self.distrance_f = ForwardKLDistance
        elif distance == 'HellingerDistance':
            self.distrance_f = HellingerDistance
        elif distance == 'GeodesicDistance':
            self.distrance_f = GeodesicDistance
        elif distance == 'ReverseKLDistance':
            self.distrance_f = ReverseKLDistance
        else:
            print('ERROR: this distance is not supported!')
            self.distrance_f = None

    def forward(self, logit, emb, log_var, target_label, years_last_followup, mh_target=None, use_sto=True, weights=None):
        device = logit.device  # use device of input tensors dynamically

        class_dim = logit.shape[-1]
        target_ = target_label.detach()
        target_[target_ > (class_dim - 1)] = class_dim - 1
        mask = 1 - ((target_.cpu() == (class_dim - 1)) & (years_last_followup.cpu() < (class_dim - 1))).int()
        KLLoss = torch.mean(torch.sum(torch.pow(emb, 2) + torch.exp(log_var) - log_var - 1.0, dim=1) * 0.5)

        count = sum(mask)
        if sum(mask) > 0:
            emb = emb[mask == 1, ...]
            log_var = log_var[mask == 1, ...]
            target = target_label[mask == 1, ...]

            var = torch.exp(log_var)
            batch_size = emb.shape[0]

            target_dis = torch.abs(target.view(-1, 1).repeat(1, batch_size) - target.view(1, -1).repeat(batch_size, 1))
            anchor_pos = [i for i in range(batch_size)]
            second_pos = [(i + 1) % batch_size for i in anchor_pos]
            target_dis = torch.abs(target_dis - torch.abs(target[anchor_pos] - target[second_pos]).view(-1, 1).repeat(1, batch_size))

            offset_m = torch.eye(batch_size, device=device).to(dtype=target_dis.dtype)  # create on device dynamically
            target_dis = target_dis + offset_m * 1000
            target_dis[target_dis == 0] = 700
            thrid_pos = torch.argmin(target_dis, dim=1)

            anchor_sign = torch.sign(torch.abs(target[anchor_pos] - target[second_pos]) - torch.abs(target[anchor_pos] - target[thrid_pos]))

            emb_dis_12 = self.distrance_f(emb[anchor_pos, :], var[anchor_pos, :], emb[second_pos, :], var[second_pos, :])
            emb_dis_13 = self.distrance_f(emb[anchor_pos, :], var[anchor_pos, :], emb[thrid_pos, :], var[thrid_pos, :])

            anchor_cons = (emb_dis_13 - emb_dis_12) * anchor_sign.float() + self.margin

            zeros_tensor = torch.zeros_like(anchor_cons)  # create zeros tensor on device dynamically
            loss_anchor = torch.max(zeros_tensor, anchor_cons) * torch.abs(anchor_sign).float()
            loss_mask = (anchor_cons > 0).to(dtype=anchor_sign.dtype)

            if sum(torch.abs(anchor_sign) * loss_mask) > 0:
                triple_loss = torch.sum(loss_anchor) / sum(torch.abs(anchor_sign) * loss_mask)
            else:
                triple_loss = torch.tensor(0.0, device=device)

        else:
            triple_loss = torch.tensor(0.0, device=device)

        return None, KLLoss * self.alpha_coeff, triple_loss * self.beta_coeff, self.alpha_coeff * KLLoss + self.beta_coeff * triple_loss
