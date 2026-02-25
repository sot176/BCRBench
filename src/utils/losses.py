import torch.nn.functional as F
import torch
import torch.nn as nn

from OA_BreaCR_new.losses.mean_variance_loss import MeanVarianceLoss



def loss_factory(model_name, args):
    """
    Returns a loss function for a given model.
    """
    if model_name == "OA-BreaCR":
        def oa_breacr_loss(outputs, batch):
            device = next(iter(outputs.values())).device  # pick device from any output tensor
            total_loss = 0.0
            max_followup = args.num_output_neurons
            MV_loss = MeanVarianceLoss()
            POE_loss = ProbOrdiLoss()

            # --- optional extra loss from network ---
            if outputs.get('loss') is not None:
                total_loss += outputs['loss'].to(device)

            # --- compute targets & masks ---
            def compute_target_mask(years_to_cancer, years_last_followup):
                B = years_to_cancer.shape[0]
                y_true = torch.zeros(B, max_followup, device=device)
                y_mask = torch.ones(B, max_followup, device=device)
                years_to_cancer = years_to_cancer.clamp(0, max_followup-1)
                years_last_followup = years_last_followup.clamp(0, max_followup-1)
                for i in range(B):
                    y_true[i, :years_to_cancer[i]+1] = 1
                    if years_to_cancer[i] == max_followup-1 and years_last_followup[i] < max_followup-1:
                        y_mask[i, years_last_followup[i]+1:] = 0
                return y_true, y_mask

            y_true_final, y_mask_final = compute_target_mask(batch['years_to_cancer'], batch['years_to_last_followup'])
            y_true_prior, y_mask_prior = compute_target_mask(batch['years_to_cancer_prior'], batch['years_to_last_followup_prior'])

            # --- BCE for all heads ---
            risk_pred = outputs["risk_prediction"]
            for head_name, logits in risk_pred.items():

                if head_name in ["pred_fused", "final", "fused"]:
                    y_true, y_mask = y_true_final, y_mask_final
                    weight = 1.0
                elif head_name in ["pred_pri", "prior"]:
                    y_true, y_mask = y_true_prior, y_mask_prior
                    weight = 0.2
                else:  # current/difference
                    y_true, y_mask = y_true_final, y_mask_final
                    weight = 0.2

                if y_mask.sum() == 0:  # avoid NaNs
                    continue

                total_loss += weight * get_risk_loss_BCE(logits.to(device), y_true.to(device), y_mask.to(device))

            # --- MV loss for main/final head ---
            if MV_loss is not None:
                total_loss += 0.2 * MV_loss(
                    risk_pred['pred_fused'].to(device),
                    batch['years_to_cancer'].to(device),
                    batch['years_to_last_followup'].to(device),
                    weights=None
                )

            # --- POE loss ---
            if POE_loss is not None and outputs.get('emb_final') is not None:
                _, _, _, loss_POE = POE_loss(
                    risk_pred['pred_fused'].to(device),
                    outputs['emb_final'].to(device),
                    outputs['log_var_final'].to(device),
                    batch['years_to_cancer'].to(device),
                    batch['years_to_last_followup'].to(device),
                    None,
                    use_sto=args.use_sto,
                    weights=None
                )
                total_loss += 0.2 * loss_POE

            return total_loss

        return oa_breacr_loss

    else:
        # Default BCE-only models
        def default_loss(outputs, batch):
            device = next(outputs["risk_prediction"]["pred_fused"].parameters()).device
            risk_heads = outputs["risk_prediction"]
            total_loss = 0.0
            for head_name, (logits, target, mask) in risk_heads.items():
                if logits is None:
                    continue
                logits, target, mask = logits.to(device), target.to(device), mask.to(device)
                total_loss += get_risk_loss_BCE(logits, target, mask)
            return total_loss

        return default_loss


#########################################################################
# ------------------ Risk loss ----------------------
#########################################################################


def get_risk_loss_BCE( pred, y_true, y_mask):
    """
    Binary cross-entropy loss adapted for cumulative risk prediction with masking.
    Args:
        pred: Logits for cumulative risk, tensor of shape [B, T]
        y_true: Binary ground truth labels, tensor of shape [B, T]
                (1 if event happened by year t)
        y_mask: Mask tensor of shape [B, T], where 1 indicates valid data for year t
                and 0 indicates censored or invalid data

    Returns:
        masked_loss: Scalar tensor representing the masked binary cross-entropy loss.
    """

    y_mask = y_mask.to(pred.device)
    y_true = y_true.to(pred.device)
    masked_loss = F.binary_cross_entropy_with_logits(
        pred, y_true.float(), weight=y_mask.float(),  reduction='sum'
    ) / torch.sum(y_mask.float())

    return masked_loss


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
