import torch
import torch.nn.functional as F

def hybrid_asymmetry(left, right, latent_h=5, latent_w=5, 
                     verbose=False, flexible=False, topk=None, 
                     bias_params=None, **kwargs):
    if verbose and (left.shape[-2] % latent_h != 0):
        print("WARNING: Height dimension {} not divisible by {}".format(left.shape[-2], latent_h))

    if verbose and (left.shape[-1] % latent_w != 0):
        print("WARNING: Width dimension {} not divisible by {}".format(left.shape[-1], latent_w))
    if bias_params is None:
        dif = torch.abs(left - right)
    else:
        dif = torch.abs(left - right + bias_params)
    
    kernel_h = dif.shape[-2] // latent_h
    kernel_w = dif.shape[-1] // latent_w
    pooling_kernel_shape = (kernel_h, kernel_w)

    if flexible:
        dif = F.max_pool2d(dif, pooling_kernel_shape, 
                    stride=(1, 1))
    else:
        dif = F.max_pool2d(dif, pooling_kernel_shape, 
                    stride=(kernel_h, kernel_w))

    dif = torch.norm(dif, dim=-3)
    
    if topk is None:
        max_by_ftr, y_argmin = torch.max(dif, dim=-1)        # (B, latent_h)
        max_asym, x_argmin = torch.max(max_by_ftr, dim=-1)   # (B,)

        # Get the y coordinate at the winning x position
        best_y_argmin = y_argmin[
            torch.arange(y_argmin.shape[0], device=y_argmin.device),
            x_argmin
        ]  # (B,) — pick the y at the column that had max asymmetry

        return max_asym, {
            'y_argmin': best_y_argmin,   # (B,) — consistent with x_argmin
            'x_argmin': x_argmin,        # (B,)
            'heatmap': dif.detach()
        }
    else:
        topk_by_ftr, indices = torch.topk(dif.view(dif.shape[0], -1), topk, dim=-1)

        return topk_by_ftr, {'y_argmin': -1, 'x_argmin': -1, 'heatmap': dif.detach()}
