import torch
from einops import rearrange

class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None,
                delta_softplus=False, return_last_state=False):
        """
        Pure PyTorch implementation of selective_scan
        u: input tensor (batch, length, dim)
        delta: delta tensor
        A, B, C, D: linear transformations
        """
        # Ensure contiguous
        u = u.contiguous()
        delta = delta.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        if z is not None:
            z = z.contiguous()
        
        # Squeeze dimensions if needed
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        else:
            ctx.squeeze_B = False
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        else:
            ctx.squeeze_C = False
        
        # Simple PyTorch loop-based forward
        batch, L, dim = u.shape
        dstate = B.shape[2]
        x = torch.zeros(batch, dim, dstate, device=u.device, dtype=u.dtype)
        out = torch.zeros_like(u)
        for t in range(L):
            # A simple linear recurrence example:
            x = torch.einsum('bd,dd->bd', u[:, t], A) + torch.einsum('bdi,di->bd', B[:, :, t], x)
            x = x + torch.einsum('bdi,di->bd', C[:, :, t], u[:, t])
            if D is not None:
                x = x + D
            out[:, t] = x  # store output at time t
        
        last_state = x.clone()
        ctx.save_for_backward(u, delta, A, B, C, D)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        
        if return_last_state:
            return out, last_state
        return out

    @staticmethod
    def backward(ctx, dout):
        # Simple PyTorch backward: autograd handles it
        u, delta, A, B, C, D = ctx.saved_tensors
        du = torch.autograd.grad(outputs=[ctx.saved_tensors[0]], inputs=[u], grad_outputs=[dout], retain_graph=True)[0]
        dA = torch.autograd.grad(outputs=[ctx.saved_tensors[0]], inputs=[A], grad_outputs=[dout], retain_graph=True)[0]
        dB = torch.autograd.grad(outputs=[ctx.saved_tensors[0]], inputs=[B], grad_outputs=[dout], retain_graph=True)[0]
        dC = torch.autograd.grad(outputs=[ctx.saved_tensors[0]], inputs=[C], grad_outputs=[dout], retain_graph=True)[0]
        dD = torch.autograd.grad(outputs=[ctx.saved_tensors[0]], inputs=[D], grad_outputs=[dout], retain_graph=True)[0] if D is not None else None
        return du, None, dA, dB, dC, dD, None, None, None, None


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)