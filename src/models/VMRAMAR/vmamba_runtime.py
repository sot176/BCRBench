from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from functools import partial
from typing import Callable
import types

import torch
import torch.nn.functional as F
from torch import nn

_SELECTIVE_SCAN_FN = None
_SELECTIVE_SCAN_BACKEND = None
_SELECTIVE_SCAN_ERRORS: dict[str, Exception] = {}


def _load_mamba_selective_scan_fn():
    """Load the selective-scan interface without executing mamba_ssm.__init__.

    Newer mamba_ssm releases import optional Hugging Face / LM modules from the
    package root, which are irrelevant for VMRNN's selective-scan backend. We
    only need ops/selective_scan_interface.py, so we register a lightweight
    package stub and load that module directly from disk.
    """

    package_name = "mamba_ssm"
    package_spec = importlib.util.find_spec(package_name)
    if package_spec is None or not package_spec.submodule_search_locations:
        raise ModuleNotFoundError(f"Package {package_name!r} is not installed.")

    package_dir = Path(next(iter(package_spec.submodule_search_locations)))
    module_name = "mamba_ssm.ops.selective_scan_interface"
    module_path = package_dir / "ops" / "selective_scan_interface.py"
    if not module_path.exists():
        raise ModuleNotFoundError(f"Expected selective-scan interface at {module_path}.")

    import sys

    package_module = types.ModuleType(package_name)
    package_module.__path__ = [str(package_dir)]
    package_module.__package__ = package_name
    package_module.__file__ = str(package_dir / "__init__.py")
    package_module.__spec__ = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    sys.modules[package_name] = package_module
    if module_name in sys.modules:
        del sys.modules[module_name]

    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Unable to create a module spec for {module_path}.")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module.selective_scan_fn


try:
    _mamba_selective_scan_fn = _load_mamba_selective_scan_fn()
except Exception as exc:  # pragma: no cover - environment dependent
    _SELECTIVE_SCAN_ERRORS["mamba_ssm"] = exc
else:  # pragma: no branch
    _SELECTIVE_SCAN_FN = _mamba_selective_scan_fn
    _SELECTIVE_SCAN_BACKEND = "mamba_ssm"

if _SELECTIVE_SCAN_FN is None:
    try:
        from selective_scan import selective_scan_fn as _selective_scan_fn_v1
    except Exception as exc:  # pragma: no cover - environment dependent
        _SELECTIVE_SCAN_ERRORS["selective_scan"] = exc
    else:  # pragma: no branch
        _SELECTIVE_SCAN_FN = _selective_scan_fn_v1
        _SELECTIVE_SCAN_BACKEND = "selective_scan"


def is_vmamba_kernel_available() -> bool:
    return _SELECTIVE_SCAN_FN is not None


def vmamba_kernel_backend_name() -> str | None:
    return _SELECTIVE_SCAN_BACKEND


def vmamba_unavailable_reason() -> str:
    if is_vmamba_kernel_available():
        return f"VMamba kernel is available through {_SELECTIVE_SCAN_BACKEND}."
    if not _SELECTIVE_SCAN_ERRORS:
        return "No selective scan backend could be imported."

    parts = []
    for name, error in _SELECTIVE_SCAN_ERRORS.items():
        parts.append(f"{name}: {type(error).__name__}: {error}")
    return " ; ".join(parts)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class SS2D(nn.Module):
    """Reference SS2D block used by VMamba-backed VMRNN blocks."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
        **_: object,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        x_proj = tuple(
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            for _ in range(4)
        )
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in x_proj], dim=0))

        dt_projs = tuple(
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(4)
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([layer.weight for layer in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([layer.bias for layer in dt_projs], dim=0))

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0 if _SELECTIVE_SCAN_BACKEND == "mamba_ssm" else self.forward_corev1
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        **factory_kwargs,
    ) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state: int, d_inner: int, copies: int = 1, device=None, merge: bool = True) -> nn.Parameter:
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).unsqueeze(0).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        parameter = nn.Parameter(A_log)
        parameter._no_weight_decay = True
        return parameter

    @staticmethod
    def D_init(d_inner: int, copies: int = 1, device=None, merge: bool = True) -> nn.Parameter:
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D.unsqueeze(0).repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        parameter = nn.Parameter(D)
        parameter._no_weight_decay = True
        return parameter

    def _scan_tensor(self, xs, dts, As, Bs, Cs, Ds, dt_projs_bias):
        if _SELECTIVE_SCAN_FN is None:
            raise RuntimeError(f"VMamba kernel requested but unavailable: {vmamba_unavailable_reason()}")

        if _SELECTIVE_SCAN_BACKEND == "mamba_ssm":
            return _SELECTIVE_SCAN_FN(
                xs,
                dts,
                As,
                Bs,
                Cs,
                Ds,
                z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            )
        return _SELECTIVE_SCAN_FN(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        )

    def forward_corev0(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self._scan_tensor(xs, dts, As, Bs, Cs, Ds, dt_projs_bias).view(B, K, -1, L)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        return self.out_norm(y).to(x.dtype)

    def forward_corev1(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.view(B, K, -1, L)
        Cs = Cs.view(B, K, -1, L)

        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        Ds = self.Ds.view(-1)
        dt_projs_bias = self.dt_projs_bias.view(-1)

        out_y = self._scan_tensor(xs, dts, As, Bs, Cs, Ds, dt_projs_bias).view(B, K, -1, L)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        return self.out_norm(y).to(x.dtype)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        if not is_vmamba_kernel_available():
            raise RuntimeError(f"VMamba kernel requested but unavailable: {vmamba_unavailable_reason()}")

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor + self.drop_path(self.self_attention(self.ln_1(input_tensor)))


class VMambaVSBBlock(VSSBlock):
    """True VMamba-backed VSB block with recurrent hidden-state fusion."""

    def __init__(
        self,
        hidden_dim: int,
        input_resolution: tuple[int, int],
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            **kwargs,
        )
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.input_resolution = input_resolution

    def forward(self, x: torch.Tensor, hx: torch.Tensor | None = None) -> torch.Tensor:
        height, width = self.input_resolution
        batch_size, token_count, channels = x.shape
        if token_count != height * width:
            raise ValueError(f"Expected {height * width} tokens for resolution {self.input_resolution}, got {token_count}.")

        shortcut = x
        x = self.ln_1(x)
        if hx is not None:
            hx = self.ln_1(hx)
            x = torch.cat((x, hx), dim=-1)
            x = self.linear(x)

        x = x.view(batch_size, height, width, channels)
        x = self.drop_path(self.self_attention(x))
        x = x.view(batch_size, height * width, channels)
        return shortcut + x
