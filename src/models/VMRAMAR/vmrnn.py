"""Formal multi-scale VMRNN blocks with VMamba-only VSS cells."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from .vmamba_runtime import (
    VMambaVSBBlock,
    is_vmamba_kernel_available,
    vmamba_kernel_backend_name,
    vmamba_unavailable_reason,
)


def _num_tokens(resolution: tuple[int, int]) -> int:
    return resolution[0] * resolution[1]


def _scale_resolution(resolution: tuple[int, int], factor: int) -> tuple[int, int]:
    return resolution[0] // factor, resolution[1] // factor


def _tokens_to_map(tokens: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
    height, width = resolution
    batch_size, token_count, channels = tokens.shape
    if token_count != height * width:
        raise ValueError(f"Expected {height * width} tokens for resolution {resolution}, got {token_count}.")
    return tokens.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()


def _map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = feature_map.shape
    return feature_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)


def _validate_multiscale_resolution(resolution: tuple[int, int], num_layers: int) -> None:
    if num_layers < 1:
        raise ValueError("VMRNN requires at least one recurrent scale.")

    divisor = 2 ** num_layers
    height, width = resolution
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution {resolution} must be divisible by {divisor} to support {num_layers} downsampling stages."
        )


def resolve_vss_backend(requested: str) -> str:
    normalized = requested.lower()
    if normalized not in {"auto", "formal", "vmamba"}:
        raise ValueError(f"Unsupported VSS backend: {requested}")
    if not is_vmamba_kernel_available():
        raise RuntimeError(f"Formal VMRNN requires the VMamba backend: {vmamba_unavailable_reason()}")
    return "vmamba"


def _extract_released_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a state-dict like checkpoint, found {type(checkpoint)!r}.")
    return checkpoint


def _map_released_weight_key(key: str) -> str | None:
    normalized = key.removeprefix("module.")
    replacements = (
        ("Downsample.layers.", "down_cells."),
        ("Downsample.downsample.", "downsamplers."),
        ("Upsample.layers.", "up_cells."),
        ("Upsample.upsample.", "upsamplers."),
        (".VSBs.", ".blocks."),
    )
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    if normalized.startswith(("down_cells.", "up_cells.", "downsamplers.", "upsamplers.")):
        return normalized
    return None


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dim: int, norm_layer: type[nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch_size, token_count, channels = tokens.shape
        if token_count != height * width:
            raise ValueError(f"Expected {height * width} tokens for resolution {self.input_resolution}, got {token_count}.")
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"PatchMerging expects even resolutions, got {self.input_resolution}.")

        feature_map = tokens.view(batch_size, height, width, channels)
        x0 = feature_map[:, 0::2, 0::2, :]
        x1 = feature_map[:, 1::2, 0::2, :]
        x2 = feature_map[:, 0::2, 1::2, :]
        x3 = feature_map[:, 1::2, 1::2, :]
        merged = torch.cat([x0, x1, x2, x3], dim=-1)
        merged = merged.view(batch_size, -1, 4 * channels)
        merged = self.norm(merged)
        return self.reduction(merged)


class PatchExpanding(nn.Module):
    def __init__(
        self,
        input_resolution: tuple[int, int],
        dim: int,
        dim_scale: int = 2,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        tokens = self.expand(tokens)
        batch_size, token_count, channels = tokens.shape
        if token_count != height * width:
            raise ValueError(f"Expected {height * width} tokens for resolution {self.input_resolution}, got {token_count}.")

        feature_map = tokens.view(batch_size, height, width, channels)
        feature_map = feature_map.reshape(batch_size, height, width, 2, 2, channels // 4)
        feature_map = feature_map.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, height * 2, width * 2, channels // 4)
        tokens = feature_map.view(batch_size, -1, channels // 4)
        return self.norm(tokens)


class InputTokenProjection(nn.Module):
    """Map a fused exam embedding into the base token grid expected by VMRNN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        resolution: tuple[int, int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_count = _num_tokens(resolution)
        self.proj = nn.Linear(input_dim, self.token_count * hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(x_t).view(x_t.size(0), self.token_count, self.hidden_dim)
        return self.dropout(self.norm(tokens))


class RecurrentStageCell(nn.Module):
    """Per-scale recurrent VMRNN cell matching the reference hidden-state update flow."""

    def __init__(
        self,
        dim: int,
        resolution: tuple[int, int],
        depth: int,
        dropout: float = 0.0,
        vmamba_d_state: int = 16,
        vmamba_drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            VMambaVSBBlock(
                hidden_dim=dim,
                input_resolution=resolution,
                drop_path=vmamba_drop_path,
                attn_drop_rate=dropout,
                d_state=vmamba_d_state,
            )
            for _ in range(max(depth, 1))
        )

    def forward(
        self,
        tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden_tokens, cell_tokens = state
        outputs = []
        for block_index, block in enumerate(self.blocks):
            outputs.append(block(tokens if block_index == 0 else outputs[-1], hidden_tokens if block_index == 0 else None))

        stage_output = outputs[-1]
        gate = torch.sigmoid(stage_output)
        cell = torch.tanh(stage_output)
        next_cell = gate * (cell_tokens + cell)
        next_hidden = gate * torch.tanh(next_cell)
        return next_hidden, (next_hidden, next_cell)


class ReconstructionBlock(nn.Module):
    def __init__(self, dim: int, resolution: tuple[int, int]) -> None:
        super().__init__()
        self.resolution = resolution
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        feature_map = _tokens_to_map(self.norm(tokens), self.resolution)
        reconstructed = self.pointwise(self.activation(self.depthwise(feature_map)))
        return tokens + _map_to_tokens(reconstructed)


class VMRNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        spatial_resolution: tuple[int, int] = (4, 4),
        downsample_depths: Sequence[int] = (2, 6),
        upsample_depths: Sequence[int] | None = (6, 2),
        dropout: float = 0.1,
        vss_backend: str = "vmamba",
        vmamba_d_state: int = 16,
        vmamba_drop_path: float = 0.0,
        released_weight_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.spatial_resolution = spatial_resolution
        self.vss_backend = resolve_vss_backend(vss_backend)
        self.vmamba_kernel_backend = vmamba_kernel_backend_name()
        self.vmamba_d_state = vmamba_d_state
        self.vmamba_drop_path = vmamba_drop_path
        self.downsample_depths = tuple(downsample_depths)
        self.upsample_depths = tuple(upsample_depths if upsample_depths is not None else tuple(reversed(self.downsample_depths)))
        if len(self.downsample_depths) != len(self.upsample_depths):
            raise ValueError("Downsample and upsample depth lists must have the same number of stages.")

        self.num_layers = len(self.downsample_depths)
        _validate_multiscale_resolution(spatial_resolution, self.num_layers)

        self.base_token_count = _num_tokens(spatial_resolution)
        self.down_stage_resolutions = tuple(
            _scale_resolution(spatial_resolution, 2 ** stage_index) for stage_index in range(self.num_layers)
        )
        self.down_stage_dims = tuple(hidden_dim * (2 ** stage_index) for stage_index in range(self.num_layers))
        self.up_stage_resolutions = tuple(
            _scale_resolution(spatial_resolution, 2 ** stage_index)
            for stage_index in range(self.num_layers, 0, -1)
        )
        self.up_stage_dims = tuple(hidden_dim * (2 ** stage_index) for stage_index in range(self.num_layers, 0, -1))

        self.input_projection = InputTokenProjection(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            resolution=spatial_resolution,
            dropout=dropout,
        )

        self.down_cells = nn.ModuleList(
            RecurrentStageCell(
                dim=self.down_stage_dims[stage_index],
                resolution=self.down_stage_resolutions[stage_index],
                depth=self.downsample_depths[stage_index],
                dropout=dropout,
                vmamba_d_state=self.vmamba_d_state,
                vmamba_drop_path=self.vmamba_drop_path,
            )
            for stage_index in range(self.num_layers)
        )
        self.downsamplers = nn.ModuleList(
            PatchMerging(
                input_resolution=self.down_stage_resolutions[stage_index],
                dim=self.down_stage_dims[stage_index],
            )
            for stage_index in range(self.num_layers)
        )

        self.up_cells = nn.ModuleList(
            RecurrentStageCell(
                dim=self.up_stage_dims[stage_index],
                resolution=self.up_stage_resolutions[stage_index],
                depth=self.upsample_depths[self.num_layers - 1 - stage_index],
                dropout=dropout,
                vmamba_d_state=self.vmamba_d_state,
                vmamba_drop_path=self.vmamba_drop_path,
            )
            for stage_index in range(self.num_layers)
        )
        self.upsamplers = nn.ModuleList(
            PatchExpanding(
                input_resolution=self.up_stage_resolutions[stage_index],
                dim=self.up_stage_dims[stage_index],
            )
            for stage_index in range(self.num_layers)
        )

        self.reconstruction = ReconstructionBlock(dim=hidden_dim, resolution=spatial_resolution)
        self.history_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reconstruction_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )
        self.released_weight_report: dict[str, object] | None = None
        if released_weight_path is not None:
            released_path = Path(released_weight_path)
            if not released_path.exists():
                raise FileNotFoundError(f"VMRNN released weights were requested but not found: {released_path}")
            self.released_weight_report = self._load_released_weights(released_path)

    def _load_released_weights(self, path: Path) -> dict[str, object]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        raw_state = _extract_released_state_dict(checkpoint)
        current_state = self.state_dict()
        filtered_state: dict[str, torch.Tensor] = {}
        for key, value in raw_state.items():
            if not isinstance(value, torch.Tensor):
                continue
            mapped_key = _map_released_weight_key(key)
            if mapped_key is None or mapped_key not in current_state:
                continue
            if current_state[mapped_key].shape != value.shape:
                continue
            filtered_state[mapped_key] = value

        if not filtered_state:
            raise RuntimeError(f"No compatible VMRNN weights could be loaded from released checkpoint: {path}")

        incompatible = self.load_state_dict(filtered_state, strict=False)
        return {
            "path": str(path),
            "loaded_tensors": len(filtered_state),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }

    def _init_state_list(
        self,
        batch_size: int,
        specs: Sequence[tuple[tuple[int, int], int]],
        reference: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        states = []
        for resolution, dim in specs:
            token_count = _num_tokens(resolution)
            zeros = reference.new_zeros(batch_size, token_count, dim)
            states.append((zeros.clone(), zeros.clone()))
        return states

    def _blend_state_list(
        self,
        new_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
        old_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
        active: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        blended = []
        for (new_hidden, new_cell), (old_hidden, old_cell) in zip(new_states, old_states, strict=True):
            blended.append(
                (
                    new_hidden * active + old_hidden * (1.0 - active),
                    new_cell * active + old_cell * (1.0 - active),
                )
            )
        return blended

    def _run_down_path(
        self,
        tokens: torch.Tensor,
        states_down: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        next_states = []
        for stage_index, cell in enumerate(self.down_cells):
            tokens, state = cell(tokens, states_down[stage_index])
            next_states.append(state)
            tokens = self.downsamplers[stage_index](tokens)
        return next_states, tokens

    def _run_up_path(
        self,
        tokens: torch.Tensor,
        states_up: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        next_states = []
        for stage_index, cell in enumerate(self.up_cells):
            tokens, state = cell(tokens, states_up[stage_index])
            next_states.append(state)
            tokens = self.upsamplers[stage_index](tokens)
        return next_states, tokens

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)

    def forward(self, sequence: torch.Tensor, exam_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, time_steps, _ = sequence.shape
        down_specs = tuple(zip(self.down_stage_resolutions, self.down_stage_dims, strict=True))
        up_specs = tuple(zip(self.up_stage_resolutions, self.up_stage_dims, strict=True))
        states_down = self._init_state_list(batch_size, down_specs, sequence)
        states_up = self._init_state_list(batch_size, up_specs, sequence)
        summary_tokens = sequence.new_zeros(batch_size, self.base_token_count, self.hidden_dim)

        pooled_states = []
        pooled_reconstructions = []

        for time_index in range(time_steps):
            tokens = self.input_projection(sequence[:, time_index])
            next_states_down, latent_tokens = self._run_down_path(tokens, states_down)
            next_states_up, upsampled_tokens = self._run_up_path(latent_tokens, states_up)
            reconstructed_tokens = self.reconstruction(upsampled_tokens)

            active = exam_mask[:, time_index].view(batch_size, 1, 1).to(sequence.dtype)
            states_down = self._blend_state_list(next_states_down, states_down, active)
            states_up = self._blend_state_list(next_states_up, states_up, active)
            summary_tokens = reconstructed_tokens * active + summary_tokens * (1.0 - active)

            pooled_state = self.history_projection(self._pool_tokens(summary_tokens))
            pooled_states.append(pooled_state)

            pooled_reconstruction = self.reconstruction_projection(self._pool_tokens(reconstructed_tokens))
            pooled_reconstructions.append(pooled_reconstruction * active.squeeze(-1))

        state_tensor = torch.stack(pooled_states, dim=1)
        reconstruction_tensor = torch.stack(pooled_reconstructions, dim=1)
        last_indices = exam_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_index = torch.arange(batch_size, device=sequence.device)
        history_embedding = state_tensor[batch_index, last_indices]
        return history_embedding, state_tensor, reconstruction_tensor
