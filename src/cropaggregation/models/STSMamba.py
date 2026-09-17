"""Spatial-Temporal-Spectral Mamba classifier for multitemporal raster patches.

Input shape: [batch, time, patch_height, patch_width, channels].
The implementation is dependency-free PyTorch and uses input-selective state
space scans in three explicit branches. It does not require mamba-ssm/CUDA
selective-scan extensions, which keeps the project runnable on Windows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, expansion=2, dropout=0.1):
        super().__init__()
        self.d_inner = int(d_model * expansion)
        self.d_state = int(d_state)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.param_proj = nn.Linear(self.d_inner, 1 + 2 * self.d_state)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1))
        self.d_skip = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = F.silu(u)
        params = self.param_proj(u)
        dt, b_in, c_in = torch.split(params, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt).clamp(max=1.0)
        a = -torch.exp(self.a_log).to(dtype=x.dtype, device=x.device)
        state = x.new_zeros(x.shape[0], self.d_inner, self.d_state)
        outputs = []
        for index in range(x.shape[1]):
            decay = torch.exp(dt[:, index].unsqueeze(-1) * a.unsqueeze(0))
            state = decay * state + u[:, index].unsqueeze(-1) * b_in[:, index].unsqueeze(1)
            y = (state * c_in[:, index].unsqueeze(1)).sum(-1)
            y = y + self.d_skip * u[:, index]
            outputs.append(y * torch.sigmoid(gate[:, index]))
        y = torch.stack(outputs, dim=1)
        return residual + self.dropout(self.out_proj(y))


class BidirectionalMamba(nn.Module):
    def __init__(self, d_model, d_state=16, expansion=2, dropout=0.1):
        super().__init__()
        self.forward_ssm = SelectiveSSM(d_model, d_state, expansion, dropout)
        self.backward_ssm = SelectiveSSM(d_model, d_state, expansion, dropout)
        self.fuse = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        forward = self.forward_ssm(x)
        backward = torch.flip(self.backward_ssm(torch.flip(x, dims=[1])), dims=[1])
        return self.norm(self.fuse(torch.cat([forward, backward], dim=-1)))


class STSMamba(nn.Module):
    """Three-branch spatial, temporal and spectral selective SSM model."""

    requires_spatial_patch = True

    def __init__(self, input_dim, num_classes, seq_len, patch_size=3,
                 d_model=64, d_state=16, depth=2, dropout=0.15, **kwargs):
        super().__init__()
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        if self.patch_size < 3 or self.patch_size % 2 == 0:
            raise ValueError("STSMamba patch_size must be an odd integer >= 3")

        self.temporal_embed = nn.Linear(input_dim, d_model)
        self.spatial_embed = nn.Linear(input_dim, d_model)
        self.spectral_embed = nn.Linear(1, d_model)
        self.temporal_blocks = nn.ModuleList([
            BidirectionalMamba(d_model, d_state, dropout=dropout) for _ in range(depth)
        ])
        self.spatial_blocks = nn.ModuleList([
            BidirectionalMamba(d_model, d_state, dropout=dropout) for _ in range(depth)
        ])
        self.spectral_blocks = nn.ModuleList([
            BidirectionalMamba(d_model, d_state, dropout=dropout) for _ in range(depth)
        ])
        self.branch_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(),
            nn.Linear(d_model, 3), nn.Softmax(dim=-1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes)
        )

    @staticmethod
    def _run(tokens, blocks):
        for block in blocks:
            tokens = block(tokens)
        return tokens.mean(dim=1)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"STSMamba expects [B,T,H,W,C], got {tuple(x.shape)}")
        _, _, height, width, channels = x.shape
        if height != self.patch_size or width != self.patch_size or channels != self.input_dim:
            raise ValueError(
                f"Expected patch [T,{self.patch_size},{self.patch_size},{self.input_dim}], got {tuple(x.shape[1:])}"
            )
        temporal = self.temporal_embed(x.mean(dim=(2, 3)))
        spatial = self.spatial_embed(x.mean(dim=1).reshape(x.shape[0], height * width, channels))
        spectral = self.spectral_embed(x.mean(dim=(1, 2, 3)).unsqueeze(-1))
        temporal = self._run(temporal, self.temporal_blocks)
        spatial = self._run(spatial, self.spatial_blocks)
        spectral = self._run(spectral, self.spectral_blocks)
        branches = torch.stack([temporal, spatial, spectral], dim=1)
        gates = self.branch_gate(torch.cat([temporal, spatial, spectral], dim=-1))
        fused = (branches * gates.unsqueeze(-1)).sum(dim=1)
        return self.classifier(fused)
