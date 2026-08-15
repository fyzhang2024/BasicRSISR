"""LR-only image-level router for SCDR-RADR."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class FixedStructureStats(nn.Module):
    """Extract fixed LR-only structure statistics per image."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        lap = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("lap", lap)

    def forward(self, lr):
        if lr.dim() != 4:
            raise ValueError("FixedStructureStats expects B,C,H,W, got {}".format(tuple(lr.shape)))
        lr = lr.float().clamp(0.0, 1.0)
        if lr.shape[1] == 3:
            gray = 0.299 * lr[:, 0:1] + 0.587 * lr[:, 1:2] + 0.114 * lr[:, 2:3]
        else:
            gray = lr.mean(dim=1, keepdim=True)

        gx = F.conv2d(gray, self.sobel_x.to(dtype=gray.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(dtype=gray.dtype), padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        edge_density = mag.mean(dim=(1, 2, 3))
        flat = mag.flatten(1)
        topk = max(1, int(round(0.25 * flat.shape[1])))
        edge_top25_mean = flat.topk(topk, dim=1).values.mean(dim=1)

        lap = F.conv2d(gray, self.lap.to(dtype=gray.dtype), padding=1)
        lap_var = lap.flatten(1).var(dim=1, unbiased=False)
        jxx = (gx * gx).mean(dim=(1, 2, 3))
        jyy = (gy * gy).mean(dim=(1, 2, 3))
        jxy = (gx * gy).mean(dim=(1, 2, 3))
        grad_coherence = torch.sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy + 1e-6) / (jxx + jyy + 1e-6)
        blur = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        hf_energy = (gray - blur).abs().mean(dim=(1, 2, 3))
        return torch.stack([edge_density, edge_top25_mean, lap_var, grad_coherence, hf_energy], dim=1)


class SCDRImageRouter(nn.Module):
    """Predict image-level alpha from LR-only fixed structure statistics."""

    def __init__(self, hidden_dim=32, init_alpha=0.49):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.init_alpha = float(init_alpha)
        init_alpha = min(max(float(init_alpha), 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init_alpha / (1.0 - init_alpha))
        self.extractor = FixedStructureStats()
        self.net = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.reset_parameters(init_logit)

    def reset_parameters(self, init_logit):
        nn.init.kaiming_uniform_(self.net[1].weight, a=math.sqrt(5))
        if self.net[1].bias is not None:
            nn.init.zeros_(self.net[1].bias)
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.constant_(self.net[-1].bias, float(init_logit))

    def forward(self, lr):
        stats = self.extractor(lr)
        logit = self.net(stats)
        alpha = torch.sigmoid(logit).view(lr.shape[0], 1, 1, 1)
        return alpha

