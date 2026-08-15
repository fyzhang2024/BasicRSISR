'''
An official Pytorch impl of `ATD: Improved Transformer with
Adaptive Token Dictionary for Image Restoration`.

Arxiv: 'https://arxiv.org/abs/2401.08209'
'''

import math
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint

from basicsr.utils.registry import ARCH_REGISTRY


# Shuffle operation for Categorization and UnCategorization operations.
def index_reverse(index):
    index_r = torch.zeros_like(index)
    
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def feature_shuffle(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    # match the shape of x and index
    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim-1, index=index)
    return shuffled_x


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN_td(nn.Module):
    def __init__(self, in_features, hidden_features=None, td_features=0, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        hidden_features += td_features
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_td, x_size):
        x = self.fc1(x)
        x = torch.cat([self.act(x), x_td], dim=-1)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x
    
    
class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

# FDG: Frequency-Decoupled Grouping
# AC MSA grouping/category ATD CA read out # ====================================================================


class FrequencyMapBuilder(nn.Module):
    # 构造 FDG 使用的轻量频率图，输出 (B,K+1,H,W)。

    def __init__(self, out_channels=5, detach=True, eps=1e-6):
        super().__init__()
        self.out_channels = int(out_channels)
        self.detach = bool(detach)
        self.eps = float(eps)
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer("lap_kernel", lap)

    def forward(self, x):
        ctx = torch.no_grad() if self.detach else contextlib.nullcontext()
        with ctx:
            if x.shape[1] == 3:
                gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
            else:
                gray = x.mean(dim=1, keepdim=True)
            gray = gray.float()
            low = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
            high = (gray - low).abs()
            dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
            dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
            grad = torch.sqrt(dx * dx + dy * dy + self.eps)
            lap = F.conv2d(gray, self.lap_kernel.to(dtype=gray.dtype, device=gray.device), padding=1).abs()
            var = F.avg_pool2d((gray - low) * (gray - low), kernel_size=3, stride=1, padding=1)
            freq_map = torch.cat([gray, high, grad, lap, var], dim=1)
            if self.out_channels < freq_map.shape[1]:
                freq_map = freq_map[:, :self.out_channels, :, :]
            elif self.out_channels > freq_map.shape[1]:
                pad = freq_map.new_zeros(freq_map.shape[0], self.out_channels - freq_map.shape[1], freq_map.shape[2], freq_map.shape[3])
                freq_map = torch.cat([freq_map, pad], dim=1)
            return freq_map.detach() if self.detach else freq_map


class FixedStructureExtractor(nn.Module):
    """SCDRC-Lite 固定结构统计提取器，不包含可学习参数。"""

    def __init__(self, eps=1e-6):
        super().__init__()
        # 中文注释：数值稳定项，避免 sqrt 或除法出现 0。
        self.eps = float(eps)
        # 中文注释：Sobel x 固定卷积核，用 register_buffer 保存以支持 GPU/DataParallel。
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        # 中文注释：Sobel y 固定卷积核，用于计算垂直方向梯度。
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        # 中文注释：Laplacian 固定卷积核，用于统计局部二阶变化。
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        # 中文注释：注册 Sobel x buffer，不参与优化。
        self.register_buffer("sobel_x", sobel_x)
        # 中文注释：注册 Sobel y buffer，不参与优化。
        self.register_buffer("sobel_y", sobel_y)
        # 中文注释：注册 Laplacian buffer，不参与优化。
        self.register_buffer("lap_kernel", lap)

    def forward(self, lr):
        # 中文注释：输入 lr 为 B,3,H,W，期望范围 [0,1]；若通道不是 3，则退化为均值灰度。
        if lr.shape[1] == 3:
            # 中文注释：按常用 RGB 权重转灰度。
            gray = 0.299 * lr[:, 0:1, :, :] + 0.587 * lr[:, 1:2, :, :] + 0.114 * lr[:, 2:3, :, :]
        else:
            # 中文注释：非 RGB 输入时用通道均值作为灰度。
            gray = lr.mean(dim=1, keepdim=True)
        # 中文注释：结构统计使用 float32，避免 AMP 半精度下方差/排序不稳定。
        gray = gray.float()
        # 中文注释：固定卷积核转到输入设备和 dtype。
        sobel_x = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
        # 中文注释：固定卷积核转到输入设备和 dtype。
        sobel_y = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
        # 中文注释：固定卷积核转到输入设备和 dtype。
        lap_kernel = self.lap_kernel.to(device=gray.device, dtype=gray.dtype)
        # 中文注释：计算 x/y 方向梯度。
        gx = F.conv2d(gray, sobel_x, padding=1)
        # 中文注释：计算 x/y 方向梯度。
        gy = F.conv2d(gray, sobel_y, padding=1)
        # 中文注释：梯度幅值。
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        # 中文注释：edge_density 使用梯度幅值均值，避免分位阈值比例恒定。
        edge_density = mag.mean(dim=(1, 2, 3))
        # 中文注释：top-25% 梯度均值刻画强边缘结构。
        mag_flat = mag.flatten(1)
        # 中文注释：至少取 1 个像素，避免极小图像 topk 为 0。
        topk = max(1, int(math.ceil(0.25 * mag_flat.shape[1])))
        # 中文注释：沿每张图自己的像素维度取 top-25% 梯度。
        edge_top25_mean = mag_flat.topk(topk, dim=1).values.mean(dim=1)
        # 中文注释：Laplacian 响应。
        lap = F.conv2d(gray, lap_kernel, padding=1)
        # 中文注释：Laplacian 方差。
        lap_var = lap.flatten(1).var(dim=1, unbiased=False)
        # 中文注释：结构张量 Jxx。
        jxx = (gx * gx).mean(dim=(1, 2, 3))
        # 中文注释：结构张量 Jyy。
        jyy = (gy * gy).mean(dim=(1, 2, 3))
        # 中文注释：结构张量 Jxy。
        jxy = (gx * gy).mean(dim=(1, 2, 3))
        # 中文注释：方向一致性，使用结构张量闭式公式。
        grad_coherence = torch.sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy + self.eps) / (jxx + jyy + self.eps)
        # 中文注释：3x3 平均池化估计低频背景。
        blur = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        # 中文注释：高频能量。
        hf_energy = (gray - blur).abs().mean(dim=(1, 2, 3))
        # 中文注释：返回 B,5，字段顺序固定供日志和 CSV 解析。
        return torch.stack([edge_density, edge_top25_mean, lap_var, grad_coherence, hf_energy], dim=1)


class FixedLocalStructureExtractor(nn.Module):
    """提取局部结构先验图，用于 SCDRC-Local。"""

    def __init__(self):
        super().__init__()
        # 中文注释：固定 Sobel x 卷积核，和 FixedStructureExtractor 保持一致。
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        # 中文注释：固定 Sobel y 卷积核，和 FixedStructureExtractor 保持一致。
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        # 中文注释：固定 Laplacian 卷积核，刻画局部二阶结构。
        lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]]
        ).view(1, 1, 3, 3)
        # 中文注释：注册 Sobel x buffer，不参与优化。
        self.register_buffer("sobel_x", sobel_x)
        # 中文注释：注册 Sobel y buffer，不参与优化。
        self.register_buffer("sobel_y", sobel_y)
        # 中文注释：注册 Laplacian buffer，不参与优化。
        self.register_buffer("lap_kernel", lap_kernel)

    def forward(self, x):
        """返回局部结构图。"""
        # 中文注释：输入 x 为 B,C,H,W，期望范围 [0,1]；非 RGB 时退化为通道均值。
        if x.shape[1] == 3:
            # 中文注释：按常用 RGB 权重转灰度。
            gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        else:
            # 中文注释：非 RGB 输入时用通道均值作为灰度。
            gray = x.mean(dim=1, keepdim=True)
        # 中文注释：局部结构图使用 float32，避免 AMP 半精度下边缘图数值不稳定。
        gray = gray.float()
        # 中文注释：固定卷积核转到输入设备和 dtype。
        sobel_x = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
        # 中文注释：固定卷积核转到输入设备和 dtype。
        sobel_y = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
        # 中文注释：固定卷积核转到输入设备和 dtype。
        lap_kernel = self.lap_kernel.to(device=gray.device, dtype=gray.dtype)
        # 中文注释：计算 Sobel x 梯度。
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        # 中文注释：计算 Sobel y 梯度。
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        # 中文注释：局部边缘强度图。
        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
        # 中文注释：局部 Laplacian 响应取绝对值。
        lap = torch.abs(F.conv2d(gray, lap_kernel, padding=1))
        # 中文注释：3x3 平均池化估计低频背景。
        smooth = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        # 中文注释：局部高频图。
        hf = torch.abs(gray - smooth)
        # 中文注释：返回 B,4,H,W，字段顺序固定：gray, edge, lap, hf。
        return torch.cat([
            gray,
            torch.log1p(edge),
            torch.log1p(lap),
            torch.log1p(hf),
        ], dim=1)


class LocalStructureResidualController(nn.Module):
    """根据局部结构图生成局部 delta map。"""

    def __init__(self, in_channels=4, hidden_channels=16, scale_max=0.05, last_weight_std=0.003):
        super().__init__()
        # 中文注释：输入通道数，第一版固定为 gray/edge/lap/hf 四通道。
        self.in_channels = int(in_channels)
        # 中文注释：局部控制器隐藏通道数。
        self.hidden_channels = int(hidden_channels)
        # 中文注释：local delta map 最大幅度，默认比 global scale 更小。
        self.scale_max = float(scale_max)
        # 中文注释：最后一层 conv 的小随机初始化标准差。
        self.last_weight_std = float(last_weight_std)
        # 中文注释：轻量 3x3 conv controller，只输出 1 通道局部 delta map。
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=3, padding=1),
        )
        # 中文注释：初始化最后一层，使 local delta 初始很小但非完全常数。
        self.reset_last_layer_init()

    def reset_last_layer_init(self):
        """初始化 local controller，使 local delta 初始很小但有空间差异。"""
        # 中文注释：第一层正常 Kaiming 初始化。
        if isinstance(self.net[0], nn.Conv2d):
            nn.init.kaiming_normal_(self.net[0].weight, mode="fan_out", nonlinearity="relu")
            if self.net[0].bias is not None:
                nn.init.zeros_(self.net[0].bias)
        # 中文注释：最后一层用小随机初始化；std=0 时退化为全零初始 delta map。
        if isinstance(self.net[-1], nn.Conv2d):
            if self.last_weight_std > 0.0:
                nn.init.normal_(self.net[-1].weight, mean=0.0, std=self.last_weight_std)
            else:
                nn.init.zeros_(self.net[-1].weight)
            if self.net[-1].bias is not None:
                nn.init.zeros_(self.net[-1].bias)

    def forward(self, local_struct_map, target_hw=None):
        """输出 B,1,h,w 的 local delta map。"""
        # 中文注释：局部结构图必须是 B,4,H,W。
        if local_struct_map.dim() != 4 or local_struct_map.shape[1] != self.in_channels:
            raise ValueError("local_struct_map must have shape B,{},H,W, got {}".format(
                self.in_channels, tuple(local_struct_map.shape)
            ))
        # 中文注释：tanh 限幅，避免局部 residual calibration 过强。
        delta_map = self.scale_max * torch.tanh(self.net(local_struct_map))
        # 中文注释：如果 TDCA residual 空间尺寸不同，则双线性插值对齐。
        if target_hw is not None and tuple(delta_map.shape[-2:]) != tuple(target_hw):
            delta_map = F.interpolate(delta_map, size=target_hw, mode="bilinear", align_corners=False)
        # 中文注释：返回 B,1,h,w。
        return delta_map


class StructureResidualController(nn.Module):
    """SCDRC-Lite / SCDRC-RR 图像级 dictionary residual calibration 控制器。"""

    def __init__(self, hidden_dim=64, scale_max=0.15, delta_init=0.0, last_weight_std=0.0, input_dim=5):
        super().__init__()
        # 中文注释：普通 SCDRC 输入 5 维；SCDRC-RR 输入 5 维结构 + 3 维 residual reliability。
        self.input_dim = int(input_dim)
        # 中文注释：每张图 controller 输入内部归一化，不依赖训练集统计。
        self.norm = nn.LayerNorm(self.input_dim)
        # 中文注释：轻量 MLP 输出共享图像级 scalar delta。
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        # 中文注释：最大绝对 residual calibration scale。
        self.scale_max = float(scale_max)
        # 中文注释：初始 residual calibration delta；默认 0 时等价当前 baseline。
        self.delta_init = float(delta_init)
        # 中文注释：最后一层 weight 的小随机初始化标准差；>0 时初始 delta 会随结构输入有轻微图间差异。
        self.last_weight_std = float(last_weight_std)
        # 中文注释：初始化最后一层，使 MLP 初始输出对应指定 delta。
        self.reset_last_layer_init()

    def reset_last_layer_init(self):
        """重新初始化最后一层，抵消 ATD 全局初始化对 SCDRC init 的覆盖。"""
        # 中文注释：最后一层 weight 决定不同结构输入是否能产生不同 delta。
        # 中文注释：zero-init 会让所有图像输出同一个 delta，导致 delta_std=0，退化成全局旋钮。
        # 中文注释：因此这里支持小随机初始化，让 delta 一开始就有图间差异。
        if self.last_weight_std > 0.0:
            nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=self.last_weight_std)
        else:
            nn.init.zeros_(self.mlp[-1].weight)
        # 中文注释：把 delta_init / scale_max 限制到 (-0.99, 0.99)，避免 atanh 数值异常。
        # 中文注释：bias 用于保留 struct_delta_init 这个共同软启动基线。
        # 中文注释：delta = scale_max * tanh(raw_delta)，所以 bias = atanh(delta_init / scale_max)。
        init_ratio = self.delta_init / max(self.scale_max, 1e-8)
        init_ratio = max(min(float(init_ratio), 0.99), -0.99)
        # 中文注释：atanh(x)=0.5*log((1+x)/(1-x))。
        init_bias = 0.5 * math.log((1.0 + init_ratio) / (1.0 - init_ratio))
        # 中文注释：bias 设置为对应的 raw delta。
        nn.init.constant_(self.mlp[-1].bias, float(init_bias))

    def forward(self, struct_vec, residual_vec=None):
        """根据结构统计和可选 TDCA residual 可靠性统计输出 delta。"""
        # 中文注释：struct_vec: B,5，来自 LR 固定结构统计。
        if struct_vec.dim() != 2 or struct_vec.shape[1] != 5:
            raise ValueError("struct_vec must have shape B,5, got {}".format(tuple(struct_vec.shape)))
        # 中文注释：对原始结构统计做 log 压缩，缓解尺度差异。
        struct_log = torch.log1p(struct_vec)
        if residual_vec is not None:
            # 中文注释：residual_vec: B,3，包含 TDCA residual 的强度和相对比例。
            if residual_vec.dim() != 2 or residual_vec.shape[1] != 3:
                raise ValueError("residual_vec must have shape B,3, got {}".format(tuple(residual_vec.shape)))
            # 中文注释：residual 统计非负，log1p 压缩尺度；clamp 避免数值误差出现负数。
            residual_log = torch.log1p(torch.clamp(residual_vec, min=0.0))
            # 中文注释：SCDRC-RR 输入为 5 个结构指标 + 3 个 residual reliability 指标。
            controller_input = torch.cat([struct_log, residual_log], dim=1)
        else:
            # 中文注释：普通 SCDRC 只使用 5 个结构指标。
            controller_input = struct_log
        # 中文注释：显式检查输入维度，避免 5/8 维模式 silent mismatch。
        if controller_input.shape[1] != self.input_dim:
            raise ValueError(
                "SCDRC controller input dim mismatch: got {}, expected {}".format(
                    int(controller_input.shape[1]), int(self.input_dim)
                )
            )
        # 中文注释：LayerNorm 只给 MLP 使用，不用它的均值作为 structure_score。
        controller_norm = self.norm(controller_input)
        # 中文注释：输出共享图像级 delta，第一版不做 per-layer delta。
        raw_delta = self.mlp(controller_norm)
        # 中文注释：tanh 限幅，避免 residual calibration 过强。
        delta = self.scale_max * torch.tanh(raw_delta)
        # 中文注释：structure_score 基于原始 log 结构强度，避免 LayerNorm 后均值恒接近 0。
        structure_score = (
            struct_log[:, 0:1]
            + struct_log[:, 1:2]
            + struct_log[:, 2:3]
            + struct_log[:, 4:5]
        ) / 4.0
        # 中文注释：返回 delta、结构分数和 controller 归一化输入。
        return delta, structure_score, controller_norm


class LocalReliabilityPredictor(nn.Module):
    """预测局部 dictionary residual 的不可靠概率。"""

    def __init__(self, dim, hidden_dim=32, out_scale="token", init_bias=-4.0):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.out_scale = str(out_scale)
        self.init_bias = float(init_bias)
        if self.out_scale == "token":
            self.net = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )
        elif self.out_scale == "spatial":
            self.net = nn.Sequential(
                nn.Conv2d(self.dim, self.hidden_dim, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, 1, kernel_size=1, stride=1, padding=0),
            )
        else:
            raise ValueError("Unsupported LocalReliabilityPredictor out_scale: {}".format(self.out_scale))
        self.reset_parameters()

    def reset_parameters(self):
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.constant_(last.bias, self.init_bias)

    def forward(self, feat):
        if self.out_scale == "token":
            if feat.dim() != 3 or feat.shape[-1] != self.dim:
                raise ValueError("RADR token predictor expects B,N,{} got {}".format(self.dim, tuple(feat.shape)))
            u_logit = self.net(feat)
        else:
            if feat.dim() != 4 or feat.shape[1] != self.dim:
                raise ValueError("RADR spatial predictor expects B,{},H,W got {}".format(self.dim, tuple(feat.shape)))
            u_logit = self.net(feat)
        u_hat = torch.sigmoid(u_logit)
        return u_hat, u_logit


class ReliabilityCorrectionBranch(nn.Module):
    """Generate a lightweight correction residual for unreliable dictionary residuals."""

    def __init__(
        self,
        dim,
        hidden_dim=64,
        corr_scale=0.10,
        init_std=1e-4,
        detach_residual=True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.corr_scale = float(corr_scale)
        self.init_std = float(init_std)
        self.detach_residual = bool(detach_residual)
        self.net = nn.Sequential(
            nn.LayerNorm(self.dim * 2),
            nn.Linear(self.dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        last = self.net[-1]
        nn.init.normal_(last.weight, mean=0.0, std=self.init_std)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, shortcut, x_atd):
        if shortcut.shape != x_atd.shape:
            raise RuntimeError(
                "RADR correction shape mismatch: shortcut {} vs x_atd {}".format(
                    tuple(shortcut.shape), tuple(x_atd.shape)
                )
            )
        if shortcut.dim() != 3:
            raise ValueError(
                "RADR correction currently expects token feature B,N,C, got {}".format(
                    tuple(shortcut.shape)
                )
            )
        residual = x_atd.detach() if self.detach_residual else x_atd
        feat = torch.cat([shortcut, residual], dim=-1)
        x_corr = self.corr_scale * torch.tanh(self.net(feat))
        return x_corr


class RADRCorrectionAuxHead(nn.Module):
    """Training-only auxiliary head: maps correction feature tokens to HR RGB residual."""

    def __init__(self, dim, upscale=4, init_std=1e-4, out_scale=0.10):
        super().__init__()
        self.dim = int(dim)
        self.upscale = int(upscale)
        self.out_channels = 3 * self.upscale * self.upscale
        self.init_std = float(init_std)
        self.out_scale = float(out_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.out_channels),
        )
        self.reset_parameters()

    def reset_parameters(self):
        last = self.net[-1]
        nn.init.normal_(last.weight, mean=0.0, std=self.init_std)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, feat, x_size):
        if feat.dim() != 3:
            raise ValueError("RADRCorrectionAuxHead expects B,N,C, got {}".format(tuple(feat.shape)))
        b, n, _ = feat.shape
        h, w = int(x_size[0]), int(x_size[1])
        if n != h * w:
            raise RuntimeError(
                "RADRCorrectionAuxHead token mismatch: N={} vs H*W={} for x_size={}".format(
                    int(n), int(h * w), x_size
                )
            )
        out = self.net(feat)
        out = out.transpose(1, 2).contiguous().view(b, self.out_channels, h, w)
        out = F.pixel_shuffle(out, self.upscale)
        return self.out_scale * torch.tanh(out)


class RADRLateCorrectionHead(nn.Module):
    """Late feature correction head gated by RADR unreliable map."""

    def __init__(
        self,
        dim,
        hidden_dim=64,
        init_std=1e-4,
        corr_scale=0.05,
        use_map=True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.init_std = float(init_std)
        self.corr_scale = float(corr_scale)
        self.use_map = bool(use_map)

        in_channels = self.dim + (1 if self.use_map else 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.dim, kernel_size=3, padding=1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.net[0], nn.Conv2d):
            nn.init.kaiming_normal_(self.net[0].weight, mode="fan_out", nonlinearity="relu")
            if self.net[0].bias is not None:
                nn.init.zeros_(self.net[0].bias)
        last = self.net[-1]
        nn.init.normal_(last.weight, mean=0.0, std=self.init_std)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, feat, u_eff_map=None):
        if feat.dim() != 4:
            raise ValueError("RADRLateCorrectionHead expects B,C,H,W feat, got {}".format(tuple(feat.shape)))
        if self.use_map:
            if u_eff_map is None:
                raise RuntimeError("u_eff_map is required when RADRLateCorrectionHead.use_map=True")
            if u_eff_map.shape[-2:] != feat.shape[-2:]:
                u_eff_map = F.interpolate(u_eff_map.float(), size=feat.shape[-2:], mode="bilinear", align_corners=False)
            u_eff_map = u_eff_map.to(device=feat.device, dtype=feat.dtype).clamp(0.0, 1.0)
            x = torch.cat([feat, u_eff_map], dim=1)
        else:
            x = feat
        return self.corr_scale * torch.tanh(self.net(x))


class SCDRRouteAdapter(nn.Module):
    """SCDR-v2 route-specific LR-feature adapter."""

    def __init__(self, dim, hidden_dim=32, scale=0.05, init_std=1e-5):
        super().__init__()
        self.scale = float(scale)
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=3, padding=1),
        )
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std):
        nn.init.kaiming_normal_(self.net[0].weight, mode="fan_out", nonlinearity="relu")
        if self.net[0].bias is not None:
            nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, mean=0.0, std=float(init_std))
        if self.net[2].bias is not None:
            nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return self.scale * self.net(x)


def compute_residual_reliability_stats(tdca_out, feat_ref, eps=1e-6):
    """计算每张图的 TDCA residual 可靠性统计。"""
    # 中文注释：tdca_out 与 feat_ref 必须处于同一表示格式，例如 B,N,C 或 B,C,H,W。
    if tdca_out.shape != feat_ref.shape:
        raise ValueError("tdca_out and feat_ref must have the same shape, got {} vs {}".format(
            tuple(tdca_out.shape), tuple(feat_ref.shape)
        ))
    # 中文注释：B,N,C 情况，对 token 和 channel 维统计。
    if tdca_out.dim() == 3:
        reduce_dims = (1, 2)
    # 中文注释：B,C,H,W 情况，对 channel 和空间维统计。
    elif tdca_out.dim() == 4:
        reduce_dims = (1, 2, 3)
    else:
        raise ValueError("Unsupported tdca_out dim: {}".format(tdca_out.dim()))
    # 中文注释：残差绝对强度。
    res_abs = tdca_out.abs()
    # 中文注释：残差绝对强度均值，B。
    res_abs_mean = res_abs.mean(dim=reduce_dims, keepdim=False)
    # 中文注释：残差绝对强度标准差，B。
    res_abs_std = res_abs.std(dim=reduce_dims, unbiased=False, keepdim=False)
    # 中文注释：参考特征强度，用于判断 residual 相对当前特征是否过强。
    feat_abs_mean = feat_ref.abs().mean(dim=reduce_dims, keepdim=False)
    # 中文注释：residual-to-feature ratio，过大可能表示 residual 可靠性不足。
    res_to_feat_ratio = res_abs_mean / (feat_abs_mean + eps)
    # 中文注释：返回 B,3，字段顺序固定：abs_mean, abs_std, ratio。
    return torch.stack([res_abs_mean, res_abs_std, res_to_feat_ratio], dim=1)


class DictionaryErrorCompensationBranch(nn.Module):
    """Dictionary Error Compensation Branch，用于学习应被抑制的错误字典残差。"""

    def __init__(
        self,
        dim,
        num_error_tokens=64,
        gate_max=0.10,
        gate_init=-4.0,
        token_init_std=0.02,
        proj_init_std=0.001,
        gate_condition="shortcut",
        residual_detach=True,
    ):
        super().__init__()
        # 中文注释：当前 token feature 维度。
        self.dim = int(dim)
        # 中文注释：error dictionary token 数量。
        self.num_error_tokens = int(num_error_tokens)
        # 中文注释：gate 最大幅度，限制补偿强度，避免 warm-start 破坏 baseline。
        self.gate_max = float(gate_max)
        # 中文注释：gate 条件源；shortcut 保持旧版 DECB-Lite 行为。
        self.gate_condition = str(gate_condition)
        if self.gate_condition not in ("shortcut", "residual", "sum", "concat"):
            raise ValueError("Unsupported DECB gate_condition: {}".format(self.gate_condition))
        # 中文注释：默认 detach 原始 ATD residual，避免 gate 条件分支反向扰动主 residual。
        self.residual_detach = bool(residual_detach)
        # 中文注释：concat 模式 gate 输入是 [shortcut, x_atd]，维度为 2C。
        self.gate_dim = int(self.dim * 2 if self.gate_condition == "concat" else self.dim)
        # 中文注释：可学习 error dictionary tokens，形状 K,C。
        self.error_tokens = nn.Parameter(torch.empty(self.num_error_tokens, self.dim))
        # 中文注释：query 归一化，稳定 query 到 error token 的相似度计算。
        self.norm_q = nn.LayerNorm(self.dim)
        # 中文注释：error token 归一化，稳定 error dictionary 读取。
        self.norm_e = nn.LayerNorm(self.dim)
        # 中文注释：error residual 输出投影，初始化极小但非零，保证梯度可流动。
        self.out_proj = nn.Linear(self.dim, self.dim)
        # 中文注释：token-wise gate，根据配置的条件特征决定每个 token 的扣除强度。
        self.gate = nn.Sequential(
            nn.LayerNorm(self.gate_dim),
            nn.Linear(self.gate_dim, 1),
        )
        # 中文注释：保存 gate bias 初始化值；负值让初始 gate 接近 0。
        self.gate_init = float(gate_init)
        # 中文注释：保存 error token 初始化标准差。
        self.token_init_std = float(token_init_std)
        # 中文注释：保存输出投影初始化标准差。
        self.proj_init_std = float(proj_init_std)
        # 中文注释：执行自定义初始化。
        self.reset_parameters()

    def reset_parameters(self):
        """初始化 DECB，使 warm-start 时近似不改变原 ATD。"""
        # 中文注释：error tokens 用小随机初始化。
        nn.init.normal_(self.error_tokens, mean=0.0, std=self.token_init_std)
        # 中文注释：输出投影使用极小随机初始化，初始 error residual 很弱但梯度不断。
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=self.proj_init_std)
        # 中文注释：输出投影 bias 置零。
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        # 中文注释：gate linear 的 weight 置零，使初始 gate 仅由 bias 控制。
        gate_linear = self.gate[-1]
        nn.init.zeros_(gate_linear.weight)
        # 中文注释：gate bias 为负，sigmoid 后接近 0。
        nn.init.constant_(gate_linear.bias, self.gate_init)

    def _to_token(self, feat):
        """把 B,N,C 或 B,C,H,W 转成 B,N,C，并返回恢复信息。"""
        # 中文注释：B,C,H,W 输入转为 B,N,C，并记录空间形状用于恢复。
        if feat.dim() == 4:
            b, c, h, w = feat.shape
            token = feat.flatten(2).transpose(1, 2)
            return token, True, (b, c, h, w)
        # 中文注释：B,N,C 输入直接使用。
        if feat.dim() == 3:
            return feat, False, None
        raise ValueError("Unsupported DECB feature dim: {}".format(feat.dim()))

    def forward(self, query_feat, residual_feat=None):
        """根据当前内容特征生成 error residual 和 token-wise gate。"""
        # 中文注释：query_feat 是当前内容特征，用于 error dictionary attention。
        query_token, input_is_4d, shape_info = self._to_token(query_feat)
        # 中文注释：检查 query 通道，避免 silent shape bug。
        if query_token.shape[-1] != self.dim:
            raise RuntimeError(
                "DECB query dim mismatch: {} vs {}".format(
                    int(query_token.shape[-1]), int(self.dim)
                )
            )
        # 中文注释：用 shortcut/query 读取 error tokens。
        q = self.norm_q(query_token)
        e = self.norm_e(self.error_tokens)
        attn_logits = torch.matmul(q, e.t()) / (self.dim ** 0.5)
        attn = torch.softmax(attn_logits, dim=-1)
        err = torch.matmul(attn, e)
        x_err_token = self.out_proj(err)

        # 中文注释：构造 gate 输入；shortcut 模式保持旧版 DECB-Lite 行为。
        residual_token = None
        if self.gate_condition == "shortcut":
            gate_input = query_token
            if residual_feat is not None:
                # 中文注释：shortcut 模式不让 residual 参与 gate，只为 debug 记录 x_atd 强度。
                residual_token, _, _ = self._to_token(residual_feat)
                if residual_token.shape != query_token.shape:
                    raise RuntimeError(
                        "DECB residual token shape mismatch: residual {} vs query {}".format(
                            tuple(residual_token.shape), tuple(query_token.shape)
                        )
                    )
        else:
            if residual_feat is None:
                raise RuntimeError("DECB residual_feat is required when gate_condition={}".format(self.gate_condition))
            # 中文注释：residual_feat 只用于 gate 条件，不参与 error token attention。
            residual_token, _, _ = self._to_token(residual_feat)
            if residual_token.shape != query_token.shape:
                raise RuntimeError(
                    "DECB residual token shape mismatch: residual {} vs query {}".format(
                        tuple(residual_token.shape), tuple(query_token.shape)
                    )
                )
            # 中文注释：可选 detach x_atd，保护原 ATD residual 主路径。
            if self.residual_detach:
                residual_token = residual_token.detach()
            # 中文注释：根据配置选择 gate 条件源。
            if self.gate_condition == "residual":
                gate_input = residual_token
            elif self.gate_condition == "sum":
                gate_input = query_token + residual_token
            elif self.gate_condition == "concat":
                gate_input = torch.cat([query_token, residual_token], dim=-1)
            else:
                raise ValueError("Unsupported DECB gate_condition: {}".format(self.gate_condition))

        # 中文注释：token-wise gate，范围 0 到 gate_max。
        gate_token = self.gate_max * torch.sigmoid(self.gate(gate_input))
        # 中文注释：诊断信息，额外记录 residual-conditioned gate 模式。
        aux = {
            "gate": gate_token.detach(),
            "err_abs_mean": x_err_token.detach().abs().mean(dim=(1, 2), keepdim=False),
            "attn_entropy": self._attention_entropy(attn).detach(),
        }
        # 中文注释：如果 gate 使用 residual，记录 residual 强度，方便分析 gate 是否真在看 x_atd。
        if residual_token is not None:
            aux["residual_abs_mean"] = residual_token.detach().abs().mean(dim=(1, 2), keepdim=False)
        else:
            aux["residual_abs_mean"] = torch.zeros(
                query_token.shape[0],
                device=query_token.device,
                dtype=query_token.dtype,
            )
        # 中文注释：如果输入是 4D，则恢复 x_err 和 gate 的空间形状。
        if input_is_4d:
            b, c, h, w = shape_info
            x_err = x_err_token.transpose(1, 2).view(b, c, h, w)
            gate = gate_token.transpose(1, 2).view(b, 1, h, w)
        else:
            # 中文注释：token 输入返回 B,N,C 和 B,N,1。
            x_err = x_err_token
            gate = gate_token
        # 中文注释：返回 error residual、gate 和诊断信息。
        return x_err, gate, aux

    @staticmethod
    def _attention_entropy(attn, eps=1e-8):
        """计算 error token attention entropy，按图像返回 B。"""
        # 中文注释：attn 形状 B,N,K；先对 error token 维计算 entropy。
        entropy = -(attn * torch.log(attn + eps)).sum(dim=-1)
        # 中文注释：对 token 维求均值，得到每张图一个 entropy。
        return entropy.mean(dim=1)


def should_enable_decb(block_idx, total_blocks, mode):
    """按 block index 决定当前 TDCA block 是否启用 DECB。"""
    # 中文注释：none 或非法空值时关闭。
    if mode == "none":
        return False
    # 中文注释：all 表示所有 TDCA block 启用。
    if mode == "all":
        return True
    # 中文注释：last 表示每个 residual group 的最后一个 TDCA block 启用。
    if mode == "last":
        return block_idx == total_blocks - 1
    # 中文注释：interval2 按需求使用奇数位 block。
    if mode == "interval2":
        return block_idx % 2 == 1
    # 中文注释：interval3 使用每 3 层的末位 block。
    if mode == "interval3":
        return block_idx % 3 == 2
    # 中文注释：未知模式保守关闭。
    return False


def should_enable_struct_prior(block_idx, num_blocks, mode):
    """按 block index 决定当前 TDCA block 是否启用 SCDRC。"""
    # 中文注释：none 或非法空值时关闭。
    if mode == "none":
        return False
    # 中文注释：all 表示所有 TDCA block 都启用。
    if mode == "all":
        return True
    # 中文注释：last 表示每个 residual group 的最后一个 TDCA block 启用。
    if mode == "last":
        return block_idx == num_blocks - 1
    # 中文注释：interval2 按需求使用奇数位或最后一层。
    if mode == "interval2":
        return block_idx % 2 == 1 or block_idx == num_blocks - 1
    # 中文注释：interval3 按需求使用每 3 层的末位或最后一层。
    if mode == "interval3":
        return block_idx % 3 == 2 or block_idx == num_blocks - 1
    # 中文注释：未知模式保守关闭。
    return False


def should_enable_radr(block_idx, num_blocks, mode):
    """按 block index 决定当前 TDCA block 是否启用 RADR。"""
    if mode == "none":
        return False
    if mode == "all":
        return True
    if mode == "last":
        return block_idx == num_blocks - 1
    if mode == "interval2":
        return block_idx % 2 == 1 or block_idx == num_blocks - 1
    if mode == "interval3":
        return block_idx % 3 == 2 or block_idx == num_blocks - 1
    return False


class FreqDecoupledGrouping(nn.Module):
    # FDG: S_group = S_content + lambda * S_freq, only for grouping.

    def __init__(self, num_dict_tokens, in_channels=5, d_f=16, lambda0=1.0, init_alpha=0.01):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Conv2d(int(in_channels), int(d_f), kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(d_f), int(d_f), kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Parameter(torch.randn(int(num_dict_tokens), int(d_f)) * 0.02)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.log_tau_f = nn.Parameter(torch.zeros(1))
        self.lambda0 = float(lambda0)
        self.eps = 1e-6

    def forward(self, S_content, freq_map, warmup_factor=1.0, tau_gumbel=1.0,
                use_gumbel=False, hard=True):
        B, N, M = S_content.shape
        Bf, Cf, H, W = freq_map.shape
        assert B == Bf
        assert N == H * W
        assert M == self.phi.shape[0]
        u = self.psi(freq_map.to(dtype=S_content.dtype, device=S_content.device))
        u = u.flatten(2).transpose(1, 2)
        u = F.normalize(u, dim=-1, eps=self.eps)
        phi = F.normalize(self.phi.to(dtype=S_content.dtype, device=S_content.device), dim=-1, eps=self.eps)
        tau_f = self.log_tau_f.exp().clamp(min=0.05, max=10.0).to(dtype=S_content.dtype, device=S_content.device)
        S_freq = torch.einsum("bnd,md->bnm", u, phi) / tau_f
        # 中文注释：gate 是 FDG 唯一可学习门，AC-MSA route bias 也会复用它，避免双 alpha 死锁。
        gate = torch.tanh(self.alpha).to(dtype=S_content.dtype, device=S_content.device)
        lam = self.lambda0 * float(warmup_factor) * gate
        S_group = S_content + lam * S_freq

        tk_id_det = torch.argmax(S_group, dim=-1, keepdim=False)
        hard_assign = F.one_hot(
            tk_id_det,
            num_classes=S_group.shape[-1],
        ).to(dtype=S_group.dtype)

        if self.training:
            if use_gumbel:
                # 中文注释：Gumbel-hard 仅用于消融，不作为主实验默认路径。
                assign = F.gumbel_softmax(
                    S_group,
                    tau=float(tau_gumbel),
                    hard=bool(hard),
                    dim=-1,
                )
                tk_id = assign.argmax(dim=-1)
                assign_mode = "gumbel"
            else:
                # 中文注释：主路径使用 deterministic ST soft assignment。
                # 前向等价 hard argmax，反向通过 softmax(S_group/tau) 回传。
                tau = max(float(tau_gumbel), 1e-6)
                soft_assign = F.softmax(S_group / tau, dim=-1)
                assign = hard_assign + soft_assign - soft_assign.detach()
                tk_id = tk_id_det
                assign_mode = "det_st"
        else:
            # 中文注释：验证/推理阶段完全 deterministic，不引入随机噪声。
            assign = hard_assign
            tk_id = tk_id_det
            assign_mode = "eval_det"

        return S_group, S_freq, lam, gate, assign, tk_id, tk_id_det, assign_mode


# ====================================================================
# DAWA Directional Anisotropic Window Attention ( 1)
# 1) DirectionEstimator Sobel
# 2) stripe_partition / stripe_reverse/
# 3) DirectionalWindowAttention
# ====================================================================


def stripe_partition(x, stripe_size):
    # x: (B, H, W, C); stripe_size: (stripe_h, stripe_w)
    sh, sw = stripe_size #
    b, h, w, c = x.shape #
    assert h % sh == 0 and w % sw == 0, 'H/W ' #
    x = x.view(b, h // sh, sh, w // sw, sw, c) # permute
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() # (B, nh, nw, sh, sw, C)
    windows = windows.view(-1, sh, sw, c) # batch (num_windows*B, sh, sw, C)
    return windows # window_partition


def stripe_reverse(windows, stripe_size, h, w):
    # windows (num_windows*B sh sw C)tripe_size (stripe_h stripe_w)
    sh, sw = stripe_size #
    b = int(windows.shape[0] / (h * w / sh / sw)) # B
    x = windows.view(b, h // sh, w // sw, sh, sw, -1) # (B, nh, nw, sh, sw, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous() # sh nw (B, nh, sh, nw, sw, C)
    x = x.view(b, h, w, -1) # (B, H, W, C)
    return x # window_reverse


class DirectionEstimator(nn.Module):
    # Lightweight fixed-Sobel direction estimator.

    def __init__(self, smooth_size=8, eps=1e-6):
        super().__init__() #
        self.smooth_size = max(int(smooth_size), 1) #
        self.eps = eps # 0
        # Sobel x (1 1 3 3)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        # Sobel y (1 1 3 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x) # buffer
        self.register_buffer('sobel_y', sobel_y) # AMP/DataParallel

    def forward(self, x_img):
        # x_img 褰㈢姸涓?(B, H, W, C)
        b, h, w, c = x_img.shape #
        x = x_img.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        x_flat = x.reshape(b * c, 1, h, w) # batch
        # AMP dtypeobel dtype
        sobel_x = self.sobel_x.to(dtype=x_flat.dtype) #
        sobel_y = self.sobel_y.to(dtype=x_flat.dtype) #
        gx = F.conv2d(x_flat, sobel_x, padding=1) # Gx
        gy = F.conv2d(x_flat, sobel_y, padding=1) # Gy
        gx = gx.view(b, c, h, w) #
        gy = gy.view(b, c, h, w) #
        # 1
        jxx = (gx * gx).mean(dim=1, keepdim=True) # Gx*Gx
        jyy = (gy * gy).mean(dim=1, keepdim=True) # Gy*Gy
        jxy = (gx * gy).mean(dim=1, keepdim=True) # Gx*Gy
        # avg_pool2d
        k = self.smooth_size #
        # H k padding
        pad_h = (k - h % k) % k #
        pad_w = (k - w % k) % k #
        if pad_h > 0 or pad_w > 0: # pad
            jxx = F.pad(jxx, (0, pad_w, 0, pad_h), mode='replicate') # replicate AMP
            jyy = F.pad(jyy, (0, pad_w, 0, pad_h), mode='replicate') #
            jxy = F.pad(jxy, (0, pad_w, 0, pad_h), mode='replicate') #
        jxx_p = F.avg_pool2d(jxx, kernel_size=k, stride=k) # (B,1,H/k,W/k)
        jyy_p = F.avg_pool2d(jyy, kernel_size=k, stride=k) #
        jxy_p = F.avg_pool2d(jxy, kernel_size=k, stride=k) #
        # (1 2)/(1 + 2) [0 1]
        diff = jxx_p - jyy_p  # 瀵硅宸?
        aniso = torch.sqrt(diff * diff + 4.0 * jxy_p * jxy_p + self.eps) # eps sqrt(0)
        denom = jxx_p + jyy_p + self.eps #
        confidence_coarse = (aniso / denom).clamp(0.0, 1.0) # [0,1]
        # Jxx >= Jyy (direction=1)
        direction_coarse = (jxx_p >= jyy_p).to(confidence_coarse.dtype) # 1==
        # (H W) kk
        direction = F.interpolate(direction_coarse, size=(h + pad_h, w + pad_w), mode='nearest')
        confidence = F.interpolate(confidence_coarse, size=(h + pad_h, w + pad_w), mode='nearest')
        direction = direction[:, :, :h, :w] # padding
        confidence = confidence[:, :, :h, :w] # padding
        return direction.squeeze(1), confidence.squeeze(1) # (B, H, W)


class DirectionalWindowAttention(nn.Module):
    # Supports non-square window attention with relative position bias.

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__() #
        self.dim = dim #
        self.window_size = tuple(window_size) # (Wh, Ww) (8, 32) (32, 8)
        self.num_heads = num_heads #
        self.qkv_bias = qkv_bias # qkv WindowAttention
        head_dim = dim // num_heads #
        self.scale = head_dim ** -0.5 # Transformer
        sh, sw = self.window_size #
        # (2*sh 1) * (2*sw 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * sh - 1) * (2 * sw - 1), num_heads))  # 褰㈢姸 (Nrel, num_heads)
        # rpi_sa
        coords_h = torch.arange(sh) # 0..sh-1
        coords_w = torch.arange(sw) # 0..sw-1
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 褰㈢姸 (2, sh, sw)
        coords_flatten = torch.flatten(coords, 1)  # 褰㈢姸 (2, sh*sw)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += sh - 1 #
        relative_coords[:, :, 1] += sw - 1 #
        relative_coords[:, :, 0] *= 2 * sw - 1 # 1D
        relative_position_index = relative_coords.sum(-1) # (N, N)
        self.register_buffer('relative_position_index', relative_position_index)  # 娉ㄥ唽涓?buffer
        self.proj = nn.Linear(dim, dim) #
        trunc_normal_(self.relative_position_bias_table, std=.02) #
        self.softmax = nn.Softmax(dim=-1) # softmax

    def forward(self, qkv_windows, mask=None):
        # qkv_windows (Bn N 3C) N = sh*swn = num_windows*B
        b_, n, c3 = qkv_windows.shape #
        c = c3 // 3 # q/k/v
        sh, sw = self.window_size #
        assert n == sh * sw, 'token ' #
        qkv = qkv_windows.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # q/k/v(Bn, nH, N, d)
        q = q * self.scale #
        attn = q @ k.transpose(-2, -1) # (Bn, nH, N, N)
        # (N N nH) (nH N N)
        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(n, n, -1)
        rpb = rpb.permute(2, 0, 1).contiguous() #
        attn = attn + rpb.unsqueeze(0) #
        if mask is not None: # mask
            nw = mask.shape[0] # mask
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)  # 娉ㄥ叆鎺╃爜
            attn = attn.view(-1, self.num_heads, n, n)  # 鎭㈠褰㈢姸
        attn = self.softmax(attn) # softmax
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c) #
        x = self.proj(x) #
        return x # (Bn, N, C)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

    def flops(self, n):
        # flops
        flops = 0  # 鍒濆鍖?
        flops += self.num_heads * n * (self.dim // self.num_heads) * n  # q @ k^T
        flops += self.num_heads * n * n * (self.dim // self.num_heads)  # attn @ v
        flops += n * self.dim * self.dim #
        return flops # flops


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

    def flops(self, n):
        flops = 0
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        #  x = (attn @ v)
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops


class ATD_CA(nn.Module):
    def __init__(self, dim, num_tokens=64, reducted_dim=10, qkv_bias=True,
                 use_gdc=False, geo_dim=5):
        # use_gdc geo_dim GDC 3 ckpt

        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.dr = reducted_dim
        self.qkv_bias = qkv_bias

        self.wq = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = nn.Parameter(torch.ones([1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

        # GDC
        self.use_gdc = bool(use_gdc) # -
        self.geo_dim = int(geo_dim) # 5
        if self.use_gdc:
            # MLP reducted_dim q
            self.geo_mlp = nn.Sequential(
                nn.Linear(self.geo_dim, dim), # geo_dim -> dim
                nn.GELU(),                       # 婵€娲?
                nn.Linear(dim, reducted_dim), # dim -> reducted_dim q
            )
        else:
            # 中文注释：关闭 GDC 时不创建额外参数。
            self.geo_mlp = None

        # 中文注释：旧频率调制路径已禁用，避免频率项污染 S_content。
        self.use_fcdm = False
        self.use_lfcdm = False
        self.use_fats = False
        self.fcdm_mlp = None
        self.fcdm_scale = None
        self.lfcdm_gate_mlp = None
        self.fats_mlp = None
        self.fats_scale = None

    def forward(self, x, td, x_size, geo_feat=None):
        # geo_feat (B N geo_dim) use_gdc=True None
        h, w = x_size
        b, n, c = x.shape
        b, m, c = td.shape

        # Q: b, n, c
        q = self.wq(x)
        # GDC MLP q
        if self.use_gdc and geo_feat is not None:
            assert geo_feat.shape[0] == b and geo_feat.shape[1] == n, 'geo_feat q '
            assert geo_feat.shape[-1] == self.geo_dim, 'geo_feat geo_dim'
            geo_bias = self.geo_mlp(geo_feat.to(q.dtype))  # (B, N, reducted_dim)
            q = q + geo_bias #
        # K: b, m, c
        k = self.wk(td)
        # V: b, m, c
        v = self.wv(td)

        # Q @ K^T
        S_content = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # b, n, m
        scale = 1 + torch.clamp(self.scale, 0, 3) * np.log(self.num_tokens)
        S_content = S_content * scale
        attn = S_content
        # 中文注释：ATD-CA read-out 只使用原始内容相似度 S_content，不使用 FDG 的 S_group。
        attn = self.softmax(S_content)

        # Attn * V
        x = (attn @ v).reshape(b, n, c)

        return x, S_content, attn

    def flops(self, n):
        flops = 0

        # qkv = self.wq(x)
        flops += n * self.dim * self.dr
        # k = self.wk(td)
        flops += self.num_tokens * self.dim * self.dr
        # v = self.wv(td)
        flops += self.num_tokens * self.dim * self.dim

        # attn = (q @ k.transpose(-2, -1))
        flops += n * self.dim * self.dr

        # x = (attn @ v)
        flops += n * self.num_tokens * self.dim

        # GDC MLPeo_dim > dim > reducted_dim flops
        if self.use_gdc:
            flops += n * self.geo_dim * self.dim    # 绗竴灞?Linear
            flops += n * self.dim * self.dr         # 绗簩灞?Linear

        return flops
    

class AC_MSA(nn.Module):
    def __init__(self, dim, num_heads=4, category_size=128, qkv_bias=True,
                 use_route_bias=False, route_bias_max=1.0, route_bias_init=0.0,
                 route_bias_detach=False):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.category_size = category_size
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = (dim // self.num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)

        # 中文注释：方案 B —— route bias 强度由外部传入的 route_beta tensor 控制，
        # 并复用 FreqDecoupledGrouping 的唯一 gate=tanh(alpha)；这里不再注册独立
        # 的 route_bias_alpha，避免“双 zero-gate 死锁”和 checkpoint 多余 key。
        self.use_route_bias = bool(use_route_bias)
        self.route_bias_max = float(route_bias_max)  # 中文注释：保留构造参数兼容旧调用方。
        self.route_bias_detach = bool(route_bias_detach)
        # 中文注释：legacy 字段——主路径不再使用，保持 None 让 checkpoint 干净。
        self.route_bias_alpha = None

        # 中文注释：最近一次 forward 的 route bias 诊断缓存；未启用时为空。
        self.route_debug_cache = {}

    def forward(self, qkv, tk_id, x_size, route_prob=None, route_beta=None, group_assign=None):
        b, n, c3 = qkv.shape
        c = c3 // 3
        gs = min(n, self.category_size)  # group size
        ng = (n + gs - 1) // gs

        # sort features by type
        x_sort_values, x_sort_indices = torch.sort(tk_id, dim=-1, stable=False)
        tk_id_inv = index_reverse(x_sort_indices)

        # feature categorization
        shuffled_qkv = feature_shuffle(qkv, x_sort_indices)  # b, n, c3
        pad_n = ng * gs - n
        paded_qkv = torch.cat((shuffled_qkv, torch.flip(shuffled_qkv[:, n-pad_n:n, :], dims=[1])), dim=1)
        y = paded_qkv.reshape(b, -1, gs, c3)  # b, ng, gs, c*3

        qkv = y.reshape(b, ng, gs, 3, self.num_heads, c//self.num_heads).permute(3, 0, 1, 4, 2, 5)  # 3, b, ng, nh, gs, c//nh
        q, k, v = qkv[0], qkv[1], qkv[2]    # b, ng, nh, gs, c//nh

        attn = (q @ k.transpose(-2, -1))  # b, ng, nh, gs, gs
        attn = attn * self.scale

        if self.use_route_bias and group_assign is not None:
            shuffled_a = feature_shuffle(group_assign, x_sort_indices)

            if pad_n > 0:
                shuffled_a = torch.cat(
                    (
                        shuffled_a,
                        torch.flip(shuffled_a[:, n - pad_n:n, :], dims=[1]),
                    ),
                    dim=1,
                )

            a_blk = shuffled_a.reshape(b, ng, gs, group_assign.shape[-1])

            same_grp = torch.matmul(a_blk, a_blk.transpose(-2, -1))
            same_grp = same_grp.to(dtype=attn.dtype)

            if route_beta is None:
                route_beta = attn.new_tensor(1.0)
            else:
                route_beta = route_beta.to(dtype=attn.dtype, device=attn.device)

            # 中文注释：same_grp 前向近似 0/1；异组 token 在 block 内被惩罚，同组 token 不惩罚。
            attn = attn + (route_beta * (same_grp - 1.0)).unsqueeze(2)

            with torch.no_grad():
                self.route_debug_cache = {
                    "route_beta": route_beta.detach().float().cpu(),
                    "same_grp_mean": same_grp.detach().float().mean().cpu(),
                    "same_grp_std": same_grp.detach().float().std().cpu(),
                }

        # 中文注释：legacy soft route_prob co-assignment bias，仅供显式消融开关使用。
        elif self.use_route_bias and route_prob is not None:
            # 中文注释：route_prob 是 [B, N, M] 的软字典分配，来自 S_group。
            # 默认不 detach，让梯度通过 AC-MSA 回传到 FDG 的 phi/psi/alpha。
            if self.route_bias_detach:
                route_prob = route_prob.detach()

            # 中文注释：按 hard tk_id 的排序索引同步重排 route_prob，使其与 block 内 qkv token 对齐。
            shuffled_route = feature_shuffle(route_prob, x_sort_indices)  # [B, N, M]

            # 中文注释：padding 必须和 shuffled_qkv 完全一致。
            if pad_n > 0:
                padded_route = torch.cat(
                    (shuffled_route, torch.flip(shuffled_route[:, n - pad_n:n, :], dims=[1])),
                    dim=1,
                )
            else:
                padded_route = shuffled_route

            # 中文注释：整理成 block 形式 [B, ng, G, M]。
            route_blk = padded_route.reshape(b, ng, gs, route_prob.shape[-1])

            # 中文注释：计算 block 内 token 的 soft co-assignment，相当于 A_i · A_j。
            route_sim = torch.matmul(route_blk, route_blk.transpose(-2, -1))  # [B, ng, G, G]

            # 中文注释：数值稳定，避免 route_sim 尺度异常。
            route_sim = route_sim.to(dtype=attn.dtype)
            route_sim = route_sim / route_sim.detach().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

            # 中文注释：route_beta 必须是 tensor 且不要 detach；调用方（ATDTransformerLayer）
            # 用 fdg_route_bias_max * fdg_gate 生成，让梯度通过 gate 回到 freq_grouping.alpha。
            if route_beta is None:
                route_beta = attn.new_tensor(0.0)
            else:
                route_beta = route_beta.to(dtype=attn.dtype, device=attn.device)

            # 中文注释：把可导 route bias 加到 AC-MSA attention logits，broadcast 到 heads。
            attn = attn + route_beta * route_sim.unsqueeze(2)

            # 中文注释：缓存诊断指标，供 ATDTransformerLayer 汇总到 fdg_stats。
            with torch.no_grad():
                self.route_debug_cache = {
                    "route_beta": route_beta.detach().float().cpu(),
                    "route_sim_mean": route_sim.detach().float().mean().cpu(),
                    "route_sim_std": route_sim.detach().float().std().cpu(),
                }
        else:
            # 中文注释：没有 route bias 或没有 route_prob 时清空缓存。
            self.route_debug_cache = {}

        attn = self.softmax(attn)  # b, ng, nh, gs, gs

        y = (attn @ v).permute(0, 1, 3, 2, 4).reshape(b, n+pad_n, c)[:, :n, :]
        x = feature_shuffle(y, tk_id_inv)
        x = self.proj(x)

        return x


    def flops(self, n):
        flops = 0

        # attn = (q @ k.transpose(-2, -1))
        flops += n * self.dim * self.category_size

        # y = (attn @ v)
        flops += n * self.dim * self.category_size

        # x = self.proj(x)
        flops += n * self.dim * self.dim

        return flops


# ====================================================================
# GAA Geometry Addressing Attention ( 2)
# geo_conf
# token n K/V
# TTST top k
# ====================================================================


class SpectralGeometryEstimator(nn.Module):
    # Local periodic vector estimator based on block FFT.

    def __init__(self, block_size=32, dc_radius=2, max_offset=16, eps=1e-6, use_amp_fp32=True, softmax_temp=10.0):
        super().__init__() #
        self.block_size = int(block_size) # FFT
        self.dc_radius = int(dc_radius) # DC/
        self.max_offset = float(max_offset) #
        self.eps = float(eps) #
        self.use_amp_fp32 = bool(use_amp_fp32) # AMP float32 FFT
        self.softmax_temp = float(softmax_temp) # soft-peak softmax argmax

    def forward(self, x_img):
        if x_img.dim() == 4 and x_img.shape[1] == 1:
            gray = x_img
            b, _, h, w = gray.shape
            dtype_orig = gray.dtype
        else:
            assert x_img.dim() == 4, "SpectralGeometryEstimator expects (B,H,W,C) or (B,1,H,W)"
            b, h, w, c = x_img.shape
            dtype_orig = x_img.dtype
            gray = x_img.mean(dim=-1, keepdim=True)
            gray = gray.permute(0, 3, 1, 2).contiguous()
        # H/W 2
        bs = max(2, min(self.block_size, h, w)) #
        # replicate padding H bs
        ph = (bs - h % bs) % bs #
        pw = (bs - w % bs) % bs #
        if ph > 0 or pw > 0: # pad
            gray = F.pad(gray, (0, pw, 0, ph), mode='replicate') # replicate
        H = h + ph #
        W = w + pw #
        nh = H // bs #
        nw = W // bs #
        # batch
        blocks = gray.view(b, 1, nh, bs, nw, bs).permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, nh, nw, 1, bs, bs)
        blocks = blocks.view(b * nh * nw, 1, bs, bs)  # (B*nh*nw, 1, bs, bs)
        # AMP FT float32
        if self.use_amp_fp32 and blocks.dtype != torch.float32:
            blocks_in = blocks.float()  # 杞?fp32
        else:
            blocks_in = blocks # dtype
        # DC
        blocks_in = blocks_in - blocks_in.mean(dim=(-2, -1), keepdim=True) #
        # 2D rfft ( bs bs//2+1)
        spec = torch.fft.rfft2(blocks_in, dim=(-2, -1)) #
        power = spec.real * spec.real + spec.imag * spec.imag #
        # DC mask
        kh = bs #
        kw = bs // 2 + 1 # rfft
        device = power.device #
        freq_y_idx = torch.arange(kh, device=device, dtype=power.dtype)  # 0..bs-1
        # rfft2
        freq_y_signed = torch.where(freq_y_idx <= bs // 2, freq_y_idx, freq_y_idx - bs) # -bs/2..bs/2
        freq_x_signed = torch.arange(kw, device=device, dtype=power.dtype) # x fft
        fy_grid, fx_grid = torch.meshgrid(freq_y_signed, freq_x_signed, indexing='ij') # (kh, kw)
        mag_grid = torch.sqrt(fy_grid * fy_grid + fx_grid * fx_grid + self.eps) #
        dc_mask = (mag_grid > float(self.dc_radius)).to(power.dtype) # dc_radius 1DC
        power = power * dc_mask.view(1, 1, kh, kw) # 0
        # soft peak softmax argmax
        power_flat = power.view(power.shape[0], 1, kh * kw) # (B', 1, kh*kw)
        power_max = power_flat.amax(dim=-1, keepdim=True) + self.eps #
        soft_logits = (power_flat / power_max) * self.softmax_temp # logits
        soft_w = F.softmax(soft_logits, dim=-1)  # (B', 1, kh*kw)
        soft_w = soft_w.view(-1, 1, kh, kw) #
        fx_ex = (soft_w * fx_grid.view(1, 1, kh, kw)).sum(dim=(-2, -1))  # (B', 1)
        fy_ex = (soft_w * fy_grid.view(1, 1, kh, kw)).sum(dim=(-2, -1))  # (B', 1)
        f_norm2 = fx_ex * fx_ex + fy_ex * fy_ex + self.eps #
        delta_x = (bs * fx_ex) / f_norm2 # x
        delta_y = (bs * fy_ex) / f_norm2 # y
        delta_x = delta_x.clamp(-self.max_offset, self.max_offset) #
        delta_y = delta_y.clamp(-self.max_offset, self.max_offset) #
        theta = torch.atan2(fy_ex, fx_ex + self.eps)  # (B', 1)
        # /
        peak_energy = (soft_w * power).sum(dim=(-2, -1)) #
        total_energy = power.sum(dim=(-2, -1)) + self.eps #
        geo_conf = (peak_energy / total_energy).clamp(0.0, 1.0)  # 鎴柇鍒?[0,1]
        # ---- reshape 鍥?(B, 1, nh, nw) ----
        delta_x = delta_x.view(b, nh, nw, 1).permute(0, 3, 1, 2).contiguous()  # (B, 1, nh, nw)
        delta_y = delta_y.view(b, nh, nw, 1).permute(0, 3, 1, 2).contiguous()  # (B, 1, nh, nw)
        theta = theta.view(b, nh, nw, 1).permute(0, 3, 1, 2).contiguous()      # (B, 1, nh, nw)
        geo_conf = geo_conf.view(b, nh, nw, 1).permute(0, 3, 1, 2).contiguous()  # (B, 1, nh, nw)
        # (H W)
        delta_x = F.interpolate(delta_x, size=(H, W), mode='nearest')[:, :, :h, :w]  # (B,1,H,W)
        delta_y = F.interpolate(delta_y, size=(H, W), mode='nearest')[:, :, :h, :w]  # (B,1,H,W)
        theta = F.interpolate(theta, size=(H, W), mode='nearest')[:, :, :h, :w]      # (B,1,H,W)
        geo_conf = F.interpolate(geo_conf, size=(H, W), mode='nearest')[:, :, :h, :w]  # (B,1,H,W)
        # _xy (B 2 H W)
        delta = torch.cat([delta_x, delta_y], dim=1)  # (B, 2, H, W)
        # dtype fp16
        delta = delta.to(dtype_orig) # dtype
        theta = theta.to(dtype_orig) # dtype
        geo_conf = geo_conf.to(dtype_orig) # dtype
        return delta, theta, geo_conf #


class GeometryAddressingAttention(nn.Module):
    # Geometry addressing attention based on periodic offsets.

    def __init__(self, dim, num_heads, num_geo=2, qkv_bias=True, max_offset=16):
        super().__init__() #
        self.dim = dim #
        self.num_heads = num_heads #
        self.num_geo = int(num_geo) # n 2*num_geo
        self.max_offset = float(max_offset) # estimator
        self.qkv_bias = qkv_bias # qkv
        head_dim = dim // num_heads #
        self.scale = head_dim ** -0.5 #
        self.proj = nn.Linear(dim, dim, bias=qkv_bias) #
        self.softmax = nn.Softmax(dim=-1)       # softmax

    def forward(self, qkv, x_size, delta, geo_conf):
        # qkv: (B, N, 3C); delta: (B, 2, H, W); geo_conf: (B, 1, H, W)
        b, n, c3 = qkv.shape #
        c = c3 // 3 #
        h, w = x_size #
        assert n == h * w, 'GAA token H*W' #
        nH = self.num_heads #
        d = c // nH #
        # q, k, v: (B, N, C)
        q = qkv[..., :c]          # 鏌ヨ
        k = qkv[..., c:2 * c]     # 閿?
        v = qkv[..., 2 * c:]      # 鍊?
        # k v grid_sample
        k_map = k.transpose(1, 2).reshape(b, c, h, w).contiguous()  # (B, C, H, W)
        v_map = v.transpose(1, 2).reshape(b, c, h, w).contiguous()  # (B, C, H, W)
        # (x y) [ 1 1]
        device = qkv.device #
        dtype = qkv.dtype    # dtype
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype) # y
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype) # x
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')  # 涓や釜 (H, W)
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0) # (1, H, W, 2) (x, y)
        base_grid = base_grid.expand(b, h, w, 2)  # (B, H, W, 2)
        # grid_sample align_corners=True
        dx_pix = delta[:, 0:1].to(dtype)  # (B, 1, H, W)
        dy_pix = delta[:, 1:2].to(dtype)  # (B, 1, H, W)
        scale_x = 2.0 / max(w - 1, 1) # x
        scale_y = 2.0 / max(h - 1, 1) # y
        norm_dx = (dx_pix * scale_x).permute(0, 2, 3, 1).contiguous()  # (B, H, W, 1)
        norm_dy = (dy_pix * scale_y).permute(0, 2, 3, 1).contiguous()  # (B, H, W, 1)
        norm_delta = torch.cat([norm_dx, norm_dy], dim=-1) # (B, H, W, 2)(x, y)
        # p n K/V=1 num_geo
        k_samples = [] # K
        v_samples = [] # V
        for n_idx in range(1, self.num_geo + 1): # 1 num_geo
            for sign in (1.0, -1.0): #
                grid = base_grid + sign * float(n_idx) * norm_delta  # (B, H, W, 2)
                k_s = F.grid_sample(k_map, grid, mode='bilinear', padding_mode='border', align_corners=True)  # (B, C, H, W)
                v_s = F.grid_sample(v_map, grid, mode='bilinear', padding_mode='border', align_corners=True)  # (B, C, H, W)
                k_samples.append(k_s) #
                v_samples.append(v_s) #
        # (B M C H W) M = 2*num_geo
        k_stack = torch.stack(k_samples, dim=1)  # (B, M, C, H, W)
        v_stack = torch.stack(v_samples, dim=1)  # (B, M, C, H, W)
        M = k_stack.shape[1] #
        # (B nH N M d)
        k_stack = k_stack.view(b, M, nH, d, h, w).permute(0, 2, 4, 5, 1, 3).contiguous()  # (B, nH, H, W, M, d)
        k_stack = k_stack.view(b, nH, n, M, d)  # (B, nH, N, M, d)
        v_stack = v_stack.view(b, M, nH, d, h, w).permute(0, 2, 4, 5, 1, 3).contiguous()  # (B, nH, H, W, M, d)
        v_stack = v_stack.view(b, nH, n, M, d)  # (B, nH, N, M, d)
        # ---- 鎷?q 涓哄澶?----
        q = q.reshape(b, n, nH, d).permute(0, 2, 1, 3).contiguous()  # (B, nH, N, d)
        q = q * self.scale  # 缂╂斁
        # token M
        attn = (q.unsqueeze(-2) @ k_stack.transpose(-2, -1)).squeeze(-2)  # (B, nH, N, M)
        attn = self.softmax(attn) # softmax
        out = (attn.unsqueeze(-2) @ v_stack).squeeze(-2)  # (B, nH, N, d)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(b, n, c)  # (B, N, C)
        out = self.proj(out) #
        # geo_conf 1 H W > B N 1
        gconf = geo_conf.view(b, 1, h * w).transpose(1, 2).to(out.dtype)  # (B, N, 1)
        out = out * gconf #
        return out # (B, N, C)

    def flops(self, n):
        # flops
        flops = 0  # 鍒濆鍖?
        M = 2 * self.num_geo #
        flops += self.num_heads * n * M * (self.dim // self.num_heads) * 2  # attn 鍜?attn@v
        flops += n * self.dim * self.dim #
        return flops #


class TopKRetrievalAttention(nn.Module):
    # TTST-style top-k retrieval attention baseline.

    def __init__(self, dim, num_heads, topk=8, qkv_bias=True, max_tokens=4096, query_chunk=1024):
        super().__init__() #
        self.dim = dim #
        self.num_heads = num_heads #
        self.topk = int(topk)                   # 妫€绱㈢殑 top-k 鏁?
        self.max_tokens = int(max_tokens) #
        self.query_chunk = int(query_chunk) # query
        self.qkv_bias = qkv_bias # qkv
        head_dim = dim // num_heads #
        self.scale = head_dim ** -0.5 #
        self.proj = nn.Linear(dim, dim, bias=qkv_bias) #

    def forward(self, qkv, x_size):
        # qkv: (B, N, 3C)
        b, n, c3 = qkv.shape #
        c = c3 // 3 #
        h, w = x_size #
        nH = self.num_heads #
        d = c // nH #
        # ---- 鎷?q, k, v ----
        q = qkv[..., :c]          # (B, N, C)
        k = qkv[..., c:2 * c]     # (B, N, C)
        v = qkv[..., 2 * c:]      # (B, N, C)
        q = q.reshape(b, n, nH, d).permute(0, 2, 1, 3).contiguous()  # (B, nH, N, d)
        k = k.reshape(b, n, nH, d).permute(0, 2, 1, 3).contiguous()  # (B, nH, N, d)
        v = v.reshape(b, n, nH, d).permute(0, 2, 1, 3).contiguous()  # (B, nH, N, d)
        # N K/V
        stride = 1  # 榛樿姝ラ暱
        if n > self.max_tokens: #
            ratio = math.sqrt(n / float(self.max_tokens)) #
            stride = max(int(math.ceil(ratio)), 1) #
        if stride > 1: #
            k_map = k.permute(0, 1, 3, 2).reshape(b * nH, d, h, w)  # (B*nH, d, H, W)
            v_map = v.permute(0, 1, 3, 2).reshape(b * nH, d, h, w)  # (B*nH, d, H, W)
            k_map = k_map[:, :, ::stride, ::stride] #
            v_map = v_map[:, :, ::stride, ::stride] #
            new_h, new_w = k_map.shape[-2], k_map.shape[-1] #
            k_pool = k_map.reshape(b, nH, d, new_h * new_w).permute(0, 1, 3, 2).contiguous()  # (B, nH, Ks, d)
            v_pool = v_map.reshape(b, nH, d, new_h * new_w).permute(0, 1, 3, 2).contiguous()  # (B, nH, Ks, d)
        else: #
            k_pool = k # K
            v_pool = v # V
        ks = k_pool.shape[2]  # 妫€绱㈡睜澶у皬
        topk = min(self.topk, ks) # top-k
        # query (N Ks)
        BH = b * nH # batch head
        q_flat = q.reshape(BH, n, d)      # (BH, N, d)
        k_flat = k_pool.reshape(BH, ks, d)  # (BH, Ks, d)
        v_flat = v_pool.reshape(BH, ks, d)  # (BH, Ks, d)
        out_chunks = [] # query chunk
        chunk = max(self.query_chunk, 1) #
        for start in range(0, n, chunk): # query
            end = min(start + chunk, n) #
            q_part = q_flat[:, start:end, :] * self.scale # q chunk (BH, nq, d)
            attn_part = q_part @ k_flat.transpose(-2, -1)   # (BH, nq, Ks)
            tk_vals, tk_idx = attn_part.topk(topk, dim=-1)  # (BH, nq, k) 鍙?top-k
            tk_w = F.softmax(tk_vals, dim=-1) # top-k softmax
            # gather v (BH nq Ks d)
            nq = end - start # chunk query
            flat_idx = tk_idx.reshape(BH, nq * topk)        # (BH, nq*topk)
            flat_idx_exp = flat_idx.unsqueeze(-1).expand(-1, -1, d)  # (BH, nq*topk, d)
            v_sel_flat = torch.gather(v_flat, dim=1, index=flat_idx_exp)  # (BH, nq*topk, d)
            v_sel = v_sel_flat.reshape(BH, nq, topk, d)     # (BH, nq, topk, d)
            out_part = (tk_w.unsqueeze(-2) @ v_sel).squeeze(-2)  # (BH, nq, d)
            out_chunks.append(out_part) #
        out = torch.cat(out_chunks, dim=1) # (BH, N, d)
        out = out.reshape(b, nH, n, d).permute(0, 2, 1, 3).contiguous().reshape(b, n, c)  # (B, N, C)
        out = self.proj(out) #
        return out # (B, N, C)

    def flops(self, n):
        # flops
        flops = 0  # 鍒濆鍖?
        ks = min(n, self.max_tokens) # max_tokens
        flops += self.num_heads * n * ks * (self.dim // self.num_heads) #
        flops += self.num_heads * n * self.topk * (self.dim // self.num_heads) # v
        flops += n * self.dim * self.dim #
        return flops #


class ATDTransformerLayer(nn.Module):
    def __init__(self,
                 dim,
                 idx,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 dim_ffn_td,
                 category_size,
                 num_tokens,
                 reducted_dim,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_dawa=False,
                 dawa_long=32,
                 dawa_short=8,
                 dawa_conf_threshold=0.15,
                 use_gaa=False,
                 gaa_num_geo=2,
                 gaa_max_offset=16,
                 gaa_block_size=32,
                 gaa_estimator_source="lr_gray",
                 use_topk_retrieval=False,
                 topk=8,
                 topk_max_tokens=4096,
                 use_gdc=False,
                 geo_feat_dim=5,
                 use_fdg=False,
                 fdg_in_channels=5,
                 fdg_df=16,
                 fdg_lambda0=1.0,
                 fdg_init_alpha=0.01,
                 fdg_layer_mode="last",
                 fdg_use_acmsa_route_bias=True,
                 fdg_route_bias_max=1.0,
                 fdg_route_bias_init=0.0,
                 fdg_route_bias_detach=False,
                 fdg_use_td_ste=False,
                 fdg_use_gumbel_grouping=False,
                 fdg_tau_gumbel=1.0,
                 fdg_use_routeprob_bias=False,
                 use_fcdm=False,
                 fcdm_desc_dim=6,
                 fcdm_hidden_dim=64,
                 fcdm_scale_init=0.0,
                 use_fats=False,
                 fats_hidden_dim=32,
                 fats_tau_range=0.3,
                 use_lfcdm=False,
                 lfcdm_desc_dim=6,
                 lfcdm_hidden_dim=32,
                 lfcdm_gate_bias=0.0,
                 use_struct_prior=False,
                 enable_struct_prior=False,
                 use_radr=False,
                 enable_radr=False,
                 radr_hidden_dim=32,
                 radr_lambda=0.10,
                 radr_tau=0.50,
                 radr_init_bias=-4.0,
                 radr_detach_feat=True,
                 radr_use_correction=False,
                 radr_corr_hidden_dim=64,
                 radr_corr_lambda=0.05,
                 radr_corr_scale=0.10,
                 radr_corr_init_std=1e-4,
                 radr_corr_detach_residual=True,
                 radr_corr_gate_mode="ueff",
                 radr_corr_feature_mode="shortcut_xatd",
                 radr_corr_train_feature_modes="",
                 use_radr_ccd=False,
                 radr_ccd_upscale=4,
                 radr_ccd_aux_init_std=1e-4,
                 radr_ccd_aux_scale=0.10,
                 radr_ccd_aux_use_gated_corr=True,
                 use_radr_lch=False,
                 radr_lch_hidden_dim=64,
                 radr_lch_lambda=0.05,
                 radr_lch_corr_scale=0.05,
                  radr_lch_init_std=1e-4,
                  radr_lch_gate_mode="u_eff",
                  radr_lch_detach_map=True,
                  use_decb=False,
                 decb_num_tokens=64,
                 decb_gate_max=0.10,
                 decb_gate_init=-4.0,
                 decb_token_init_std=0.02,
                 decb_proj_init_std=0.001,
                 decb_gate_condition="shortcut",
                 decb_residual_detach=True):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens=num_tokens
        self.softmax = nn.Softmax(dim=-1)
        # self.lrelu = nn.LeakyReLU()
        # self.sigmoid = nn.Sigmoid()
        self.reducted_dim = reducted_dim
        self.dim_ffn_td = dim_ffn_td

        # 中文注释：DAWA/GAA/top-k/GDC 保持原有可选分支配置。
        self.use_dawa = bool(use_dawa)
        self.dawa_long = int(dawa_long)
        self.dawa_short = int(dawa_short)
        self.dawa_conf_threshold = float(dawa_conf_threshold)
        self.use_gaa = bool(use_gaa)
        self.gaa_num_geo = int(gaa_num_geo)
        self.gaa_max_offset = float(gaa_max_offset)
        self.gaa_block_size = int(gaa_block_size)
        self.gaa_estimator_source = str(gaa_estimator_source)
        _valid_gaa_estimator_sources = ["current", "shallow", "shallow_detach", "lr_gray"]
        if self.gaa_estimator_source not in _valid_gaa_estimator_sources:
            raise ValueError("gaa_estimator_source must be one of {}, got {!r}".format(
                _valid_gaa_estimator_sources, self.gaa_estimator_source))
        self.use_topk_retrieval = bool(use_topk_retrieval) and not self.use_gaa
        self.topk = int(topk)
        self.topk_max_tokens = int(topk_max_tokens)
        self.use_gdc = bool(use_gdc)
        self.geo_feat_dim = int(geo_feat_dim)
        self.use_fdg = bool(use_fdg)                          # 中文注释：FDG 只控制 AC-MSA grouping，不进入 read-out。
        if self.use_fdg:
            # 中文注释：每个启用层独立学习频率原型 phi，生成 S_group 供 category 使用。
            self.freq_grouping = FreqDecoupledGrouping(
                num_dict_tokens=num_tokens,
                in_channels=fdg_in_channels,
                d_f=fdg_df,
                lambda0=fdg_lambda0,
                init_alpha=fdg_init_alpha,
            )
        else:
            # 中文注释：关闭 FDG 时不创建新增参数，baseline 行为不变。
            self.freq_grouping = None
        # 中文注释：记录当前层的 FDG 层级模式，便于排查实际启用策略。
        self.fdg_layer_mode = str(fdg_layer_mode)
        # 中文注释：方案 B 新增开关——AC-MSA route bias 和 x_td STE。
        self.fdg_use_acmsa_route_bias = bool(fdg_use_acmsa_route_bias)
        self.fdg_route_bias_max = float(fdg_route_bias_max)
        self.fdg_route_bias_init = float(fdg_route_bias_init)
        self.fdg_route_bias_detach = bool(fdg_route_bias_detach)
        self.fdg_use_td_ste = bool(fdg_use_td_ste)
        self.fdg_use_gumbel_grouping = bool(fdg_use_gumbel_grouping)
        self.fdg_tau_gumbel = float(fdg_tau_gumbel)
        self.fdg_use_routeprob_bias = bool(fdg_use_routeprob_bias)
        self.use_fcdm = False
        self.use_fats = False
        self.use_lfcdm = False
        # 中文注释：SCDRC 只校准当前层的 ATD-CA/dictionary residual，不改 logits 或 token。
        self.use_struct_prior = bool(use_struct_prior)
        # 中文注释：enable_struct_prior 由 struct_layer_mode 决定，默认 last 表示每个 group 最后一层。
        self.enable_struct_prior = bool(enable_struct_prior)
        # 中文注释：RADR 只对 dictionary residual 做软抑制，不改 logits、token 或其它分支。
        self.use_radr = bool(use_radr)
        self.enable_radr = bool(enable_radr)
        self.radr_lambda = float(radr_lambda)
        self.radr_tau = float(radr_tau)
        self.radr_detach_feat = bool(radr_detach_feat)
        self.radr_use_correction = bool(radr_use_correction)
        self.radr_corr_lambda = float(radr_corr_lambda)
        self.radr_corr_gate_mode = str(radr_corr_gate_mode)
        if self.radr_corr_gate_mode not in ("ueff", "sqrt", "binary", "none"):
            raise ValueError("Unsupported radr_corr_gate_mode: {}".format(self.radr_corr_gate_mode))
        self.radr_corr_feature_mode = str(radr_corr_feature_mode)
        self.radr_corr_train_feature_modes = str(radr_corr_train_feature_modes or "")
        valid_corr_feature_modes = ("shortcut_xatd", "shortcut_only", "shortcut_xwin", "shortcut_xaca")
        if self.radr_corr_feature_mode not in valid_corr_feature_modes:
            raise ValueError("Unsupported radr_corr_feature_mode: {}".format(self.radr_corr_feature_mode))
        if self.radr_corr_train_feature_modes:
            train_modes = tuple(
                mode.strip() for mode in self.radr_corr_train_feature_modes.split(",") if mode.strip()
            )
            for mode in train_modes:
                if mode not in valid_corr_feature_modes:
                    raise ValueError("Unsupported train radr_corr_feature_mode: {}".format(mode))
            self.radr_corr_train_feature_modes_tuple = train_modes
        else:
            self.radr_corr_train_feature_modes_tuple = ()
        self.radr_predictor = LocalReliabilityPredictor(
            dim=dim,
            hidden_dim=radr_hidden_dim,
            out_scale="token",
            init_bias=radr_init_bias,
        ) if (self.use_radr and self.enable_radr) else None
        self.radr_correction = ReliabilityCorrectionBranch(
            dim=dim,
            hidden_dim=radr_corr_hidden_dim,
            corr_scale=radr_corr_scale,
            init_std=radr_corr_init_std,
            detach_residual=radr_corr_detach_residual,
        ) if (self.use_radr and self.enable_radr and self.radr_use_correction) else None
        self.use_radr_ccd = bool(use_radr_ccd)
        self.radr_ccd_aux_use_gated_corr = bool(radr_ccd_aux_use_gated_corr)
        self.radr_ccd_aux_head = RADRCorrectionAuxHead(
            dim=dim,
            upscale=radr_ccd_upscale,
            init_std=radr_ccd_aux_init_std,
            out_scale=radr_ccd_aux_scale,
        ) if (
            self.use_radr
            and self.enable_radr
            and self.radr_use_correction
            and self.use_radr_ccd
        ) else None
        # 中文注释：DECB 只作用于当前层的 ATD-CA/dictionary residual，不改 logits 或 token。
        self.use_decb = bool(use_decb)
        # 中文注释：每个启用 block 拥有独立 error compensation branch，默认关闭时不注册参数。
        self.decb_branch = DictionaryErrorCompensationBranch(
            dim=dim,
            num_error_tokens=decb_num_tokens,
            gate_max=decb_gate_max,
            gate_init=decb_gate_init,
            token_init_std=decb_token_init_std,
            proj_init_std=decb_proj_init_std,
            gate_condition=decb_gate_condition,
            residual_detach=decb_residual_detach,
        ) if self.use_decb else None
        # GAA
        self.enable_geo_debug = False
        # forward geo_conf/theta/delta train py TensorBoard
        self.geo_debug_cache = {}

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.wqkv = nn.Linear(dim, 3*dim, bias=qkv_bias)

        self.attn_win = WindowAttention(
            self.dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.attn_atd = ATD_CA(
            self.dim,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim,
            qkv_bias=qkv_bias,
            use_gdc=self.use_gdc,            # 中文注释：透传 GDC 开关到 ATD_CA。
            geo_dim=self.geo_feat_dim,       # 中文注释：透传几何特征维度。
        )
        self.attn_aca = AC_MSA(
            self.dim,
            num_heads=num_heads,
            category_size=category_size,
            qkv_bias=qkv_bias,
            # 中文注释：只在 FDG 启用且开关打开时为 AC-MSA 注册 route bias 参数，
            # 这样关闭时 baseline state_dict 不受影响。
            use_route_bias=self.use_fdg and self.fdg_use_acmsa_route_bias,
            route_bias_max=self.fdg_route_bias_max,
            route_bias_init=self.fdg_route_bias_init,
            route_bias_detach=self.fdg_route_bias_detach,
        )

        # DAWA ckpt
        if self.use_dawa:
            self.dir_estimator = DirectionEstimator(smooth_size=self.dawa_short)
            self.attn_hstripe = DirectionalWindowAttention(
                dim=self.dim,
                window_size=(self.dawa_short, self.dawa_long),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
            )
            self.attn_vstripe = DirectionalWindowAttention(
                dim=self.dim,
                window_size=(self.dawa_long, self.dawa_short),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
            )
            # DAWA 0 baseline
            self.dawa_scale = nn.Parameter(torch.zeros(1))
        else:
            # DAWA None state_dict
            self.dir_estimator = None
            self.attn_hstripe = None
            self.attn_vstripe = None
            # DAWA baseline state_dict
            self.dawa_scale = None

        # GAA
        if self.use_gaa:
            self.geo_estimator = SpectralGeometryEstimator(
                block_size=self.gaa_block_size, # FFT
                dc_radius=2, # DC
                max_offset=self.gaa_max_offset, #
            )
            self.attn_gaa = GeometryAddressingAttention(
                dim=self.dim, #
                num_heads=num_heads, #
                num_geo=self.gaa_num_geo, #
                qkv_bias=qkv_bias, # qkv
                max_offset=self.gaa_max_offset, #
            )
            # 0 tanh baseline
            self.gaa_scale = nn.Parameter(torch.zeros(1)) #
        else:
            self.geo_estimator = None # GAA
            self.attn_gaa = None # GAA
            self.gaa_scale = None # ckpt

        # top k use_topk_retrieval GAA
        if self.use_topk_retrieval:
            self.attn_topk = TopKRetrievalAttention(
                dim=self.dim, #
                num_heads=num_heads, #
                topk=self.topk,                     # top-k 鏁?
                qkv_bias=qkv_bias, # qkv
                max_tokens=self.topk_max_tokens, #
            )
            self.topk_scale = nn.Parameter(torch.zeros(1)) #
        else:
            self.attn_topk = None # top-k
            self.topk_scale = None #

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc_td = nn.Linear(dim, dim_ffn_td)
        self.convffn = ConvFFN_td(in_features=dim, hidden_features=mlp_hidden_dim, td_features=dim_ffn_td, kernel_size=convffn_kernel_size, act_layer=act_layer)

    def _compute_radr_corr_gate(self, u_eff):
        mode = str(getattr(self, "radr_corr_gate_mode", "ueff"))
        if mode == "ueff":
            return u_eff
        if mode == "sqrt":
            return torch.sqrt(u_eff.clamp_min(0.0) + 1e-6)
        if mode == "binary":
            return (u_eff > 0.0).to(dtype=u_eff.dtype, device=u_eff.device)
        if mode == "none":
            return torch.ones_like(u_eff)
        raise ValueError("Unsupported radr_corr_gate_mode: {}".format(mode))

    def _select_radr_corr_feature_mode(self, device=None):
        modes = getattr(self, "radr_corr_train_feature_modes_tuple", ())
        if self.training and modes:
            if device is None:
                idx = int(torch.randint(len(modes), (1,)).item())
            else:
                idx = int(torch.randint(len(modes), (1,), device=device).item())
            return modes[idx]
        return str(getattr(self, "radr_corr_feature_mode", "shortcut_xatd"))

    def _to_token_like(self, feat, shortcut, x_size, name="feature"):
        if feat is None:
            raise RuntimeError("{} is required for RADR correction feature routing".format(name))
        if feat.dim() == 3:
            if feat.shape != shortcut.shape:
                raise RuntimeError(
                    "{} token shape {} must match shortcut {}".format(
                        name, tuple(feat.shape), tuple(shortcut.shape)
                    )
                )
            return feat
        if feat.dim() == 4:
            h, w = x_size
            b, n, c = shortcut.shape
            if feat.shape[0] == b and feat.shape[1] == h and feat.shape[2] == w and feat.shape[3] == c:
                return feat.reshape(b, h * w, c)
            if feat.shape[0] == b and feat.shape[1] == c and feat.shape[2] == h and feat.shape[3] == w:
                return feat.flatten(2).transpose(1, 2).contiguous()
        raise RuntimeError(
            "{} has unsupported shape {}; expected B,N,C or B,H,W,C/B,C,H,W compatible with shortcut {}".format(
                name, tuple(feat.shape), tuple(shortcut.shape)
            )
        )

    def _build_radr_corr_context(self, mode, shortcut, x_atd_orig, x_win=None, x_aca=None, x_size=None):
        if mode == "shortcut_xatd":
            return x_atd_orig, 0.0
        if mode == "shortcut_only":
            return torch.zeros_like(shortcut), 1.0
        if mode == "shortcut_xwin":
            return self._to_token_like(x_win, shortcut, x_size, name="x_win"), 2.0
        if mode == "shortcut_xaca":
            return self._to_token_like(x_aca, shortcut, x_size, name="x_aca"), 3.0
        raise ValueError("Unsupported radr_corr_feature_mode: {}".format(mode))


    def forward(self, x, td, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        b, m, c = td.shape
        c3 = 3 * c

        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)

        # ====================================================
        # GDC attn_atd
        # DAWA / GAA FFT/Sobel
        # ====================================================
        cached_direction = None # DAWA (B,H,W)
        cached_dir_conf = None # DAWA (B,H,W)
        cached_delta = None # GAA (B,2,H,W)
        cached_theta = None # GAA (B,1,H,W)
        cached_geo_conf = None # GAA (B,1,H,W)

        def _select_gaa_estimator_input():
            # 中文注释：按配置选择 GAA 几何估计输入，保持原有 GAA 行为。
            if self.gaa_estimator_source == "lr_gray" and "gaa_lr_gray" in params:
                # 中文注释：lr_gray 优先直接使用原始 LR 灰度图。
                lr_gray = params["gaa_lr_gray"]
                lr_gray_size = params.get("gaa_lr_gray_size", x_size)
                if lr_gray_size == x_size and lr_gray.shape[-2:] == (h, w):
                    return lr_gray
                # 中文注释：尺寸不一致时插值到当前层空间分辨率。
                lr_gray = F.interpolate(lr_gray.float(), size=(h, w), mode="bilinear", align_corners=False)
                return lr_gray.to(x.dtype)

            use_shallow = (
                self.gaa_estimator_source in ["shallow", "shallow_detach"]
                and "gaa_shallow_feat" in params
            )
            if use_shallow:
                shallow = params["gaa_shallow_feat"]
                shallow_size = params.get("gaa_shallow_size", x_size)
                # 中文注释：shallow_detach 模式切断几何估计器到浅层主干的梯度。
                if self.gaa_estimator_source == "shallow_detach":
                    shallow = shallow.detach()
                assert shallow.shape[-1] == c, "GAA shallow feature channels must match current layer"
                if shallow_size == x_size and shallow.shape[1] == h * w:
                    return shallow.reshape(b, h, w, c)
                # 中文注释：把浅层 token 还原成空间特征后插值到当前尺寸。
                hs, ws = shallow_size
                shallow_img = shallow.transpose(1, 2).reshape(b, c, hs, ws)
                shallow_img = F.interpolate(shallow_img, size=(h, w), mode="bilinear", align_corners=False)
                return shallow_img.permute(0, 2, 3, 1).contiguous()

            # 中文注释：current 模式或无浅层缓存时，使用当前层特征估计几何。
            return x.reshape(b, h, w, c)

        # 中文注释：GDC 只在 DAWA/GAA 提供几何信息时构建 geo_feat。
        need_geo_for_gdc = self.use_gdc and (self.use_gaa or self.use_dawa)
        # 中文注释：GAA 或 GDC+GAA 需要提前计算一次几何估计。
        if self.use_gaa or (self.use_gdc and self.use_gaa):
            x_img_for_est = _select_gaa_estimator_input()  # (B, H, W, C)
            cached_delta, cached_theta, cached_geo_conf = self.geo_estimator(x_img_for_est)
        # GAA DAWA GDC + DAWA DAWA
        if (self.use_dawa or (self.use_gdc and self.use_dawa and not self.use_gaa)):
            x_img_for_dir = x.view(b, h, w, c)  # (B, H, W, C)
            cached_direction, cached_dir_conf = self.dir_estimator(x_img_for_dir)

        # geo_feat use_gdc=True None
        geo_feat = None
        if need_geo_for_gdc:
            if cached_delta is not None: # GAA
                sin_t = torch.sin(cached_theta).to(x.dtype)        # (B,1,H,W)
                cos_t = torch.cos(cached_theta).to(x.dtype)        # (B,1,H,W)
                dx = (cached_delta[:, 0:1] / max(self.gaa_max_offset, 1e-6)).to(x.dtype) # [-1,1]
                dy = (cached_delta[:, 1:2] / max(self.gaa_max_offset, 1e-6)).to(x.dtype) #
                conf = cached_geo_conf.to(x.dtype)                 # (B,1,H,W)
                geo_feat_map = torch.cat([sin_t, cos_t, dx, dy, conf], dim=1)  # (B,5,H,W)
            else:
                # DAWAirection {0 1} {0 /2} sin=directionos=1 direction
                d = cached_direction.unsqueeze(1).to(x.dtype)      # (B,1,H,W)
                cf = cached_dir_conf.unsqueeze(1).to(x.dtype)      # (B,1,H,W)
                zero = torch.zeros_like(d) # 0
                geo_feat_map = torch.cat([d, 1.0 - d, zero, zero, cf], dim=1)  # (B,5,H,W)
            # (B 5 H W) (B N 5) token
            geo_feat = geo_feat_map.view(b, self.geo_feat_dim, h * w).transpose(1, 2).contiguous()

        # ATD_CA
        # 中文注释：x_atd 的 read-out 只由 S_content 的 softmax 产生，不接收 FDG 频率项。
        x_atd, S_content, attn_content = self.attn_atd(x, td, x_size, geo_feat=geo_feat)
        if params.get("capture_dict_attn", False):
            # 中文注释：FDPP 如需字典 attention，缓存原始内容相似度 softmax 后的 attention。
            params["dict_attn_list"].append(attn_content)

        # AC_MSA grouping/category
        S_group = S_content
        S_freq = None
        lam = None
        fdg_gate = None  # 中文注释：FDG 共享 gate，AC-MSA route bias 也用它生成 route_beta。
        fdg_assign = None
        category_base = torch.argmax(S_content, dim=-1, keepdim=False)
        if self.use_fdg and self.freq_grouping is not None and params.get("freq_map", None) is not None:
            # 中文注释：FDG 只生成 grouping 用的 S_group，不改变 S_content。
            S_group, S_freq, lam, fdg_gate, fdg_assign, tk_id, tk_id_det, assign_mode = self.freq_grouping(
                S_content,
                params["freq_map"],
                warmup_factor=params.get("fdg_warmup_factor", 1.0),
                tau_gumbel=params.get("fdg_tau_gumbel", 1.0),
                use_gumbel=params.get("fdg_use_gumbel_grouping", False),
                hard=True,
            )
            with torch.no_grad():
                # 中文注释：确定性 FDG 分组，不含 Gumbel 噪声。
                change_det = (category_base != tk_id_det).float().mean()

                # 中文注释：forward 表示训练时实际写入 AC-MSA 的分组相对 baseline 的变化。
                change_forward = (category_base != tk_id).float().mean()

                # 中文注释：sample_noise 表示 forward 分组相对 deterministic argmax 的额外扰动。
                change_sample_vs_det = (tk_id_det != tk_id).float().mean()
                hist = torch.bincount(tk_id.reshape(-1), minlength=S_group.shape[-1]).float()
                prob = hist / hist.sum().clamp_min(1.0)
                entropy = -(prob * (prob + 1e-12).log()).sum() / math.log(max(int(S_group.shape[-1]), 2))
                params.setdefault("fdg_stats", []).append({
                    "lambda": lam.detach().float(),
                    "alpha": self.freq_grouping.alpha.detach().float(),
                    "fdg_gate": fdg_gate.detach().float(),
                    "S_freq_abs_mean": S_freq.detach().abs().mean().float(),
                    "category_change_ratio": change_det.detach().float(),
                    "category_change_det": change_det.detach().float(),
                    "category_change_forward": change_forward.detach().float(),
                    "category_change_sample_vs_det": change_sample_vs_det.detach().float(),
                    "category_change_gumbel": change_forward.detach().float(),
                    "category_change_gumbel_vs_det": change_sample_vs_det.detach().float(),
                    "category_entropy": entropy.detach().float(),
                    "fdg_assign_mode": assign_mode,
                    "fdg_tau_gumbel": torch.tensor(
                        float(params.get("fdg_tau_gumbel", 1.0)),
                        dtype=torch.float32,
                    ),
                    "fdg_use_gumbel_grouping": torch.tensor(
                        float(bool(params.get("fdg_use_gumbel_grouping", False))),
                        dtype=torch.float32,
                    ),
                    "S_content_shape": tuple(S_content.shape),
                    "freq_map_shape": tuple(params["freq_map"].shape),
                    "S_freq_shape": tuple(S_freq.shape),
                    "S_group_shape": tuple(S_group.shape),
                    "fdg_assign_shape": tuple(fdg_assign.shape),
                })
        else:
            # 中文注释：关闭 FDG 时，grouping 完全回退为原始内容相似度 argmax。
            tk_id = category_base
        td_proj = self.fc_td(td)

        # 中文注释：主路径使用 deterministic ST one-hot same-group gate；旧 route_prob 只作消融。
        route_prob = None
        route_beta = None
        group_assign = None
        if (
            self.use_fdg
            and self.freq_grouping is not None
            and S_group is not None
            and self.fdg_use_acmsa_route_bias
            and fdg_gate is not None
        ):
            # 中文注释：AC-MSA route bias 复用 FDG 的唯一 gate，不再使用独立 alpha。
            # 注意：route_beta 不乘 warmup_factor，让可导代理更早提供梯度；
            # hard grouping 的实际扰动仍由 lam = warmup * gate 控制。
            route_beta = self.fdg_route_bias_max * fdg_gate
            if self.fdg_use_routeprob_bias:
                route_prob = F.softmax(S_group, dim=-1)
                group_assign = None
            else:
                route_prob = None
                group_assign = fdg_assign

        # 中文注释：AC-MSA 使用 hard tk_id 前向路由，same-group gate 提供可导 logits bias。
        x_aca = self.attn_aca(
            qkv,
            tk_id,
            x_size,
            route_prob=route_prob,
            route_beta=route_beta,
            group_assign=group_assign,
        )

        # 中文注释：把 AC-MSA route bias 诊断写入当前层的 fdg_stats（如果存在）。
        if params.get("fdg_stats", None):
            route_cache = getattr(self.attn_aca, "route_debug_cache", {})
            if route_cache:
                if "route_beta" in route_cache:
                    params["fdg_stats"][-1]["acmsa_route_beta"] = route_cache["route_beta"]
                if "route_sim_mean" in route_cache:
                    params["fdg_stats"][-1]["acmsa_route_sim_mean"] = route_cache["route_sim_mean"]
                if "route_sim_std" in route_cache:
                    params["fdg_stats"][-1]["acmsa_route_sim_std"] = route_cache["route_sim_std"]
                if "same_grp_mean" in route_cache:
                    params["fdg_stats"][-1]["same_grp_mean"] = route_cache["same_grp_mean"]
                if "same_grp_std" in route_cache:
                    params["fdg_stats"][-1]["same_grp_std"] = route_cache["same_grp_std"]

        if (
            self.use_fdg
            and self.freq_grouping is not None
            and S_group is not None
            and self.fdg_use_td_ste
        ):
            # 中文注释：可选 FFN 条件分支 STE，仅用于消融，不作为默认主路径。
            soft_assign_td = F.softmax(S_group, dim=-1)
            hard_onehot = F.one_hot(tk_id, num_classes=S_group.shape[-1]).to(dtype=soft_assign_td.dtype)
            # 中文注释：前向等价于 hard one-hot，反向通过 softmax(S_group) 回传。
            ste_assign = hard_onehot + soft_assign_td - soft_assign_td.detach()
            x_td = ste_assign @ td_proj
            if params.get("fdg_stats", None):
                params["fdg_stats"][-1]["td_ste_enabled"] = True
                params["fdg_stats"][-1]["ste_assign_shape"] = tuple(ste_assign.shape)
                params["fdg_stats"][-1]["x_td_shape"] = tuple(x_td.shape)
        else:
            # 中文注释：默认使用原始 hard gather，保证 FDG 增益主要来自 AC-MSA route bias。
            x_td = torch.gather(
                td_proj,
                dim=1,
                index=tk_id.reshape(b, n, 1).expand(-1, -1, self.dim_ffn_td),
            )  # b, n, c
            if params.get("fdg_stats", None):
                params["fdg_stats"][-1]["td_ste_enabled"] = False
                params["fdg_stats"][-1]["x_td_shape"] = tuple(x_td.shape)

        # SW-MSA
        qkv = qkv.reshape(b, h, w, c3)

        # cyclic shift
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv
            attn_mask = None

        x_windows = window_partition(shifted_qkv, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)  # nw*b, window_size*window_size, c
        attn_windows = self.attn_win(x_windows, rpi=params['rpi_sa'], mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x # (B, H, W, C)

        # ============================
        # DAWA
        # ============================
        dawa_extra = None # DAWA
        if self.use_dawa: # DAWA # Sobel +
            if cached_direction is not None and cached_dir_conf is not None:
                direction_map, confidence_map = cached_direction, cached_dir_conf #
            else:
                x_img_for_dir = x.view(b, h, w, c)  # (B, H, W, C)
                direction_map, confidence_map = self.dir_estimator(x_img_for_dir)  # (B, H, W) 涓や釜
            thr = self.dawa_conf_threshold #
            denom = max(1.0 - thr, 1e-6) # 0
            confidence_map = ((confidence_map - thr).clamp(min=0.0)) / denom # [0,1]
            confidence_map = confidence_map.clamp(0.0, 1.0) #

            # QKV qkv cyclic shift
            qkv_for_stripe = qkv  # 褰㈢姸 (B, H, W, 3C)

            # irection=0
            h_stripe = (self.dawa_short, self.dawa_long)  # (Hs, Ws)
            hs_windows = stripe_partition(qkv_for_stripe, h_stripe)  # (nw*B, Hs, Ws, 3C)
            hs_windows = hs_windows.view(-1, h_stripe[0] * h_stripe[1], c3)  # (nw*B, Hs*Ws, 3C)
            hs_attn = self.attn_hstripe(hs_windows, mask=None)  # (nw*B, Hs*Ws, C)
            hs_attn = hs_attn.view(-1, h_stripe[0], h_stripe[1], c) #
            x_hstripe = stripe_reverse(hs_attn, h_stripe, h, w)  # (B, H, W, C)

            # irection=1
            v_stripe = (self.dawa_long, self.dawa_short)  # (Hs, Ws)
            vs_windows = stripe_partition(qkv_for_stripe, v_stripe)  # (nw*B, Hs, Ws, 3C)
            vs_windows = vs_windows.view(-1, v_stripe[0] * v_stripe[1], c3)  # (nw*B, Hs*Ws, 3C)
            vs_attn = self.attn_vstripe(vs_windows, mask=None)  # (nw*B, Hs*Ws, C)
            vs_attn = vs_attn.view(-1, v_stripe[0], v_stripe[1], c) #
            x_vstripe = stripe_reverse(vs_attn, v_stripe, h, w)  # (B, H, W, C)

            direction_map = direction_map.unsqueeze(-1).to(x_hstripe.dtype)  # (B,H,W,1)
            confidence_map = confidence_map.unsqueeze(-1).to(x_hstripe.dtype)  # (B,H,W,1)
            x_stripe_mix = (1.0 - direction_map) * x_hstripe + direction_map * x_vstripe #
            # DAWA tanh(0)=0
            dawa_alpha = torch.tanh(self.dawa_scale)
            # DAWA x_win
            dawa_extra = dawa_alpha * confidence_map * x_stripe_mix
            # DAWA token eshape view
            dawa_extra = dawa_extra.reshape(b, n, c)

        # ============================
        # GAA
        # ============================
        gaa_extra = None #
        if self.use_gaa: # GAA
            # FFT
            if cached_delta is not None:
                delta, theta, geo_conf = cached_delta, cached_theta, cached_geo_conf #
            else:
                x_img_for_geo = _select_gaa_estimator_input()  # (B, H, W, C)
                delta, theta, geo_conf = self.geo_estimator(x_img_for_geo)  # delta:(B,2,H,W), conf:(B,1,H,W)
            if self.enable_geo_debug and not self.geo_debug_cache:
                # validation forward batch
                with torch.no_grad():
                    # detach + cpu
                    self.geo_debug_cache = {
                        "geo_conf": geo_conf[0, 0].detach().float().cpu(),
                        "theta": theta[0, 0].detach().float().cpu(),
                        "delta_x": delta[0, 0].detach().float().cpu(),
                        "delta_y": delta[0, 1].detach().float().cpu(),
                    }
            # qkv SW MSA reshape (B H W 3C) (B N 3C) GAA
            qkv_seq = qkv.view(b, h * w, c3)  # (B, N, 3C)
            x_gaa = self.attn_gaa(qkv_seq, x_size=(h, w), delta=delta, geo_conf=geo_conf)  # (B, N, C)
            assert x_gaa.shape == (b, n, c), "GAA output shape must match input"
            # tanh(scale) ~0
            gaa_extra = torch.tanh(self.gaa_scale) * x_gaa  # (B, N, C)
        elif self.use_topk_retrieval: # GAA top-k
            qkv_seq = qkv.view(b, h * w, c3)  # (B, N, 3C)
            x_topk = self.attn_topk(qkv_seq, x_size=(h, w))  # (B, N, C)
            assert x_topk.shape == (b, n, c), "top-k output shape must match input"
            gaa_extra = torch.tanh(self.topk_scale) * x_topk  # (B, N, C)

        if self.use_decb and self.decb_branch is not None:
            # 中文注释：DECB query 使用进入 TDCA residual branch 前的内容特征 shortcut。
            query_feat_for_decb = shortcut
            # 中文注释：x_atd 只作为 gate 条件输入，不参与 error dictionary attention。
            x_err, decb_gate, decb_aux = self.decb_branch(
                query_feat=query_feat_for_decb,
                residual_feat=x_atd,
            )
            # 中文注释：防御性检查 x_err 与 x_atd 形状一致，避免 silent broadcast bug。
            if x_err.shape != x_atd.shape:
                raise RuntimeError("DECB x_err shape {} must match x_atd shape {}".format(
                    tuple(x_err.shape), tuple(x_atd.shape)
                ))
            # 中文注释：只从 TDCA / dictionary residual branch 中扣除错误 residual，不影响 logits/FFN/其它分支。
            x_atd = x_atd - decb_gate.to(dtype=x_atd.dtype, device=x_atd.device) * x_err.to(dtype=x_atd.dtype, device=x_atd.device)
            # 中文注释：把 gate 展平成每张图的向量，便于计算 per-image 诊断统计。
            gate_flat = decb_gate.detach().float().view(decb_gate.shape[0], -1)
            # 中文注释：把 error residual 展平成每张图的向量，便于计算 per-image 强度。
            err_flat = x_err.detach().float().view(x_err.shape[0], -1)
            # 中文注释：如果多个 block 启用，第一版只缓存最后一个启用 block 的 DECB 诊断值。
            residual_abs_mean = decb_aux.get("residual_abs_mean", None)
            params["decb_debug_cache"] = {
                "gate_mean": gate_flat.mean(dim=1, keepdim=True),
                "gate_std": gate_flat.std(dim=1, unbiased=False, keepdim=True),
                "gate_min": gate_flat.min(dim=1).values.view(-1, 1),
                "gate_max": gate_flat.max(dim=1).values.view(-1, 1),
                "err_abs_mean": err_flat.abs().mean(dim=1, keepdim=True),
                "attn_entropy": decb_aux["attn_entropy"].detach().float().view(-1, 1),
                "residual_abs_mean": residual_abs_mean.detach().float().view(x_atd.shape[0], 1) if residual_abs_mean is not None else None,
            }

        if (
            self.use_struct_prior
            and self.enable_struct_prior
            and params.get("struct_vec", None) is not None
        ):
            # 中文注释：struct_vec 来自原始 LR 固定结构提取器，送到与 x_atd 相同设备。
            struct_vec = params["struct_vec"].to(device=x_atd.device)
            # 中文注释：用原始 5 维结构统计计算日志用 structure_score，不依赖 global controller。
            struct_log = torch.log1p(struct_vec)
            # 中文注释：structure_score 仅用于日志/CSV/分析，不参与 local delta map 叠加。
            structure_score = (
                struct_log[:, 0:1]
                + struct_log[:, 1:2]
                + struct_log[:, 2:3]
                + struct_log[:, 4:5]
            ) / 4.0
            # 中文注释：local scope 第一版只使用局部 delta map，不叠加 global scalar delta。
            if params.get("struct_prior_scope", "global") == "local":
                # 中文注释：local_struct_map 来自原始 LR 的固定局部结构图。
                local_struct_map = params.get("local_struct_map", None)
                # 中文注释：local_struct_controller 是共享轻量 conv controller。
                local_struct_controller = params.get("local_struct_controller", None)
                if local_struct_map is not None and local_struct_controller is not None:
                    # 中文注释：把局部结构图送到当前 TDCA residual 所在设备。
                    local_struct_map = local_struct_map.to(device=x_atd.device)
                    # 中文注释：x_atd 为 B,N,C 时，使用当前 block 的 x_size 对齐局部 delta map。
                    if x_atd.dim() == 3:
                        local_delta_map = local_struct_controller(local_struct_map, target_hw=x_size)
                        # 中文注释：B,1,H,W -> B,N,1，用于逐 token 缩放 dictionary residual。
                        local_delta_token = local_delta_map.flatten(2).transpose(1, 2)
                        # 中文注释：防御性检查 token 数，避免 silent broadcast bug。
                        if local_delta_token.shape[1] != x_atd.shape[1]:
                            raise RuntimeError(
                                "Local SCDRC token size mismatch: delta N={} vs x_atd N={}".format(
                                    int(local_delta_token.shape[1]), int(x_atd.shape[1])
                                )
                            )
                        # 中文注释：只校准 TDCA / dictionary residual branch x_atd。
                        x_atd = x_atd * (1.0 + local_delta_token.to(dtype=x_atd.dtype, device=x_atd.device))
                    elif x_atd.dim() == 4:
                        # 中文注释：B,C,H,W 情况直接对齐到 residual 空间尺寸。
                        local_delta_map = local_struct_controller(local_struct_map, target_hw=x_atd.shape[-2:])
                        # 中文注释：只校准 TDCA / dictionary residual branch x_atd。
                        x_atd = x_atd * (1.0 + local_delta_map.to(dtype=x_atd.dtype, device=x_atd.device))
                    else:
                        raise ValueError("Unsupported x_atd dim for Local SCDRC: {}".format(x_atd.dim()))
                    # 中文注释：把 local delta map 聚合成 image-level delta，兼容旧日志和 CSV 字段。
                    image_level_delta = local_delta_map.flatten(1).mean(dim=1, keepdim=True)
                    # 中文注释：如果多个 block 启用，第一版只缓存最后一个启用 block 的 local delta 统计。
                    params["struct_debug_cache"] = {
                        "delta": image_level_delta.detach(),
                        "delta_std_spatial": local_delta_map.flatten(1).std(dim=1, unbiased=False).detach(),
                        "delta_min_spatial": local_delta_map.flatten(1).min(dim=1).values.detach(),
                        "delta_max_spatial": local_delta_map.flatten(1).max(dim=1).values.detach(),
                        "structure_score": structure_score.detach(),
                        "struct_vec": struct_vec.detach(),
                        "struct_norm": None,
                        "residual_vec": None,
                    }
            elif params.get("struct_controller", None) is not None:
                # 中文注释：global scope 保留原 image-level scalar SCDRC/SCDRC-RR 逻辑。
                tdca_out = x_atd
                # 中文注释：feat_ref 使用进入 dictionary residual branch 前的 block shortcut，形状同为 B,N,C。
                feat_ref = shortcut
                # 中文注释：默认普通 SCDRC 不使用 residual reliability。
                residual_vec = None
                # 中文注释：只有显式开启 SCDRC-RR 时才计算 residual 可靠性统计。
                if params.get("struct_use_residual_reliability", False):
                    residual_vec = compute_residual_reliability_stats(tdca_out, feat_ref)
                    # 中文注释：默认 detach stats，避免 controller 通过统计路径反向扰动 TDCA residual 本体。
                    if params.get("struct_residual_detach", True):
                        residual_vec = residual_vec.detach()
                # 中文注释：把 residual stats 对齐到 struct_vec dtype，避免 AMP 下 half/float cat 维度拼接报错。
                if residual_vec is not None:
                    residual_vec = residual_vec.to(device=x_atd.device, dtype=struct_vec.dtype)
                # 中文注释：共享 controller 在启用 SCDRC 的 block 内结合结构和可选 residual stats 生成当前 delta。
                struct_delta, structure_score, struct_norm = params["struct_controller"](
                    struct_vec,
                    residual_vec=residual_vec,
                )
                # 中文注释：struct_delta 是图像级 scalar，广播到 B,N,C，只缩放 dictionary residual x_atd。
                struct_delta_view = struct_delta.to(dtype=x_atd.dtype, device=x_atd.device).view(b, 1, 1)
                # 中文注释：SCDRC 公式 Y = X + (1+delta) * TDCA(X,D)，不影响 x_win/x_aca/FFN。
                x_atd = x_atd * (1.0 + struct_delta_view)
                # 中文注释：如果多个 block 启用，第一版只缓存最后一个启用 block 的调试值。
                params["struct_debug_cache"] = {
                    "delta": struct_delta.detach(),
                    "structure_score": structure_score.detach(),
                    "struct_vec": struct_vec.detach(),
                    "struct_norm": struct_norm.detach(),
                    "residual_vec": residual_vec.detach() if residual_vec is not None else None,
                }

        zero_dict_residual = params is not None and bool(params.get("zero_dict_residual", False))
        if zero_dict_residual:
            # 中文注释：zero_dict_residual 仅用于分析 dictionary residual 帮助度。
            # 中文注释：只置零 TDCA / ATD dictionary residual x_atd，不影响 window attention、ACA 或 FFN。
            x_atd = x_atd * 0.0
        elif (
            self.use_radr
            and self.enable_radr
            and self.radr_predictor is not None
            and not zero_dict_residual
            and params is not None
        ):
            # 中文注释：RADR 只看 LR 可见的当前内容特征 shortcut，不读取 HR/SR 误差。
            radr_feat = shortcut.detach() if self.radr_detach_feat else shortcut
            u_hat, u_logit = self.radr_predictor(radr_feat)
            # 中文注释：v1.1 只抑制高不可靠概率区域；不 hard zero、不增强 residual。
            tau = float(self.radr_tau)
            denom = max(1.0 - tau, 1e-6)
            u_eff = ((u_hat - tau) / denom).clamp(0.0, 1.0)
            x_atd_orig = x_atd
            disable_suppression = bool(params.get("radr_disable_suppression", False))
            if disable_suppression:
                x_supp = x_atd_orig
                suppression = torch.zeros_like(u_eff.detach().float())
            else:
                x_supp = (
                    1.0 - self.radr_lambda * u_eff.to(dtype=x_atd_orig.dtype, device=x_atd_orig.device)
                ) * x_atd_orig
                suppression = float(self.radr_lambda) * u_eff.detach().float()
            x_corr = None
            corr_gate = None
            corr_feature_mode = str(getattr(self, "radr_corr_feature_mode", "shortcut_xatd"))
            corr_feature_mode_id_value = 0.0
            corr_context_abs_mean = None
            corr_context_abs_std = None
            disable_correction = bool(params.get("radr_disable_correction", False))
            if self.radr_use_correction and self.radr_correction is not None and not disable_correction:
                corr_feature_mode = self._select_radr_corr_feature_mode(device=u_eff.device)
                corr_context, corr_feature_mode_id_value = self._build_radr_corr_context(
                    corr_feature_mode,
                    shortcut=shortcut,
                    x_atd_orig=x_atd_orig,
                    x_win=x_win,
                    x_aca=x_aca,
                    x_size=x_size,
                )
                if corr_context.shape != shortcut.shape:
                    raise RuntimeError(
                        "RADR correction context shape {} must match shortcut {}".format(
                            tuple(corr_context.shape), tuple(shortcut.shape)
                        )
                    )
                x_corr = self.radr_correction(shortcut, corr_context)
                if x_corr.shape != x_atd_orig.shape:
                    raise RuntimeError(
                        "RADR correction output shape {} must match x_atd {}".format(
                            tuple(x_corr.shape), tuple(x_atd_orig.shape)
                        )
                    )
                corr_gate = self._compute_radr_corr_gate(u_eff)
                x_atd = x_supp + self.radr_corr_lambda * corr_gate.to(
                    dtype=x_corr.dtype, device=x_corr.device
                ) * x_corr
                if bool(params.get("capture_radr_ccd", False)) and self.radr_ccd_aux_head is not None:
                    if self.radr_ccd_aux_use_gated_corr and corr_gate is not None:
                        aux_feat = corr_gate.to(dtype=x_corr.dtype, device=x_corr.device) * x_corr
                    else:
                        aux_feat = x_corr
                    delta_pred = self.radr_ccd_aux_head(aux_feat, x_size=x_size)
                    params.setdefault("radr_ccd_delta_list", []).append(delta_pred)
                corr_context_det = corr_context.detach().float()
                corr_context_flat = corr_context_det.view(corr_context_det.shape[0], -1)
                corr_context_abs_mean = corr_context_flat.abs().mean(dim=1, keepdim=True)
                corr_context_abs_std = corr_context_flat.abs().std(dim=1, unbiased=False, keepdim=True)
            else:
                x_atd = x_supp
            u_hat_det = u_hat.detach().float()
            u_eff_det = u_eff.detach().float()
            u_flat = u_hat_det.view(u_hat_det.shape[0], -1)
            u_eff_flat = u_eff_det.view(u_eff_det.shape[0], -1)
            supp_flat = suppression.view(suppression.shape[0], -1)
            corr_enabled = torch.full(
                (u_hat_det.shape[0], 1),
                float(x_corr is not None),
                device=u_hat_det.device,
                dtype=u_hat_det.dtype,
            )
            corr_lambda = torch.full(
                (u_hat_det.shape[0], 1),
                float(self.radr_corr_lambda),
                device=u_hat_det.device,
                dtype=u_hat_det.dtype,
            )
            if x_corr is not None:
                corr_det = x_corr.detach().float()
                corr_flat = corr_det.view(corr_det.shape[0], -1)
                corr_gate_det = corr_gate.detach().float()
                corr_gate_flat = corr_gate_det.view(corr_gate_det.shape[0], -1)
                gated_corr = (
                    float(self.radr_corr_lambda)
                    * corr_gate_det.to(device=corr_det.device, dtype=corr_det.dtype)
                    * corr_det
                )
                gated_corr_flat = gated_corr.view(gated_corr.shape[0], -1)
                corr_abs_mean = corr_flat.abs().mean(dim=1, keepdim=True)
                corr_abs_std = corr_flat.abs().std(dim=1, unbiased=False, keepdim=True)
                corr_gated_abs_mean = gated_corr_flat.abs().mean(dim=1, keepdim=True)
                corr_gated_abs_std = gated_corr_flat.abs().std(dim=1, unbiased=False, keepdim=True)
                corr_gate_mean = corr_gate_flat.mean(dim=1, keepdim=True)
                corr_gate_std = corr_gate_flat.std(dim=1, unbiased=False, keepdim=True)
                corr_gate_min = corr_gate_flat.min(dim=1).values.view(-1, 1)
                corr_gate_max = corr_gate_flat.max(dim=1).values.view(-1, 1)
            else:
                zeros = torch.zeros((u_hat_det.shape[0], 1), device=u_hat_det.device, dtype=u_hat_det.dtype)
                corr_abs_mean = zeros
                corr_abs_std = zeros
                corr_gated_abs_mean = zeros
                corr_gated_abs_std = zeros
                corr_gate_mean = zeros
                corr_gate_std = zeros
                corr_gate_min = zeros
                corr_gate_max = zeros
            mode_to_id = {"ueff": 0.0, "sqrt": 1.0, "binary": 2.0, "none": 3.0}
            corr_gate_mode_id = torch.full(
                (u_hat_det.shape[0], 1),
                float(mode_to_id.get(self.radr_corr_gate_mode, -1.0)),
                device=u_hat_det.device,
                dtype=u_hat_det.dtype,
            )
            corr_feature_mode_id = torch.full(
                (u_hat_det.shape[0], 1),
                float(corr_feature_mode_id_value),
                device=u_hat_det.device,
                dtype=u_hat_det.dtype,
            )
            if corr_context_abs_mean is None:
                zeros = torch.zeros((u_hat_det.shape[0], 1), device=u_hat_det.device, dtype=u_hat_det.dtype)
                corr_context_abs_mean = zeros
                corr_context_abs_std = zeros
            params["radr_debug_cache"] = {
                "u_mean": u_flat.mean(dim=1, keepdim=True),
                "u_std": u_flat.std(dim=1, unbiased=False, keepdim=True),
                "u_min": u_flat.min(dim=1).values.view(-1, 1),
                "u_max": u_flat.max(dim=1).values.view(-1, 1),
                "u_eff_mean": u_eff_flat.mean(dim=1, keepdim=True),
                "u_eff_std": u_eff_flat.std(dim=1, unbiased=False, keepdim=True),
                "u_eff_min": u_eff_flat.min(dim=1).values.view(-1, 1),
                "u_eff_max": u_eff_flat.max(dim=1).values.view(-1, 1),
                "lambda": torch.full(
                    (u_hat_det.shape[0], 1),
                    float(self.radr_lambda),
                    device=u_hat_det.device,
                    dtype=u_hat_det.dtype,
                ),
                "tau": torch.full(
                    (u_hat_det.shape[0], 1),
                    float(self.radr_tau),
                    device=u_hat_det.device,
                    dtype=u_hat_det.dtype,
                ),
                "suppression_mean": supp_flat.mean(dim=1, keepdim=True),
                "corr_enabled": corr_enabled,
                "corr_lambda": corr_lambda,
                "corr_abs_mean": corr_abs_mean,
                "corr_abs_std": corr_abs_std,
                "corr_gated_abs_mean": corr_gated_abs_mean,
                "corr_gated_abs_std": corr_gated_abs_std,
                "corr_gate_mode_id": corr_gate_mode_id,
                "corr_gate_mean": corr_gate_mean,
                "corr_gate_std": corr_gate_std,
                "corr_gate_min": corr_gate_min,
                "corr_gate_max": corr_gate_max,
                "corr_feature_mode_id": corr_feature_mode_id,
                "corr_context_abs_mean": corr_context_abs_mean,
                "corr_context_abs_std": corr_context_abs_std,
            }
            params["radr_train_cache"] = {
                "u_hat": u_hat,
                "u_logit": u_logit,
                "u_eff": u_eff,
                "spatial_shape": tuple(x_size) if u_hat.dim() == 3 else tuple(u_hat.shape[-2:]),
            }

        x = shortcut + x_atd + x_win.view(b, n, c) + x_aca
        if dawa_extra is not None:
            # 中文注释：仅在启用 DAWA 时叠加并行残差。
            x = x + dawa_extra
        if gaa_extra is not None:
            # 中文注释：仅在启用 GAA 或 top-k 时叠加并行残差。
            x = x + gaa_extra

        # FFN
        x = x + self.convffn(self.norm2(x), x_td, x_size)

        return x


    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution

        # qkv = self.wqkv(x)
        flops += self.dim * 3 * self.dim * h * w

        # SWMSA, ATDCA, ACMSA
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size)
        flops += self.attn_atd.flops(h * w)
        flops += self.attn_aca.flops(h * w)

        # DAWA flops + use_dawa=False
        if self.use_dawa:
            n_stripe = self.dawa_short * self.dawa_long # token
            nw_stripe = h * w / n_stripe #
            flops += nw_stripe * self.attn_hstripe.flops(n_stripe)  # 妯悜鏉″甫娉ㄦ剰鍔?flops
            flops += nw_stripe * self.attn_vstripe.flops(n_stripe) # flops

        # GAA flops
        if self.use_gaa:
            flops += self.attn_gaa.flops(h * w) # flops
        # top k
        if self.use_topk_retrieval:
            flops += self.attn_topk.flops(h * w)  # top-k 妫€绱?flops

        # mlp
        flops += h * w * self.dim * (self.dim*2 + self.dim_ffn_td) * self.mlp_ratio
        flops += h * w * (self.dim + self.dim_ffn_td) * self.convffn_kernel_size**2 * self.mlp_ratio

        return flops



class BasicBlock(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 idx,
                 depth,
                 num_heads,
                 window_size,
                 dim_ffn_td,
                 category_size,
                 num_tokens,
                 convffn_kernel_size,
                 reducted_dim,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 use_dawa=False,
                 dawa_long=32,
                 dawa_short=8,
                 dawa_conf_threshold=0.15,
                 dawa_layer_mode="last",
                 use_gaa=False,
                 gaa_layer_mode="last",
                 gaa_num_geo=2,
                 gaa_max_offset=16,
                 gaa_block_size=32,
                 gaa_estimator_source="lr_gray",
                 use_topk_retrieval=False,
                 topk=8,
                 topk_max_tokens=4096,
                 use_gdc=False,
                 geo_feat_dim=5,
                 use_fdg=False,
                 fdg_in_channels=5,
                 fdg_df=16,
                 fdg_lambda0=1.0,
                 fdg_init_alpha=0.01,
                 fdg_layer_mode="last",
                 fdg_use_acmsa_route_bias=True,
                 fdg_route_bias_max=1.0,
                 fdg_route_bias_init=0.0,
                 fdg_route_bias_detach=False,
                 fdg_use_td_ste=False,
                 fdg_use_gumbel_grouping=False,
                 fdg_tau_gumbel=1.0,
                 fdg_use_routeprob_bias=False,
                 use_fcdm=False,
                 fcdm_desc_dim=6,
                 fcdm_hidden_dim=64,
                 fcdm_scale_init=0.0,
                 fcdm_layer_mode="last",
                 use_fats=False,
                 fats_hidden_dim=32,
                 fats_tau_range=0.3,
                 fats_layer_mode="last",
                 use_lfcdm=False,
                 lfcdm_desc_dim=6,
                 lfcdm_hidden_dim=32,
                 lfcdm_gate_bias=0.0,
                 use_struct_prior=False,
                 struct_layer_mode="last",
                 use_radr=False,
                 radr_layer_mode="last",
                 radr_hidden_dim=32,
                 radr_lambda=0.10,
                 radr_tau=0.50,
                 radr_init_bias=-4.0,
                 radr_detach_feat=True,
                 radr_use_correction=False,
                 radr_corr_hidden_dim=64,
                 radr_corr_lambda=0.05,
                 radr_corr_scale=0.10,
                 radr_corr_init_std=1e-4,
                 radr_corr_detach_residual=True,
                 radr_corr_gate_mode="ueff",
                 radr_corr_feature_mode="shortcut_xatd",
                 radr_corr_train_feature_modes="",
                 use_radr_ccd=False,
                 radr_ccd_upscale=4,
                 radr_ccd_aux_init_std=1e-4,
                 radr_ccd_aux_scale=0.10,
                 radr_ccd_aux_use_gated_corr=True,
                 use_radr_lch=False,
                 radr_lch_hidden_dim=64,
                 radr_lch_lambda=0.05,
                 radr_lch_corr_scale=0.05,
                 radr_lch_init_std=1e-4,
                 radr_lch_gate_mode="u_eff",
                 radr_lch_detach_map=True,
                 use_decb=False,
                 decb_num_tokens=64,
                 decb_gate_max=0.10,
                 decb_gate_init=-4.0,
                 decb_token_init_std=0.02,
                 decb_proj_init_std=0.001,
                 decb_layer_mode="last",
                 decb_gate_condition="shortcut",
                 decb_residual_detach=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.idx = idx

        # 中文注释：仅保留 DAWA/GAA/top-k/GDC 的原有层级控制。
        valid_modes = ["none", "last", "interval2", "interval3", "all"]
        self.dawa_layer_mode = str(dawa_layer_mode)
        self.gaa_layer_mode = str(gaa_layer_mode)
        self.fdg_layer_mode = str(fdg_layer_mode)
        self.struct_layer_mode = str(struct_layer_mode)
        self.radr_layer_mode = str(radr_layer_mode)
        self.decb_layer_mode = str(decb_layer_mode)
        if self.dawa_layer_mode not in valid_modes:
            raise ValueError("dawa_layer_mode must be one of {}, got {!r}".format(valid_modes, self.dawa_layer_mode))
        if self.gaa_layer_mode not in valid_modes:
            raise ValueError("gaa_layer_mode must be one of {}, got {!r}".format(valid_modes, self.gaa_layer_mode))
        if self.fdg_layer_mode not in valid_modes:
            raise ValueError("fdg_layer_mode must be one of {}, got {!r}".format(valid_modes, self.fdg_layer_mode))
        if self.struct_layer_mode not in valid_modes:
            raise ValueError("struct_layer_mode must be one of {}, got {!r}".format(valid_modes, self.struct_layer_mode))
        if self.radr_layer_mode not in valid_modes:
            raise ValueError("radr_layer_mode must be one of {}, got {!r}".format(valid_modes, self.radr_layer_mode))
        if self.decb_layer_mode not in valid_modes:
            raise ValueError("decb_layer_mode must be one of {}, got {!r}".format(valid_modes, self.decb_layer_mode))
        self.gaa_estimator_source = str(gaa_estimator_source)
        valid_gaa_sources = ["current", "shallow", "shallow_detach", "lr_gray"]
        if self.gaa_estimator_source not in valid_gaa_sources:
            raise ValueError("gaa_estimator_source must be one of {}, got {!r}".format(valid_gaa_sources, self.gaa_estimator_source))

        def _layer_enabled(global_flag, mode, layer_idx):
            # 中文注释：复用原有 none/last/interval/all 层级选择规则。
            if not global_flag or mode == "none":
                return False
            if mode == "all":
                return True
            if mode == "last":
                return layer_idx == depth - 1
            if mode == "interval2":
                return layer_idx % 2 == 0
            if mode == "interval3":
                return layer_idx % 3 == 0
            return False

        self.layers = nn.ModuleList()
        for i in range(depth):
            layer_use_dawa = _layer_enabled(use_dawa, self.dawa_layer_mode, i)
            layer_use_gaa = _layer_enabled(use_gaa, self.gaa_layer_mode, i)
            layer_use_topk = _layer_enabled(use_topk_retrieval, self.gaa_layer_mode, i)
            layer_use_fdg = _layer_enabled(use_fdg, self.fdg_layer_mode, i)
            layer_use_gdc = bool(use_gdc and (layer_use_gaa or layer_use_dawa))
            # 中文注释：旧频率调制路径在方案 B 中禁用，避免污染 S_content。
            layer_use_fcdm = False
            layer_use_fats = False
            layer_use_lfcdm = False
            # 中文注释：SCDRC 按每个 residual group 内 block index 控制，默认只启用最后一个 TDCA block。
            layer_use_struct = bool(use_struct_prior) and should_enable_struct_prior(i, depth, self.struct_layer_mode)
            # 中文注释：RADR 第一版默认只启用每个 residual group 最后一个 TDCA block。
            layer_use_radr = bool(use_radr) and should_enable_radr(i, depth, self.radr_layer_mode)
            # 中文注释：DECB 按每个 residual group 内 block index 控制，第一版默认只启用最后一个 TDCA block。
            layer_use_decb = bool(use_decb) and should_enable_decb(i, depth, self.decb_layer_mode)

            self.layers.append(
                ATDTransformerLayer(
                    dim=dim,
                    idx=i,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    dim_ffn_td=dim_ffn_td,
                    category_size=category_size,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    reducted_dim=reducted_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    use_dawa=layer_use_dawa,
                    dawa_long=dawa_long,
                    dawa_short=dawa_short,
                    dawa_conf_threshold=dawa_conf_threshold,
                    use_gaa=layer_use_gaa,
                    gaa_num_geo=gaa_num_geo,
                    gaa_max_offset=gaa_max_offset,
                    gaa_block_size=gaa_block_size,
                    gaa_estimator_source=self.gaa_estimator_source,
                    use_topk_retrieval=layer_use_topk,
                    topk=topk,
                    topk_max_tokens=topk_max_tokens,
                    use_gdc=layer_use_gdc,
                    geo_feat_dim=geo_feat_dim,
                    use_fdg=layer_use_fdg,
                    fdg_in_channels=fdg_in_channels,
                    fdg_df=fdg_df,
                    fdg_lambda0=fdg_lambda0,
                    fdg_init_alpha=fdg_init_alpha,
                    fdg_layer_mode=self.fdg_layer_mode,
                    fdg_use_acmsa_route_bias=fdg_use_acmsa_route_bias,
                    fdg_route_bias_max=fdg_route_bias_max,
                    fdg_route_bias_init=fdg_route_bias_init,
                    fdg_route_bias_detach=fdg_route_bias_detach,
                    fdg_use_td_ste=fdg_use_td_ste,
                    fdg_use_gumbel_grouping=fdg_use_gumbel_grouping,
                    fdg_tau_gumbel=fdg_tau_gumbel,
                    fdg_use_routeprob_bias=fdg_use_routeprob_bias,
                    use_fcdm=layer_use_fcdm,
                    fcdm_desc_dim=fcdm_desc_dim,
                    fcdm_hidden_dim=fcdm_hidden_dim,
                    fcdm_scale_init=fcdm_scale_init,
                    use_fats=layer_use_fats,
                    fats_hidden_dim=fats_hidden_dim,
                    fats_tau_range=fats_tau_range,
                    use_lfcdm=layer_use_lfcdm,
                    lfcdm_desc_dim=lfcdm_desc_dim,
                    lfcdm_hidden_dim=lfcdm_hidden_dim,
                    lfcdm_gate_bias=lfcdm_gate_bias,
                    use_struct_prior=use_struct_prior,
                    enable_struct_prior=layer_use_struct,
                    use_radr=use_radr,
                    enable_radr=layer_use_radr,
                    radr_hidden_dim=radr_hidden_dim,
                    radr_lambda=radr_lambda,
                    radr_tau=radr_tau,
                    radr_init_bias=radr_init_bias,
                    radr_detach_feat=radr_detach_feat,
                    radr_use_correction=radr_use_correction,
                    radr_corr_hidden_dim=radr_corr_hidden_dim,
                    radr_corr_lambda=radr_corr_lambda,
                    radr_corr_scale=radr_corr_scale,
                    radr_corr_init_std=radr_corr_init_std,
                    radr_corr_detach_residual=radr_corr_detach_residual,
                    radr_corr_gate_mode=radr_corr_gate_mode,
                    radr_corr_feature_mode=radr_corr_feature_mode,
                    radr_corr_train_feature_modes=radr_corr_train_feature_modes,
                    use_radr_ccd=use_radr_ccd,
                    radr_ccd_upscale=radr_ccd_upscale,
                    radr_ccd_aux_init_std=radr_ccd_aux_init_std,
                    radr_ccd_aux_scale=radr_ccd_aux_scale,
                    radr_ccd_aux_use_gated_corr=radr_ccd_aux_use_gated_corr,
                    use_decb=layer_use_decb,
                    decb_num_tokens=decb_num_tokens,
                    decb_gate_max=decb_gate_max,
                    decb_gate_init=decb_gate_init,
                    decb_token_init_std=decb_token_init_std,
                    decb_proj_init_std=decb_proj_init_std,
                    decb_gate_condition=decb_gate_condition,
                    decb_residual_detach=decb_residual_detach,
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # Token Dictionary
        self.td = nn.Parameter(torch.randn([num_tokens, dim]), requires_grad=True)

    def forward(self, x, x_size, params):
        b, n, c = x.shape
        td = self.td.expand([b, -1, -1])
        idx_checkpoint = 5
        for layer in self.layers:
            if self.use_checkpoint and self.idx < idx_checkpoint:
                x = checkpoint(layer, x, td, x_size, params, use_reentrant=False)
            else:
                x = layer(x, td, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"depth={self.depth}, dawa_layer_mode={self.dawa_layer_mode}, "
            f"gaa_layer_mode={self.gaa_layer_mode}, "
            f"fdg_layer_mode={self.fdg_layer_mode}, "
            f"struct_layer_mode={self.struct_layer_mode}, "
            f"decb_layer_mode={self.decb_layer_mode}, "
            f"gaa_estimator_source={self.gaa_estimator_source}"
        )

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops


class ATDB(nn.Module):
    def __init__(self,
                 dim,
                 idx,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 dim_ffn_td,
                 category_size,
                 num_tokens,
                 reducted_dim,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection="1conv",
                 use_dawa=False,
                 dawa_long=32,
                 dawa_short=8,
                 dawa_conf_threshold=0.15,
                 dawa_layer_mode="last",
                 use_gaa=False,
                 gaa_layer_mode="last",
                 gaa_num_geo=2,
                 gaa_max_offset=16,
                 gaa_block_size=32,
                 gaa_estimator_source="lr_gray",
                 use_topk_retrieval=False,
                 topk=8,
                 topk_max_tokens=4096,
                 use_gdc=False,
                 geo_feat_dim=5,
                 use_fdg=False,
                 fdg_in_channels=5,
                 fdg_df=16,
                 fdg_lambda0=1.0,
                 fdg_init_alpha=0.01,
                 fdg_layer_mode="last",
                 fdg_use_acmsa_route_bias=True,
                 fdg_route_bias_max=1.0,
                 fdg_route_bias_init=0.0,
                 fdg_route_bias_detach=False,
                 fdg_use_td_ste=False,
                 fdg_use_gumbel_grouping=False,
                 fdg_tau_gumbel=1.0,
                 fdg_use_routeprob_bias=False,
                 use_fcdm=False,
                 fcdm_desc_dim=6,
                 fcdm_hidden_dim=64,
                 fcdm_scale_init=0.0,
                 fcdm_layer_mode="last",
                 use_fats=False,
                 fats_hidden_dim=32,
                 fats_tau_range=0.3,
                 fats_layer_mode="last",
                 use_lfcdm=False,
                 lfcdm_desc_dim=6,
                 lfcdm_hidden_dim=32,
                 lfcdm_gate_bias=0.0,
                 use_struct_prior=False,
                 struct_layer_mode="last",
                 use_radr=False,
                 radr_layer_mode="last",
                 radr_hidden_dim=32,
                 radr_lambda=0.10,
                 radr_tau=0.50,
                 radr_init_bias=-4.0,
                 radr_detach_feat=True,
                 radr_use_correction=False,
                 radr_corr_hidden_dim=64,
                 radr_corr_lambda=0.05,
                 radr_corr_scale=0.10,
                 radr_corr_init_std=1e-4,
                 radr_corr_detach_residual=True,
                 radr_corr_gate_mode="ueff",
                 radr_corr_feature_mode="shortcut_xatd",
                 radr_corr_train_feature_modes="",
                 use_radr_ccd=False,
                 radr_ccd_upscale=4,
                 radr_ccd_aux_init_std=1e-4,
                 radr_ccd_aux_scale=0.10,
                 radr_ccd_aux_use_gated_corr=True,
                 use_radr_lch=False,
                 radr_lch_hidden_dim=64,
                 radr_lch_lambda=0.05,
                 radr_lch_corr_scale=0.05,
                 radr_lch_init_std=1e-4,
                 radr_lch_gate_mode="u_eff",
                 radr_lch_detach_map=True,
                 use_decb=False,
                 decb_num_tokens=64,
                 decb_gate_max=0.10,
                 decb_gate_init=-4.0,
                 decb_token_init_std=0.02,
                 decb_proj_init_std=0.001,
                 decb_layer_mode="last",
                 decb_gate_condition="shortcut",
                 decb_residual_detach=True):
        super(ATDB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim)

        self.residual_group = BasicBlock(
            dim=dim,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            num_tokens=num_tokens,
            dim_ffn_td=dim_ffn_td,
            category_size=category_size,
            reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            use_dawa=use_dawa,
            dawa_long=dawa_long,
            dawa_short=dawa_short,
            dawa_conf_threshold=dawa_conf_threshold,
            dawa_layer_mode=dawa_layer_mode,
            use_gaa=use_gaa,
            gaa_layer_mode=gaa_layer_mode,
            gaa_num_geo=gaa_num_geo,
            gaa_max_offset=gaa_max_offset,
            gaa_block_size=gaa_block_size,
            gaa_estimator_source=gaa_estimator_source,
            use_topk_retrieval=use_topk_retrieval,
            topk=topk,
            topk_max_tokens=topk_max_tokens,
            use_gdc=use_gdc,
            geo_feat_dim=geo_feat_dim,
            use_fdg=use_fdg,
            fdg_in_channels=fdg_in_channels,
            fdg_df=fdg_df,
            fdg_lambda0=fdg_lambda0,
            fdg_init_alpha=fdg_init_alpha,
            fdg_layer_mode=fdg_layer_mode,
            fdg_use_acmsa_route_bias=fdg_use_acmsa_route_bias,
            fdg_route_bias_max=fdg_route_bias_max,
            fdg_route_bias_init=fdg_route_bias_init,
            fdg_route_bias_detach=fdg_route_bias_detach,
            fdg_use_td_ste=fdg_use_td_ste,
            fdg_use_gumbel_grouping=fdg_use_gumbel_grouping,
            fdg_tau_gumbel=fdg_tau_gumbel,
            fdg_use_routeprob_bias=fdg_use_routeprob_bias,
            use_fcdm=False,
            fcdm_desc_dim=fcdm_desc_dim,
            fcdm_hidden_dim=fcdm_hidden_dim,
            fcdm_scale_init=fcdm_scale_init,
            fcdm_layer_mode=fcdm_layer_mode,
            use_fats=False,
            fats_hidden_dim=fats_hidden_dim,
            fats_tau_range=fats_tau_range,
            fats_layer_mode=fats_layer_mode,
            use_lfcdm=False,
            lfcdm_desc_dim=lfcdm_desc_dim,
            lfcdm_hidden_dim=lfcdm_hidden_dim,
            lfcdm_gate_bias=lfcdm_gate_bias,
            use_struct_prior=use_struct_prior,
            struct_layer_mode=struct_layer_mode,
            use_radr=use_radr,
            radr_layer_mode=radr_layer_mode,
            radr_hidden_dim=radr_hidden_dim,
            radr_lambda=radr_lambda,
            radr_tau=radr_tau,
            radr_init_bias=radr_init_bias,
            radr_detach_feat=radr_detach_feat,
            radr_use_correction=radr_use_correction,
            radr_corr_hidden_dim=radr_corr_hidden_dim,
            radr_corr_lambda=radr_corr_lambda,
            radr_corr_scale=radr_corr_scale,
            radr_corr_init_std=radr_corr_init_std,
            radr_corr_detach_residual=radr_corr_detach_residual,
            radr_corr_gate_mode=radr_corr_gate_mode,
            radr_corr_feature_mode=radr_corr_feature_mode,
            radr_corr_train_feature_modes=radr_corr_train_feature_modes,
            use_radr_ccd=use_radr_ccd,
            radr_ccd_upscale=radr_ccd_upscale,
            radr_ccd_aux_init_std=radr_ccd_aux_init_std,
            radr_ccd_aux_scale=radr_ccd_aux_scale,
            radr_ccd_aux_use_gated_corr=radr_ccd_aux_use_gated_corr,
            use_decb=use_decb,
            decb_num_tokens=decb_num_tokens,
            decb_gate_max=decb_gate_max,
            decb_gate_init=decb_gate_init,
            decb_token_init_std=decb_token_init_std,
            decb_proj_init_std=decb_proj_init_std,
            decb_layer_mode=decb_layer_mode,
            decb_gate_condition=decb_gate_condition,
            decb_residual_detach=decb_residual_detach,
        )
        self.norm = norm_layer(dim)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size, params):
        # return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x
        return self.norm(self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x)

    def flops(self, input_resolution=None):
        flops = 0
        flops += self.residual_group.flops(input_resolution)
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops(input_resolution)
        flops += self.patch_unembed.flops(input_resolution)

        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        h, w = self.patches_resolution if input_resolution is None else input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


@ARCH_REGISTRY.register()
class ATD(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=90,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=8,
                 dim_ffn_td=16,
                 category_size=128,
                 num_tokens=64,
                 reducted_dim=4,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler="",
                 resi_connection="1conv",
                 atd_use_dawa=False,
                 atd_dawa_long=32,
                 atd_dawa_short=8,
                 atd_dawa_conf_threshold=0.15,
                 atd_dawa_layer_mode="last",
                 atd_use_gaa=False,
                 atd_gaa_layer_mode="last",
                 atd_gaa_num_geo=2,
                 atd_gaa_max_offset=16,
                 atd_gaa_block_size=32,
                 atd_gaa_estimator_source="lr_gray",
                 atd_use_topk_retrieval=False,
                 atd_topk=8,
                 atd_topk_max_tokens=4096,
                 atd_use_gdc=False,
                 atd_geo_feat_dim=5,
                 use_fdg=False,
                 fdg_df=16,
                 fdg_lambda0=1.0,
                 fdg_init_alpha=0.01,
                 fdg_in_channels=5,
                 fdg_layer_mode="last",
                 fdg_use_acmsa_route_bias=True,
                 fdg_route_bias_max=1.0,
                 fdg_route_bias_init=0.0,
                 fdg_route_bias_detach=False,
                 fdg_use_td_ste=False,
                 fdg_use_gumbel_grouping=False,
                 fdg_tau_gumbel=1.0,
                 fdg_use_routeprob_bias=False,
                 atd_use_fcdm=False,
                 atd_fcdm_desc_dim=6,
                 atd_fcdm_hidden_dim=64,
                 atd_fcdm_scale_init=0.0,
                 atd_fcdm_layer_mode="last",
                 atd_use_fats=False,
                 atd_fats_hidden_dim=32,
                 atd_fats_tau_range=0.3,
                 atd_fats_layer_mode="last",
                 atd_use_lfcdm=False,
                 atd_lfcdm_desc_dim=6,
                 atd_lfcdm_hidden_dim=32,
                 atd_lfcdm_window_size=16,
                 atd_lfcdm_gate_bias=0.0,
                 atd_lfcdm_detach=True,
                 use_struct_prior=False,
                 struct_prior_type="scdrc",
                 struct_scale_max=0.15,
                 struct_delta_init=0.0,
                 struct_last_weight_std=0.0,
                 struct_use_residual_reliability=False,
                 struct_residual_detach=True,
                 struct_rr_stats="abs_mean,abs_std,ratio",
                 struct_prior_scope="global",
                 struct_local_channels=16,
                 struct_local_scale_max=0.05,
                 struct_local_last_weight_std=0.003,
                 struct_local_detach=True,
                 struct_layer_mode="last",
                 struct_hidden_dim=64,
                 use_radr=False,
                 radr_layer_mode="last",
                 radr_hidden_dim=32,
                 radr_lambda=0.10,
                 radr_tau=0.50,
                 radr_init_bias=-4.0,
                 radr_detach_feat=True,
                 radr_use_correction=False,
                 radr_corr_hidden_dim=64,
                 radr_corr_lambda=0.05,
                 radr_corr_scale=0.10,
                 radr_corr_init_std=1e-4,
                 radr_corr_detach_residual=True,
                 radr_corr_gate_mode="ueff",
                 radr_corr_feature_mode="shortcut_xatd",
                 radr_corr_train_feature_modes="",
                 use_radr_ccd=False,
                 radr_ccd_upscale=4,
                 radr_ccd_aux_init_std=1e-4,
                 radr_ccd_aux_scale=0.10,
                 radr_ccd_aux_use_gated_corr=True,
                 use_radr_lch=False,
                 radr_lch_hidden_dim=64,
                 radr_lch_lambda=0.05,
                 radr_lch_corr_scale=0.05,
                 radr_lch_init_std=1e-4,
                 radr_lch_gate_mode="u_eff",
                 radr_lch_detach_map=True,
                 use_scdr=False,
                 scdr_alpha=0.49,
                 scdr_return_routes=False,
                 scdr_detach_routes=False,
                 use_scdr_adapter=False,
                 scdr_adapter_hidden_dim=32,
                 scdr_adapter_scale=0.05,
                 scdr_adapter_init_std=1e-5,
                 use_decb=False,
                 decb_num_tokens=64,
                 decb_gate_max=0.10,
                 decb_gate_init=-4.0,
                 decb_token_init_std=0.02,
                 decb_proj_init_std=0.001,
                 decb_layer_mode="last",
                 decb_gate_condition="shortcut",
                 decb_residual_detach=True,
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # 中文注释：保留 polished ATD baseline 的 DAWA/GAA/top-k/GDC 可选配置。
        valid_layer_modes = ["none", "last", "interval2", "interval3", "all"]
        self.atd_use_dawa = bool(atd_use_dawa)
        self.atd_dawa_long = int(atd_dawa_long)
        self.atd_dawa_short = int(atd_dawa_short)
        self.atd_dawa_conf_threshold = float(atd_dawa_conf_threshold)
        self.atd_dawa_layer_mode = str(atd_dawa_layer_mode)
        if self.atd_dawa_layer_mode not in valid_layer_modes:
            raise ValueError("atd_dawa_layer_mode must be one of {}, got {!r}".format(valid_layer_modes, self.atd_dawa_layer_mode))
        self.atd_use_gaa = bool(atd_use_gaa)
        self.atd_gaa_layer_mode = str(atd_gaa_layer_mode)
        if self.atd_gaa_layer_mode not in valid_layer_modes:
            raise ValueError("atd_gaa_layer_mode must be one of {}, got {!r}".format(valid_layer_modes, self.atd_gaa_layer_mode))
        self.atd_gaa_num_geo = int(atd_gaa_num_geo)
        self.atd_gaa_max_offset = int(atd_gaa_max_offset)
        self.atd_gaa_block_size = int(atd_gaa_block_size)
        self.atd_gaa_estimator_source = str(atd_gaa_estimator_source)
        valid_gaa_sources = ["current", "shallow", "shallow_detach", "lr_gray"]
        if self.atd_gaa_estimator_source not in valid_gaa_sources:
            raise ValueError("atd_gaa_estimator_source must be one of {}, got {!r}".format(valid_gaa_sources, self.atd_gaa_estimator_source))
        self.atd_use_topk_retrieval = bool(atd_use_topk_retrieval) and not self.atd_use_gaa
        self.atd_topk = int(atd_topk)
        self.atd_topk_max_tokens = int(atd_topk_max_tokens)
        self.atd_use_gdc = bool(atd_use_gdc)
        self.atd_geo_feat_dim = int(atd_geo_feat_dim)
        # 中文注释：SCDRC-Lite 默认关闭；开启后只校准 dictionary residual branch 强度。
        self.use_struct_prior = bool(use_struct_prior)
        self.struct_prior_type = str(struct_prior_type)
        if self.struct_prior_type != "scdrc":
            raise ValueError("struct_prior_type must be 'scdrc', got {!r}".format(self.struct_prior_type))
        self.struct_scale_max = float(struct_scale_max)
        self.struct_delta_init = float(struct_delta_init)
        self.struct_last_weight_std = float(struct_last_weight_std)
        self.struct_use_residual_reliability = bool(struct_use_residual_reliability)
        self.struct_residual_detach = bool(struct_residual_detach)
        self.struct_rr_stats = str(struct_rr_stats)
        self.struct_prior_scope = str(struct_prior_scope)
        if self.struct_prior_scope not in ["global", "local"]:
            raise ValueError("struct_prior_scope must be one of ['global', 'local'], got {!r}".format(self.struct_prior_scope))
        self.struct_local_channels = int(struct_local_channels)
        self.struct_local_scale_max = float(struct_local_scale_max)
        self.struct_local_last_weight_std = float(struct_local_last_weight_std)
        self.struct_local_detach = bool(struct_local_detach)
        self.struct_layer_mode = str(struct_layer_mode)
        if self.struct_layer_mode not in valid_layer_modes:
            raise ValueError("struct_layer_mode must be one of {}, got {!r}".format(valid_layer_modes, self.struct_layer_mode))
        self.struct_hidden_dim = int(struct_hidden_dim)
        # 中文注释：RADR 默认关闭；开启后仅软抑制 dictionary residual，不改 baseline 默认行为。
        self.use_radr = bool(use_radr)
        self.radr_layer_mode = str(radr_layer_mode)
        if self.radr_layer_mode not in valid_layer_modes:
            raise ValueError("radr_layer_mode must be one of {}, got {!r}".format(valid_layer_modes, self.radr_layer_mode))
        self.radr_hidden_dim = int(radr_hidden_dim)
        self.radr_lambda = float(radr_lambda)
        self.radr_tau = float(radr_tau)
        self.radr_init_bias = float(radr_init_bias)
        self.radr_detach_feat = bool(radr_detach_feat)
        self.radr_use_correction = bool(radr_use_correction)
        self.radr_corr_hidden_dim = int(radr_corr_hidden_dim)
        self.radr_corr_lambda = float(radr_corr_lambda)
        self.radr_corr_scale = float(radr_corr_scale)
        self.radr_corr_init_std = float(radr_corr_init_std)
        self.radr_corr_detach_residual = bool(radr_corr_detach_residual)
        self.radr_corr_gate_mode = str(radr_corr_gate_mode)
        if self.radr_corr_gate_mode not in ("ueff", "sqrt", "binary", "none"):
            raise ValueError("Unsupported radr_corr_gate_mode: {}".format(self.radr_corr_gate_mode))
        self.radr_corr_feature_mode = str(radr_corr_feature_mode)
        self.radr_corr_train_feature_modes = str(radr_corr_train_feature_modes or "")
        valid_corr_feature_modes = ("shortcut_xatd", "shortcut_only", "shortcut_xwin", "shortcut_xaca")
        if self.radr_corr_feature_mode not in valid_corr_feature_modes:
            raise ValueError("Unsupported radr_corr_feature_mode: {}".format(self.radr_corr_feature_mode))
        if self.radr_corr_train_feature_modes:
            for mode in [m.strip() for m in self.radr_corr_train_feature_modes.split(",") if m.strip()]:
                if mode not in valid_corr_feature_modes:
                    raise ValueError("Unsupported train radr_corr_feature_mode: {}".format(mode))
        self.use_radr_ccd = bool(use_radr_ccd)
        self.radr_ccd_upscale = int(radr_ccd_upscale)
        self.radr_ccd_aux_init_std = float(radr_ccd_aux_init_std)
        self.radr_ccd_aux_scale = float(radr_ccd_aux_scale)
        self.radr_ccd_aux_use_gated_corr = bool(radr_ccd_aux_use_gated_corr)
        self.use_radr_lch = bool(use_radr_lch)
        self.radr_lch_lambda = float(radr_lch_lambda)
        self.radr_lch_gate_mode = str(radr_lch_gate_mode)
        self.radr_lch_detach_map = bool(radr_lch_detach_map)
        if self.radr_lch_gate_mode not in ("u_eff", "sqrt", "binary", "none"):
            raise ValueError("Unsupported radr_lch_gate_mode: {}".format(self.radr_lch_gate_mode))
        self.use_scdr = bool(use_scdr)
        self.scdr_alpha = float(scdr_alpha)
        self.scdr_return_routes = bool(scdr_return_routes)
        self.scdr_detach_routes = bool(scdr_detach_routes)
        self.use_scdr_adapter = bool(use_scdr_adapter)
        self.scdr_adapter_hidden_dim = int(scdr_adapter_hidden_dim)
        self.scdr_adapter_scale = float(scdr_adapter_scale)
        self.scdr_adapter_init_std = float(scdr_adapter_init_std)
        if not (0.0 <= self.scdr_alpha <= 1.0):
            raise ValueError("scdr_alpha must be in [0,1], got {}".format(self.scdr_alpha))
        if self.use_scdr_adapter and not self.use_scdr:
            raise ValueError("use_scdr_adapter=True requires use_scdr=True")
        if self.use_scdr:
            if not self.use_radr:
                raise ValueError("use_scdr=True requires use_radr=True")
            if not self.radr_use_correction:
                raise ValueError("use_scdr=True requires radr_use_correction=True")
            if self.use_radr_ccd:
                raise ValueError("use_scdr=True does not support use_radr_ccd=True in v1")
            if self.use_radr_lch:
                raise ValueError("use_scdr=True does not support use_radr_lch=True in v1")
        self.scdr_adapter_a = SCDRRouteAdapter(
            dim=embed_dim,
            hidden_dim=self.scdr_adapter_hidden_dim,
            scale=self.scdr_adapter_scale,
            init_std=self.scdr_adapter_init_std,
        ) if self.use_scdr_adapter else None
        self.scdr_adapter_b = SCDRRouteAdapter(
            dim=embed_dim,
            hidden_dim=self.scdr_adapter_hidden_dim,
            scale=self.scdr_adapter_scale,
            init_std=self.scdr_adapter_init_std,
        ) if self.use_scdr_adapter else None
        self.radr_lch = RADRLateCorrectionHead(
            dim=embed_dim,
            hidden_dim=radr_lch_hidden_dim,
            init_std=radr_lch_init_std,
            corr_scale=radr_lch_corr_scale,
            use_map=True,
        ) if (self.use_radr and self.use_radr_lch) else None
        # 中文注释：DECB 默认关闭，开启后只扣除 TDCA/dictionary residual 中的 error residual。
        self.use_decb = bool(use_decb)
        # 中文注释：DECB error dictionary token 数量。
        self.decb_num_tokens = int(decb_num_tokens)
        # 中文注释：DECB gate 最大幅度。
        self.decb_gate_max = float(decb_gate_max)
        # 中文注释：DECB gate 初始 bias，负值使初始补偿接近 0。
        self.decb_gate_init = float(decb_gate_init)
        # 中文注释：DECB error token 初始化标准差。
        self.decb_token_init_std = float(decb_token_init_std)
        # 中文注释：DECB 输出投影初始化标准差。
        self.decb_proj_init_std = float(decb_proj_init_std)
        # 中文注释：DECB 层级控制，第一版默认 last。
        self.decb_layer_mode = str(decb_layer_mode)
        if self.decb_layer_mode not in valid_layer_modes:
            raise ValueError("decb_layer_mode must be one of {}, got {!r}".format(valid_layer_modes, self.decb_layer_mode))
        # 中文注释：DECB gate 条件源；shortcut 保持旧行为，concat 为 RC-DECB 主实验。
        self.decb_gate_condition = str(decb_gate_condition)
        # 中文注释：默认 detach x_atd 后再喂给 gate 条件分支。
        self.decb_residual_detach = bool(decb_residual_detach)
        # 中文注释：zero_dict_residual 仅用于分析 dictionary residual 帮助度。
        # 中文注释：开启时将 TDCA / ATD dictionary residual 输出置零，用于构造 SR_nodict。
        self.zero_dict_residual = False
        # 中文注释：固定结构提取器无可学习参数；关闭时不注册，baseline state_dict 不变。
        self.struct_extractor = FixedStructureExtractor() if self.use_struct_prior else None
        # 中文注释：普通 SCDRC 输入 5 维；SCDRC-RR 额外拼接 3 个 residual reliability 统计。
        struct_input_dim = 8 if self.struct_use_residual_reliability else 5
        # 中文注释：轻量 MLP 控制器输出图像级 scalar delta；关闭时不注册。
        self.struct_controller = StructureResidualController(
            hidden_dim=self.struct_hidden_dim,
            scale_max=self.struct_scale_max,
            delta_init=self.struct_delta_init,
            last_weight_std=self.struct_last_weight_std,
            input_dim=struct_input_dim,
        ) if self.use_struct_prior else None
        # 中文注释：local scope 额外注册固定局部结构图提取器；global scope 不注册，保持原行为。
        self.local_struct_extractor = FixedLocalStructureExtractor() if (self.use_struct_prior and self.struct_prior_scope == "local") else None
        # 中文注释：local scope 额外注册局部 delta map controller；第一版不叠加 global scalar delta。
        self.local_struct_controller = LocalStructureResidualController(
            in_channels=4,
            hidden_channels=self.struct_local_channels,
            scale_max=self.struct_local_scale_max,
            last_weight_std=self.struct_local_last_weight_std,
        ) if (self.use_struct_prior and self.struct_prior_scope == "local") else None
        # 中文注释：保存每次 forward 的 per-image 结构统计和 delta，供训练日志与验证 CSV 使用。
        self.struct_debug_cache = {}
        # 中文注释：保存最近一次 forward 的 RADR per-image 统计，供日志与验证 CSV 使用。
        self.radr_debug_cache = {}
        # 中文注释：保存最近一次启用 RADR 层的非 detach 预测缓存，供 supervised reliability loss 使用。
        self.radr_train_cache = {}
        # 中文注释：保存最近一次 CCD auxiliary residual 预测，仅训练 capture_radr_ccd=True 时写入。
        self.radr_ccd_cache = {}
        # 中文注释：保存模型级 LCH late correction 诊断值，默认关闭时保持为空。
        self.radr_lch_debug_cache = {}
        # 中文注释：保存 SCDR 双路融合的诊断值；默认关闭时为空，不进 state_dict。
        self.scdr_debug_cache = {}
        # 中文注释：保存每次 forward 的 DECB per-image 诊断值，供训练日志与验证 CSV 使用。
        self.decb_debug_cache = {}

        # 中文注释：FDG 只影响 AC-MSA grouping/category，不进入 ATD-CA read-out 或字典更新。
        self.use_fdg = bool(use_fdg)
        self.fdg_df = int(fdg_df)
        self.fdg_lambda0 = float(fdg_lambda0)
        self.fdg_init_alpha = float(fdg_init_alpha)
        self.fdg_in_channels = int(fdg_in_channels)
        self.fdg_layer_mode = str(fdg_layer_mode)
        # 中文注释：方案 B 新增——AC-MSA route bias / x_td STE 总控开关。
        self.fdg_use_acmsa_route_bias = bool(fdg_use_acmsa_route_bias)
        self.fdg_route_bias_max = float(fdg_route_bias_max)
        self.fdg_route_bias_init = float(fdg_route_bias_init)
        self.fdg_route_bias_detach = bool(fdg_route_bias_detach)
        self.fdg_use_td_ste = bool(fdg_use_td_ste)
        self.fdg_use_gumbel_grouping = bool(fdg_use_gumbel_grouping)
        self.fdg_tau_gumbel = float(fdg_tau_gumbel)
        self.fdg_use_routeprob_bias = bool(fdg_use_routeprob_bias)
        if self.fdg_layer_mode not in valid_layer_modes:
            raise ValueError("fdg_layer_mode must be one of {}, got {!r}".format(valid_layer_modes, self.fdg_layer_mode))
        self.freq_map_builder = FrequencyMapBuilder(out_channels=self.fdg_in_channels, detach=True) if self.use_fdg else None
        self.fdg_debug_cache = {}

        # 中文注释：旧拼接式/调制式频率路径在方案 B 中禁用。
        self.atd_use_fcdm = False
        self.atd_fcdm_desc_dim = int(atd_fcdm_desc_dim)
        self.atd_fcdm_hidden_dim = int(atd_fcdm_hidden_dim)
        self.atd_fcdm_scale_init = float(atd_fcdm_scale_init)
        self.atd_fcdm_layer_mode = "none"
        self.atd_use_fats = False
        self.atd_fats_hidden_dim = int(atd_fats_hidden_dim)
        self.atd_fats_tau_range = float(atd_fats_tau_range)
        self.atd_fats_layer_mode = "none"
        self.atd_use_lfcdm = False
        self.atd_lfcdm_desc_dim = int(atd_lfcdm_desc_dim)
        self.atd_lfcdm_hidden_dim = int(atd_lfcdm_hidden_dim)
        self.atd_lfcdm_window_size = int(atd_lfcdm_window_size)
        self.atd_lfcdm_gate_bias = float(atd_lfcdm_gate_bias)
        self.atd_lfcdm_detach = bool(atd_lfcdm_detach)
        self.freq_descriptor = None
        self.local_freq_descriptor = None

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        # build Residual Adaptive Token Dictionary Blocks (ATDB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ATDB(
                dim=embed_dim,
                idx=i_layer,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                dim_ffn_td=dim_ffn_td,
                category_size=category_size,
                num_tokens=num_tokens,
                reducted_dim=reducted_dim,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                use_dawa=self.atd_use_dawa,
                dawa_long=self.atd_dawa_long,
                dawa_short=self.atd_dawa_short,
                dawa_conf_threshold=self.atd_dawa_conf_threshold,
                dawa_layer_mode=self.atd_dawa_layer_mode,
                use_gaa=self.atd_use_gaa,
                gaa_layer_mode=self.atd_gaa_layer_mode,
                gaa_num_geo=self.atd_gaa_num_geo,
                gaa_max_offset=self.atd_gaa_max_offset,
                gaa_block_size=self.atd_gaa_block_size,
                gaa_estimator_source=self.atd_gaa_estimator_source,
                use_topk_retrieval=self.atd_use_topk_retrieval,
                topk=self.atd_topk,
                topk_max_tokens=self.atd_topk_max_tokens,
                use_gdc=self.atd_use_gdc,
                geo_feat_dim=self.atd_geo_feat_dim,
                use_fdg=self.use_fdg,
                fdg_in_channels=self.fdg_in_channels,
                fdg_df=self.fdg_df,
                fdg_lambda0=self.fdg_lambda0,
                fdg_init_alpha=self.fdg_init_alpha,
                fdg_layer_mode=self.fdg_layer_mode,
                fdg_use_acmsa_route_bias=self.fdg_use_acmsa_route_bias,
                fdg_route_bias_max=self.fdg_route_bias_max,
                fdg_route_bias_init=self.fdg_route_bias_init,
                fdg_route_bias_detach=self.fdg_route_bias_detach,
                fdg_use_td_ste=self.fdg_use_td_ste,
                fdg_use_gumbel_grouping=self.fdg_use_gumbel_grouping,
                fdg_tau_gumbel=self.fdg_tau_gumbel,
                fdg_use_routeprob_bias=self.fdg_use_routeprob_bias,
                use_fcdm=False,
                fcdm_desc_dim=self.atd_fcdm_desc_dim,
                fcdm_hidden_dim=self.atd_fcdm_hidden_dim,
                fcdm_scale_init=self.atd_fcdm_scale_init,
                fcdm_layer_mode=self.atd_fcdm_layer_mode,
                use_fats=False,
                fats_hidden_dim=self.atd_fats_hidden_dim,
                fats_tau_range=self.atd_fats_tau_range,
                fats_layer_mode=self.atd_fats_layer_mode,
                use_lfcdm=False,
                lfcdm_desc_dim=self.atd_lfcdm_desc_dim,
                lfcdm_hidden_dim=self.atd_lfcdm_hidden_dim,
                lfcdm_gate_bias=self.atd_lfcdm_gate_bias,
                use_struct_prior=self.use_struct_prior,
                struct_layer_mode=self.struct_layer_mode,
                use_radr=self.use_radr,
                radr_layer_mode=self.radr_layer_mode,
                radr_hidden_dim=self.radr_hidden_dim,
                radr_lambda=self.radr_lambda,
                radr_tau=self.radr_tau,
                radr_init_bias=self.radr_init_bias,
                radr_detach_feat=self.radr_detach_feat,
                radr_use_correction=self.radr_use_correction,
                radr_corr_hidden_dim=self.radr_corr_hidden_dim,
                radr_corr_lambda=self.radr_corr_lambda,
                radr_corr_scale=self.radr_corr_scale,
                radr_corr_init_std=self.radr_corr_init_std,
                radr_corr_detach_residual=self.radr_corr_detach_residual,
                radr_corr_gate_mode=self.radr_corr_gate_mode,
                radr_corr_feature_mode=self.radr_corr_feature_mode,
                radr_corr_train_feature_modes=self.radr_corr_train_feature_modes,
                use_radr_ccd=self.use_radr_ccd,
                radr_ccd_upscale=self.radr_ccd_upscale,
                radr_ccd_aux_init_std=self.radr_ccd_aux_init_std,
                radr_ccd_aux_scale=self.radr_ccd_aux_scale,
                radr_ccd_aux_use_gated_corr=self.radr_ccd_aux_use_gated_corr,
                use_decb=self.use_decb,
                decb_num_tokens=self.decb_num_tokens,
                decb_gate_max=self.decb_gate_max,
                decb_gate_init=self.decb_gate_init,
                decb_token_init_std=self.decb_token_init_std,
                decb_proj_init_std=self.decb_proj_init_std,
                decb_layer_mode=self.decb_layer_mode,
                decb_gate_condition=self.decb_gate_condition,
                decb_residual_detach=self.decb_residual_detach,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)
        if self.struct_controller is not None:
            # 中文注释：全局 _init_weights 会初始化 Linear，这里重新设置最后一层，保证指定 init delta 生效。
            self.struct_controller.reset_last_layer_init()
        if self.local_struct_controller is not None:
            # 中文注释：显式重置 local controller，防止后续初始化策略变化时覆盖小随机最后一层。
            self.local_struct_controller.reset_last_layer_init()
        for module in self.modules():
            if isinstance(module, DictionaryErrorCompensationBranch):
                # 中文注释：全局 _init_weights 会初始化 Linear/LayerNorm，这里恢复 DECB 自定义 warm-start 初始化。
                module.reset_parameters()
            if isinstance(module, LocalReliabilityPredictor):
                # 中文注释：恢复 RADR predictor 最后一层 zero-weight + negative-bias 的 warm-start 初始化。
                module.reset_parameters()

    def _init_weights(self, m):
        # 中文注释：旧频率调制特殊初始化已移除，恢复 ATD baseline 初始化。
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_fdg_grad_stats(self):
        # 中文注释：轻量读取 FDG 梯度诊断值，不影响 forward 和参数更新。
        alpha_grads = []
        phi_grads = []
        psi_grads = []
        route_bias_alpha_grads = []
        for module in self.modules():
            if isinstance(module, FreqDecoupledGrouping):
                if module.alpha.grad is not None:
                    alpha_grads.append(module.alpha.grad.detach().abs().mean())
                if module.phi.grad is not None:
                    phi_grads.append(module.phi.grad.detach().abs().mean())
                for param in module.psi.parameters():
                    if param.grad is not None:
                        psi_grads.append(param.grad.detach().abs().mean())
            if isinstance(module, AC_MSA) and getattr(module, "route_bias_alpha", None) is not None:
                # 中文注释：AC-MSA route bias 的梯度——FDG 是否通过 route bias 拿到主任务梯度的直接证据。
                if module.route_bias_alpha.grad is not None:
                    route_bias_alpha_grads.append(module.route_bias_alpha.grad.detach().abs().mean())

        def _mean_or_zero(values):
            # 中文注释：未启用 FDG 或尚无梯度时返回 0，避免训练日志报错。
            if len(values) == 0:
                return torch.tensor(0.0)
            return torch.stack(values).mean().detach().cpu()

        return {
            "fdg/grad_alpha": _mean_or_zero(alpha_grads),
            "fdg/grad_phi": _mean_or_zero(phi_grads),
            "fdg/grad_psi": _mean_or_zero(psi_grads),
            "fdg/grad_route_bias_alpha": _mean_or_zero(route_bias_alpha_grads),
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        if self.atd_use_gaa and self.atd_gaa_estimator_source in ["shallow", "shallow_detach"]:
            params["gaa_shallow_feat"] = x
            params["gaa_shallow_size"] = x_size

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x
    
    def calculate_rpi_sa(self):
        # calculate relative position index for SW-MSA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index
    
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def _get_radr_u_eff_map_for_lch(self, target_hw, device, dtype):
        cache = getattr(self, "radr_train_cache", {})
        u_eff = cache.get("u_eff", None)
        if u_eff is None:
            return None

        if u_eff.dim() == 3:
            spatial_shape = cache.get("spatial_shape", None)
            if spatial_shape is None:
                return None
            h, w = int(spatial_shape[0]), int(spatial_shape[1])
            if u_eff.shape[1] != h * w or u_eff.shape[-1] != 1:
                return None
            u_map = u_eff.transpose(1, 2).reshape(u_eff.shape[0], 1, h, w)
        elif u_eff.dim() == 4:
            u_map = u_eff
        else:
            return None

        if self.radr_lch_detach_map:
            u_map = u_map.detach()

        target_hw = (int(target_hw[0]), int(target_hw[1]))
        if u_map.shape[-2:] != target_hw:
            u_map = F.interpolate(u_map.float(), size=target_hw, mode="bilinear", align_corners=False)

        return u_map.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    def _compute_radr_lch_gate(self, u_map):
        mode = str(getattr(self, "radr_lch_gate_mode", "u_eff"))
        if mode == "u_eff":
            return u_map
        if mode == "sqrt":
            return torch.sqrt(u_map.clamp_min(0.0) + 1e-6)
        if mode == "binary":
            return (u_map > 0.0).to(dtype=u_map.dtype, device=u_map.device)
        if mode == "none":
            return torch.ones_like(u_map)
        raise ValueError("Unsupported radr_lch_gate_mode: {}".format(mode))

    def _apply_radr_lch(self, feat):
        if not (self.use_radr_lch and self.radr_lch is not None):
            self.radr_lch_debug_cache = {}
            return feat
        u_map = self._get_radr_u_eff_map_for_lch(
            target_hw=feat.shape[-2:],
            device=feat.device,
            dtype=feat.dtype,
        )
        if u_map is None:
            self.radr_lch_debug_cache = {}
            return feat
        lch_gate = self._compute_radr_lch_gate(u_map)
        late_corr = self.radr_lch(feat, u_map)
        gated_corr = float(self.radr_lch_lambda) * lch_gate.to(dtype=feat.dtype, device=feat.device) * late_corr
        feat = feat + gated_corr
        with torch.no_grad():
            self.radr_lch_debug_cache = {
                "u_mean": u_map.detach().float().mean(dim=(1, 2, 3), keepdim=True),
                "gate_mean": lch_gate.detach().float().mean(dim=(1, 2, 3), keepdim=True),
                "late_corr_abs_mean": late_corr.detach().float().abs().mean(dim=(1, 2, 3), keepdim=True),
                "gated_corr_abs_mean": gated_corr.detach().float().abs().mean(dim=(1, 2, 3), keepdim=True),
                "lambda": torch.full(
                    (feat.shape[0], 1),
                    float(self.radr_lch_lambda),
                    device=feat.device,
                    dtype=torch.float32,
                ),
            }
        return feat

    def _apply_scdr_route_adapter(self, feat, scdr_route=None):
        if not (self.use_scdr and self.use_scdr_adapter):
            return feat
        if scdr_route == "a" and self.scdr_adapter_a is not None:
            return feat + self.scdr_adapter_a(feat)
        if scdr_route == "b" and self.scdr_adapter_b is not None:
            return feat + self.scdr_adapter_b(feat)
        return feat

    def _forward_impl(
        self,
        x,
        return_aux=False,
        fdg_warmup_factor=1.0,
        radr_disable_suppression=False,
        radr_disable_correction=False,
        capture_radr_ccd=False,
        scdr_route=None,
    ):
        # 中文注释：SCDRC 使用原始 LR 图像 [0,1] 计算跨域稳定结构统计，不使用 scene 标签。
        # 中文注释：struct_vec 保存 5 个原始结构指标。
        struct_vec = None
        # 中文注释：local_struct_map 保存 B,4,H,W 局部结构图，仅 local scope 使用。
        local_struct_map = None
        if self.use_struct_prior and self.struct_extractor is not None and self.struct_controller is not None:
            # 中文注释：固定结构提取器不含可学习参数；输入为未 padding、未 mean-shift 的 LR。
            lr_for_struct = x.clamp(0.0, 1.0)
            # 中文注释：global 统计始终保留，用于日志、CSV 和 full-validation 分析。
            struct_vec = self.struct_extractor(lr_for_struct)
            # 中文注释：local scope 额外计算局部结构图，第一版不使用 scene 或 residual reliability。
            if self.struct_prior_scope == "local" and self.local_struct_extractor is not None:
                local_struct_map = self.local_struct_extractor(lr_for_struct)
                # 中文注释：默认 detach local structure map，避免通过固定结构图路径改动输入梯度形态。
                if self.struct_local_detach:
                    local_struct_map = local_struct_map.detach()
            # 中文注释：delta 在启用的 TDCA block 内生成；forward 开头只清空 cache。
            self.struct_debug_cache = {}
        else:
            # 中文注释：关闭 SCDRC 时清空 cache，避免验证 CSV 误读旧值。
            self.struct_debug_cache = {}
        # 中文注释：每次 forward 开头清空 RADR cache，避免训练 loss 或验证 CSV 误读旧值。
        self.radr_debug_cache = {}
        self.radr_train_cache = {}
        self.radr_ccd_cache = {}
        self.radr_lch_debug_cache = {}
        # 中文注释：每次 forward 开头清空 DECB cache，避免验证 CSV 误读旧值。
        self.decb_debug_cache = {}

        # padding
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        # DAWA H window_size dawa_long lcm
        if self.atd_use_dawa and self.atd_dawa_layer_mode != "none":
            mod = int(np.lcm(int(self.window_size), int(self.atd_dawa_long))) #
        else:
            mod = self.window_size # ATD
        # GAA padding mod pad
        if self.atd_use_gaa:
            mod = int(np.lcm(mod, int(self.atd_gaa_block_size))) # GAA
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {"attn_mask": attn_mask, "rpi_sa": self.relative_position_index_SA}
        # 中文注释：zero_dict_residual 仅用于分析 dictionary residual 帮助度，默认关闭。
        params["zero_dict_residual"] = bool(getattr(self, "zero_dict_residual", False))
        if self.use_radr:
            params["radr_debug_cache"] = {}
            params["radr_train_cache"] = {}
            params["radr_disable_suppression"] = bool(radr_disable_suppression)
            params["radr_disable_correction"] = bool(radr_disable_correction)
            params["capture_radr_ccd"] = bool(capture_radr_ccd)
            params["radr_ccd_delta_list"] = []
        if self.use_decb:
            # 中文注释：启用 DECB 的 block 会覆盖写入最后一个 block 的诊断统计。
            params["decb_debug_cache"] = {}
        if struct_vec is not None:
            # 中文注释：把结构向量和共享 controller 传入各层；只有 enable_struct_prior 的层会使用。
            params["struct_vec"] = struct_vec
            # 中文注释：记录 scope，global 使用标量 controller，local 使用局部 delta map controller。
            params["struct_prior_scope"] = self.struct_prior_scope
            # 中文注释：controller 仍是共享图像级 MLP，不做 per-layer delta 模块。
            params["struct_controller"] = self.struct_controller
            # 中文注释：local scope 的局部结构图，只在 local 分支使用。
            params["local_struct_map"] = local_struct_map
            # 中文注释：local scope 的共享 conv controller，只在 local 分支使用。
            params["local_struct_controller"] = self.local_struct_controller
            # 中文注释：SCDRC-RR 开关控制是否拼接 residual reliability 统计。
            params["struct_use_residual_reliability"] = self.struct_use_residual_reliability
            # 中文注释：默认 detach residual stats，避免通过统计路径改动 TDCA residual。
            params["struct_residual_detach"] = self.struct_residual_detach
            # 中文注释：保留 RR stats 字符串用于日志/追踪，当前第一版使用 abs_mean,abs_std,ratio 三项。
            params["struct_rr_stats"] = self.struct_rr_stats
            # 中文注释：启用层会覆盖写入最后一个 block 的 delta/residual 统计。
            params["struct_debug_cache"] = {}
        if return_aux:
            # 中文注释：FDPP 只捕获原始 S_content softmax 后的字典 attention。
            params["capture_dict_attn"] = True
            params["dict_attn_list"] = []
        if self.use_fdg and self.freq_map_builder is not None:
            # 中文注释：FDG 使用 pad 后、归一化后的 LR 构造频率图，只供 grouping 使用。
            params["freq_map"] = self.freq_map_builder(x)
            params["fdg_warmup_factor"] = float(fdg_warmup_factor)
            params["fdg_tau_gumbel"] = self.fdg_tau_gumbel
            params["fdg_use_gumbel_grouping"] = self.fdg_use_gumbel_grouping
            params["fdg_use_routeprob_bias"] = self.fdg_use_routeprob_bias
            params["fdg_stats"] = []
        if self.atd_use_gaa and self.atd_gaa_estimator_source == "lr_gray":
            # 中文注释：GAA lr_gray 输入保持原逻辑，且不参与输入图像反向梯度。
            with torch.no_grad():
                if x.shape[1] == 3:
                    lr_gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
                else:
                    lr_gray = x.mean(dim=1, keepdim=True)
            params["gaa_lr_gray"] = lr_gray.detach()
            params["gaa_lr_gray_size"] = (h, w)

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            self.radr_train_cache = params.get("radr_train_cache", {})
            x = self._apply_scdr_route_adapter(x, scdr_route=scdr_route)
            x = self._apply_radr_lch(x)
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            self.radr_train_cache = params.get("radr_train_cache", {})
            x = self._apply_scdr_route_adapter(x, scdr_route=scdr_route)
            x = self._apply_radr_lch(x)
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            self.radr_train_cache = params.get("radr_train_cache", {})
            x = self._apply_scdr_route_adapter(x, scdr_route=scdr_route)
            x = self._apply_radr_lch(x)
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            self.radr_train_cache = params.get("radr_train_cache", {})
            res = self._apply_scdr_route_adapter(res, scdr_route=scdr_route)
            res = self._apply_radr_lch(res)
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        # unpadding
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]

        if self.use_struct_prior:
            # 中文注释：把启用 block 写入 params 的最后一次 SCDRC/SCDRC-RR cache 挂到模型上，供日志和 CSV 使用。
            self.struct_debug_cache = params.get("struct_debug_cache", {})
        if self.use_radr:
            # 中文注释：把最后一个启用 RADR block 的预测统计和 train cache 挂到模型上。
            self.radr_debug_cache = params.get("radr_debug_cache", {})
            self.radr_train_cache = params.get("radr_train_cache", {})
            delta_list = params.get("radr_ccd_delta_list", [])
            if delta_list:
                delta_pred = delta_list[-1]
                self.radr_ccd_cache = {
                    "delta_pred": delta_pred,
                    "num_delta": torch.tensor(
                        float(len(delta_list)),
                        device=delta_pred.device,
                        dtype=delta_pred.dtype,
                    ),
                    "delta_pred_abs_mean": delta_pred.detach().float().abs().mean(
                        dim=(1, 2, 3), keepdim=True
                    ),
                }
            else:
                self.radr_ccd_cache = {}
        if self.use_decb:
            # 中文注释：把启用 block 写入 params 的最后一次 DECB cache 挂到模型上，供日志和 CSV 使用。
            self.decb_debug_cache = params.get("decb_debug_cache", {})

        if self.use_fdg:
            # 中文注释：聚合本次 forward 的 FDG 诊断指标，供 train.py 写入日志。
            stats = params.get("fdg_stats", [])
            if stats:
                # 中文注释：必有的核心指标——直接对所有启用层做均值。
                # 中文注释：fdg_gate 是新增的共享门均值，方便诊断 lam/route_beta 同步。
                core_keys = [
                    "lambda",
                    "alpha",
                    "fdg_gate",
                    "S_freq_abs_mean",
                    "category_change_ratio",
                    "category_change_det",
                    "category_change_forward",
                    "category_change_sample_vs_det",
                    "category_change_gumbel",
                    "category_change_gumbel_vs_det",
                    "category_entropy",
                    "fdg_tau_gumbel",
                    "fdg_use_gumbel_grouping",
                ]
                self.fdg_debug_cache = {
                    key: torch.stack([item[key].reshape(()) for item in stats]).mean().detach().cpu()
                    for key in core_keys
                    if all(key in item for item in stats)
                }
                self.fdg_debug_cache["fdg_assign_mode"] = stats[0].get("fdg_assign_mode", "")
                # 中文注释：可选指标——只有真正启用 AC-MSA route bias 的层才会写入。
                optional_keys = [
                    "acmsa_route_beta",
                    "acmsa_route_sim_mean",
                    "acmsa_route_sim_std",
                    "same_grp_mean",
                    "same_grp_std",
                ]
                for key in optional_keys:
                    vals = [item[key].reshape(()) for item in stats if key in item]
                    if vals:
                        self.fdg_debug_cache[key] = torch.stack(vals).mean().detach().cpu()
                # 中文注释：所有启用层的 td_ste 开关状态是否一致；按 OR 汇总作为单值。
                td_ste_flags = [bool(item.get("td_ste_enabled", False)) for item in stats]
                self.fdg_debug_cache["td_ste_enabled"] = bool(any(td_ste_flags))
                self.fdg_debug_cache["shapes"] = {
                    "S_content": stats[0]["S_content_shape"],
                    "freq_map": stats[0]["freq_map_shape"],
                    "S_freq": stats[0]["S_freq_shape"],
                    "S_group": stats[0]["S_group_shape"],
                    "fdg_assign": stats[0].get("fdg_assign_shape", None),
                    "ste_assign": stats[0].get("ste_assign_shape", None),
                    "x_td": stats[0].get("x_td_shape", None),
                }
            else:
                self.fdg_debug_cache = {}

        if return_aux:
            aux = {"dict_attn": params.get("dict_attn_list", [])}
            if self.use_struct_prior and self.struct_debug_cache:
                # 中文注释：不破坏原 aux 逻辑，只追加 SCDRC 调试字段。
                aux["struct_delta"] = self.struct_debug_cache["delta"].detach()
                aux["structure_score"] = self.struct_debug_cache["structure_score"].detach()
                aux["struct_vec"] = self.struct_debug_cache["struct_vec"].detach()
                # 中文注释：local scope 不经过 global controller，struct_norm 允许为 None。
                if self.struct_debug_cache.get("struct_norm", None) is not None:
                    aux["struct_norm"] = self.struct_debug_cache["struct_norm"].detach()
                # 中文注释：SCDRC-RR 开启时额外返回 residual reliability stats；普通 SCDRC 时为 None。
                if self.struct_debug_cache.get("residual_vec", None) is not None:
                    aux["residual_vec"] = self.struct_debug_cache["residual_vec"].detach()
                # 中文注释：local scope 额外返回空间 delta 统计，供调试使用。
                for key in ["delta_std_spatial", "delta_min_spatial", "delta_max_spatial"]:
                    if self.struct_debug_cache.get(key, None) is not None:
                        aux[key] = self.struct_debug_cache[key].detach()
            if self.use_decb and self.decb_debug_cache:
                # 中文注释：不破坏原 aux 逻辑，只追加 DECB 调试字段。
                for key, value in self.decb_debug_cache.items():
                    if value is not None:
                        aux["decb_{}".format(key)] = value.detach()
            return x, aux
        return x

    def forward(
        self,
        x,
        return_aux=False,
        fdg_warmup_factor=1.0,
        radr_disable_suppression=False,
        radr_disable_correction=False,
        capture_radr_ccd=False,
        return_scdr=False,
    ):
        if not self.use_scdr:
            self.scdr_debug_cache = {}
            return self._forward_impl(
                x,
                return_aux=return_aux,
                fdg_warmup_factor=fdg_warmup_factor,
                radr_disable_suppression=radr_disable_suppression,
                radr_disable_correction=radr_disable_correction,
                capture_radr_ccd=capture_radr_ccd,
            )

        sr_a = self._forward_impl(
            x,
            return_aux=False,
            fdg_warmup_factor=fdg_warmup_factor,
            radr_disable_suppression=radr_disable_suppression,
            radr_disable_correction=True,
            capture_radr_ccd=False,
            scdr_route="a",
        )
        sr_b = self._forward_impl(
            x,
            return_aux=False,
            fdg_warmup_factor=fdg_warmup_factor,
            radr_disable_suppression=radr_disable_suppression,
            radr_disable_correction=False,
            capture_radr_ccd=False,
            scdr_route="b",
        )
        if isinstance(sr_a, (tuple, list)):
            sr_a = sr_a[0]
        if isinstance(sr_b, (tuple, list)):
            sr_b = sr_b[0]

        if self.scdr_detach_routes:
            sr_a_for_fuse = sr_a.detach()
            sr_b_for_fuse = sr_b.detach()
        else:
            sr_a_for_fuse = sr_a
            sr_b_for_fuse = sr_b

        alpha = float(self.scdr_alpha)
        sr = alpha * sr_a_for_fuse + (1.0 - alpha) * sr_b_for_fuse
        with torch.no_grad():
            self.scdr_debug_cache = {
                "alpha": torch.full((x.shape[0], 1), alpha, device=x.device, dtype=torch.float32),
                "route_a_abs_mean": sr_a.detach().float().abs().mean(dim=(1, 2, 3), keepdim=True),
                "route_b_abs_mean": sr_b.detach().float().abs().mean(dim=(1, 2, 3), keepdim=True),
                "route_diff_abs_mean": (sr_a.detach().float() - sr_b.detach().float()).abs().mean(
                    dim=(1, 2, 3), keepdim=True
                ),
            }
        if return_scdr or self.scdr_return_routes:
            return {
                "sr": sr,
                "sr_a": sr_a,
                "sr_b": sr_b,
                "alpha": torch.full((x.shape[0], 1, 1, 1), alpha, device=x.device, dtype=sr.dtype),
            }
        return sr

    def flops(self, input_resolution=None):
        flops = 0
        resolution = self.patches_resolution if input_resolution is None else input_resolution
        h, w = resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops(resolution)
        for layer in self.layers:
            flops += layer.flops(resolution)
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        if self.upsampler == 'pixelshuffle':
            flops += self.upsample.flops(resolution)
        else:
            flops += self.upsample.flops(resolution)

        return flops


SCDR_RADR = ATD
