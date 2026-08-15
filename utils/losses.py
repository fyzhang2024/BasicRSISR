"""SR 任务训练期损失：L1 与高频幅度谱一致性 DFC Loss。

说明：
1. DFC Loss 只在训练阶段使用，验证 / 推理不需要修改。
2. 第一版只实现高频环形 mask，后续可以继续加入 θ 方向加权。
3. 为了稳定训练，FFT 使用 norm='ortho'，并且只在高频 mask 区域求平均。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalFrequencyLoss(nn.Module):
    """高频幅度谱一致性损失。

    当前版本本质是 High-Frequency Amplitude Consistency Loss。
    后续如果加入 θ 方向 mask，再严格称为 Directional Frequency Consistency Loss。
    """

    def __init__(self, high_freq_ratio=0.25, eps=1e-6, use_fp32=True, use_log_amp=True):
        super().__init__()  # 中文注释：调用父类构造函数
        self.high_freq_ratio = float(high_freq_ratio)  # 中文注释：高频环形 mask 的内半径
        self.eps = float(eps)  # 中文注释：数值稳定项
        self.use_fp32 = bool(use_fp32)  # 中文注释：AMP 下临时转 float32，避免 FFT 不稳定
        self.use_log_amp = bool(use_log_amp)  # 中文注释：是否使用 log 幅度，减小大幅值频谱的主导作用

    def _build_mask(self, h, w, device, dtype):
        # 中文注释：rfft2 输出宽度为 w//2+1
        kw = w // 2 + 1

        # 中文注释：构造 y 方向频率索引
        fy = torch.arange(h, device=device, dtype=dtype)

        # 中文注释：将 y 方向频率折叠到负频率形式
        fy_signed = torch.where(fy <= h // 2, fy, fy - h)

        # 中文注释：归一化 y 方向频率到约 [-1, 1]
        fy_norm = fy_signed / max(h // 2, 1)

        # 中文注释：x 方向 rfft 只保留非负频率，范围约 [0, 1]
        fx_norm = torch.arange(kw, device=device, dtype=dtype) / max(w // 2, 1)

        # 中文注释：构造二维频率网格
        fyg, fxg = torch.meshgrid(fy_norm, fx_norm, indexing="ij")

        # 中文注释：计算归一化频率半径
        radius = torch.sqrt(fyg * fyg + fxg * fxg + self.eps)

        # 中文注释：半径大于阈值的位置视为高频
        mask = (radius > self.high_freq_ratio).to(dtype)

        # 中文注释：返回形状为 (1, 1, H, W//2+1)，便于和 B×C×H×Wf 广播
        return mask.view(1, 1, h, kw)

    def forward(self, sr, hr):
        # 中文注释：要求 SR 和 HR 形状一致
        assert sr.shape == hr.shape, "sr 与 hr 形状必须一致"

        # 中文注释：记录原始 dtype
        orig_dtype = sr.dtype

        # 中文注释：FFT 建议用 float32，尤其是 AMP 训练时
        if self.use_fp32:
            sr_f = sr.float()
            hr_f = hr.float()
        else:
            sr_f = sr
            hr_f = hr

        # 中文注释：取空间尺寸
        h, w = sr_f.shape[-2], sr_f.shape[-1]

        # 中文注释：使用正交归一化 FFT，避免频谱幅度随图像尺寸线性放大
        spec_sr = torch.fft.rfft2(sr_f, dim=(-2, -1), norm="ortho")
        spec_hr = torch.fft.rfft2(hr_f, dim=(-2, -1), norm="ortho")

        # 中文注释：计算幅度谱
        amp_sr = torch.abs(spec_sr)
        amp_hr = torch.abs(spec_hr)

        # 中文注释：可选 log 幅度，减小极大频谱值对损失的支配
        if self.use_log_amp:
            amp_sr = torch.log1p(amp_sr)
            amp_hr = torch.log1p(amp_hr)

        # 中文注释：构造高频 mask
        mask = self._build_mask(h, w, sr_f.device, amp_sr.dtype)

        # 中文注释：只在高频 mask 区域计算 L1，不把低频 0 区域计入平均
        diff = torch.abs(amp_sr - amp_hr) * mask

        # 中文注释：分母为有效高频元素数量，并乘以 batch 和 channel 数
        denom = mask.sum() * sr_f.shape[0] * sr_f.shape[1] + self.eps

        # 中文注释：得到高频幅度谱平均误差
        loss = diff.sum() / denom

        # 中文注释：返回标量损失；保持 fp32 更稳定，不需要强行转回 fp16
        return loss


class SRCompositeLoss(nn.Module):
    """L1 + DFC 的组合损失。

    可直接替换 nn.L1Loss()，不需要修改 train_one_epoch。
    """

    def __init__(
        self,
        l1_weight=1.0,
        freq_weight=0.01,
        high_freq_ratio=0.25,
        use_log_amp=True,
    ):
        super().__init__()  # 中文注释：调用父类构造函数
        self.l1_weight = float(l1_weight)  # 中文注释：L1 损失权重
        self.freq_weight = float(freq_weight)  # 中文注释：频域损失权重
        self.freq_loss = DirectionalFrequencyLoss(
            high_freq_ratio=high_freq_ratio,
            use_log_amp=use_log_amp,
        )  # 中文注释：实例化频域损失

    def forward(self, sr, hr):
        # 中文注释：像素级 L1 重建损失
        loss_l1 = F.l1_loss(sr, hr)

        # 中文注释：高频幅度谱一致性损失
        loss_freq = self.freq_loss(sr, hr)

        # 中文注释：加权总损失
        loss = self.l1_weight * loss_l1 + self.freq_weight * loss_freq

        # 中文注释：只返回标量，兼容现有 train_one_epoch
        return loss


class AdaptiveBandFrequencyLoss(nn.Module):
    """自适应多频带频域一致性损失。"""

    def __init__(self, num_bands=4, log_amp=True, detach_weight=True, eps=1e-6, use_fp32=True):
        super().__init__()
        self.num_bands = int(num_bands)
        self.log_amp = bool(log_amp)
        self.detach_weight = bool(detach_weight)
        self.eps = float(eps)
        self.use_fp32 = bool(use_fp32)
        self.last_band_weights = None
        self.last_band_losses = None

    def _radius(self, h, w, device, dtype):
        # 中文注释：rfft2 宽度为 w//2+1，x 方向只保留非负频率。
        kw = w // 2 + 1
        fy = torch.arange(h, device=device, dtype=dtype)
        fy = torch.where(fy <= h // 2, fy, fy - h) / max(h // 2, 1)
        fx = torch.arange(kw, device=device, dtype=dtype) / max(w // 2, 1)
        fyg, fxg = torch.meshgrid(fy, fx, indexing="ij")
        # 中文注释：半径裁剪到 [0,1]，便于均匀切分频带。
        return torch.sqrt(fyg * fyg + fxg * fxg + self.eps).clamp(0.0, 1.0).view(1, 1, h, kw)

    def forward(self, sr, hr):
        # 中文注释：ADFC 要求 SR/HR 空间尺寸和通道完全一致。
        assert sr.shape == hr.shape, "sr 和 hr 形状必须一致"
        sr_f = sr.float() if self.use_fp32 else sr
        hr_f = hr.float() if self.use_fp32 else hr
        h, w = sr_f.shape[-2], sr_f.shape[-1]

        # 中文注释：频域幅度默认用 log1p 压缩，降低极大频谱值的支配作用。
        amp_sr = torch.abs(torch.fft.rfft2(sr_f, dim=(-2, -1), norm="ortho"))
        amp_hr = torch.abs(torch.fft.rfft2(hr_f, dim=(-2, -1), norm="ortho"))
        if self.log_amp:
            amp_sr = torch.log1p(amp_sr)
            amp_hr = torch.log1p(amp_hr)

        radius = self._radius(h, w, sr_f.device, amp_sr.dtype)
        band_losses = []
        for idx in range(self.num_bands):
            # 中文注释：按半径把 [0,1] 均匀切成 num_bands 个频带。
            low = float(idx) / float(self.num_bands)
            high = float(idx + 1) / float(self.num_bands)
            if idx == self.num_bands - 1:
                mask = ((radius >= low) & (radius <= high)).to(amp_sr.dtype)
            else:
                mask = ((radius >= low) & (radius < high)).to(amp_sr.dtype)
            diff = torch.abs(amp_sr - amp_hr) * mask
            denom = mask.sum() * sr_f.shape[0] * sr_f.shape[1] + self.eps
            band_losses.append(diff.sum() / denom)

        band_losses = torch.stack(band_losses)
        weights = band_losses / (band_losses.sum() + self.eps)
        if self.detach_weight:
            # 中文注释：默认截断权重梯度，只让频带误差本身参与反传。
            weights = weights.detach()
        loss = torch.sum(weights * band_losses)

        # 中文注释：缓存最近一次频带统计，供 train.py 写日志和 TensorBoard。
        self.last_band_weights = weights.detach()
        self.last_band_losses = band_losses.detach()
        return loss


class HardRegionWeightedL1(nn.Module):
    """空间-频域困难区域重加权 L1。"""

    def __init__(self, sfhr_weight=0.5, eps=1e-6):
        super().__init__()
        self.sfhr_weight = float(sfhr_weight)
        self.eps = float(eps)

    def forward(self, sr, hr):
        # 中文注释：困难区域仅由 HR 构造，不引入 SR 侧的额外梯度路径。
        with torch.no_grad():
            if hr.shape[1] == 3:
                gray = 0.299 * hr[:, 0:1, :, :] + 0.587 * hr[:, 1:2, :, :] + 0.114 * hr[:, 2:3, :, :]
            else:
                gray = hr.mean(dim=1, keepdim=True)
            gray = gray.float()

            # 中文注释：Sobel 梯度刻画边缘强度。
            sobel_x = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
            sobel_y = gray.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
            gx = F.conv2d(gray, sobel_x, padding=1)
            gy = F.conv2d(gray, sobel_y, padding=1)
            grad_mag = torch.sqrt(gx * gx + gy * gy + self.eps)

            # 中文注释：Laplacian 高通响应刻画纹理和突变区域。
            lap = gray.new_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3)
            hpf_mag = torch.abs(F.conv2d(gray, lap, padding=1))
            hard_map = grad_mag + hpf_mag

            # 中文注释：逐图归一化到 [0,1]，避免不同样本动态范围互相影响。
            b = hard_map.shape[0]
            flat = hard_map.view(b, -1)
            min_v = flat.min(dim=1)[0].view(b, 1, 1, 1)
            max_v = flat.max(dim=1)[0].view(b, 1, 1, 1)
            hard_map = (hard_map - min_v) / (max_v - min_v + self.eps)
            weight_map = (1.0 + self.sfhr_weight * hard_map).detach()

        # 中文注释：weight_map 自动广播到所有颜色通道。
        return torch.mean(weight_map.to(sr.dtype) * torch.abs(sr - hr))


class SRAdvancedCompositeLoss(nn.Module):
    """L1 / SFHR-L1 + 可选 ADFC 的组合损失。"""

    def __init__(
        self,
        use_adfc=False,
        adfc_weight=0.01,
        adfc_num_bands=4,
        adfc_log_amp=True,
        adfc_detach_weight=True,
        use_sfhr=False,
        sfhr_weight=0.5,
        sfhr_eps=1e-6,
    ):
        super().__init__()
        self.use_adfc = bool(use_adfc)
        self.adfc_weight = float(adfc_weight)
        self.use_sfhr = bool(use_sfhr)
        self.l1_loss = HardRegionWeightedL1(sfhr_weight=sfhr_weight, eps=sfhr_eps) if self.use_sfhr else nn.L1Loss()
        self.adfc_loss = AdaptiveBandFrequencyLoss(
            num_bands=adfc_num_bands,
            log_amp=adfc_log_amp,
            detach_weight=adfc_detach_weight,
            eps=sfhr_eps,
        ) if self.use_adfc else None
        self.last_l1 = None
        self.last_adfc = None
        self.last_total = None
        self.last_band_weights = None
        self.last_band_losses = None

    def forward(self, sr, hr):
        # 中文注释：基础重建项根据配置选择普通 L1 或困难区域重加权 L1。
        loss_l1 = self.l1_loss(sr, hr)
        total = loss_l1
        if self.use_adfc:
            loss_adfc = self.adfc_loss(sr, hr)
            total = total + self.adfc_weight * loss_adfc
            self.last_adfc = loss_adfc.detach()
            self.last_band_weights = self.adfc_loss.last_band_weights
            self.last_band_losses = self.adfc_loss.last_band_losses
        else:
            self.last_adfc = None
            self.last_band_weights = None
            self.last_band_losses = None

        # 中文注释：缓存最近一次标量，训练循环在 epoch 末写日志。
        self.last_l1 = loss_l1.detach()
        self.last_total = total.detach()
        return total


class FDPPPolishLoss(nn.Module):
    """FCDM 后期 PSNR polish 损失：像素项 + 字典分配保持项。"""

    def __init__(self, mse_weight=10.0, dac_weight=0.01, eps=1e-8, conf_power=1.0):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.dac_weight = float(dac_weight)
        self.eps = float(eps)
        self.conf_power = float(conf_power)
        self.last_l1 = None
        self.last_mse = None
        self.last_pix = None
        self.last_dac = None
        self.last_total = None

    def forward(self, sr, hr, student_aux, teacher_aux):
        # 中文注释：像素项保留 L1 的稳健性，并加入 MSE 直接优化 PSNR。
        loss_l1 = F.l1_loss(sr, hr)
        loss_mse = F.mse_loss(sr, hr)
        loss_pix = loss_l1 + self.mse_weight * loss_mse

        # 中文注释：FDPP 的 DAC 项只读取 return_aux=True 时缓存的 FCDM dictionary attention。
        student_list = student_aux.get("dict_attn", []) if student_aux is not None else []
        teacher_list = teacher_aux.get("dict_attn", []) if teacher_aux is not None else []
        num_layers = min(len(student_list), len(teacher_list))

        if num_layers == 0:
            # 中文注释：没有可用字典 attention 时返回同 device/dtype 的 0，便于日志看出 DAC 未生效。
            loss_dac = sr.new_zeros(())
        else:
            layer_losses = []
            for i in range(num_layers):
                # 中文注释：student attention 保留梯度；teacher attention detach 后作为固定目标分布。
                p_s = student_list[i].clamp_min(self.eps)
                p_t = teacher_list[i].detach().clamp_min(self.eps)
                p_s = p_s / p_s.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                p_t = p_t / p_t.sum(dim=-1, keepdim=True).clamp_min(self.eps)

                # 中文注释：计算 KL(P_t || P_s)，并用 teacher 最大分配概率作为可靠性权重。
                kl = (p_t * (torch.log(p_t) - torch.log(p_s))).sum(dim=-1)
                conf = p_t.max(dim=-1).values.detach()
                if self.conf_power != 1.0:
                    conf = conf.pow(self.conf_power)
                layer_loss = (kl * conf).sum() / (conf.sum() + self.eps)
                layer_losses.append(layer_loss)
            # 中文注释：多层 FCDM dictionary attention 的保持项取平均。
            loss_dac = torch.stack(layer_losses).mean()

        total = loss_pix + self.dac_weight * loss_dac

        # 中文注释：缓存最近一次分项，供 train.py 在 epoch 末写日志和 TensorBoard。
        self.last_l1 = loss_l1.detach()
        self.last_mse = loss_mse.detach()
        self.last_pix = loss_pix.detach()
        self.last_dac = loss_dac.detach()
        self.last_total = total.detach()
        return total
