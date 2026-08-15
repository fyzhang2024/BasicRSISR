import torch
import torch.nn.functional as F


_GAUSSIAN_CACHE = {}


def quantize_rgb_255(img):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img.float().clamp(0.0, 1.0).mul(255.0).round()


def rgb_to_y_torch(img):
    img_255 = quantize_rgb_255(img)
    r = img_255[:, 0:1, :, :]
    g = img_255[:, 1:2, :, :]
    b = img_255[:, 2:3, :, :]
    return 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def calc_psnr(pred, gt, border=4):
    pred_y = rgb_to_y_torch(pred)
    gt_y = rgb_to_y_torch(gt)
    if border > 0 and pred_y.size(-2) > border * 2 and pred_y.size(-1) > border * 2:
        pred_y = pred_y[:, :, border:-border, border:-border]
        gt_y = gt_y[:, :, border:-border, border:-border]
    mse = F.mse_loss(pred_y, gt_y)
    if mse.item() == 0:
        return torch.tensor(100.0, device=pred.device)
    return 20.0 * torch.log10(torch.tensor(255.0, device=pred.device) / torch.sqrt(mse))


def _gaussian_window(channel, device, dtype, window_size=11, sigma=1.5):
    key = (channel, device.type, device.index, str(dtype), window_size, float(sigma))
    if key in _GAUSSIAN_CACHE:
        return _GAUSSIAN_CACHE[key]
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).view(1, 1, window_size, window_size)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    _GAUSSIAN_CACHE[key] = window
    return window


def _ssim(img1, img2, window_size=11, sigma=1.5):
    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)
    channel = img1.size(1)
    window = _gaussian_window(channel, img1.device, img1.dtype, window_size, sigma)
    mu1 = F.conv2d(img1, window, groups=channel)
    mu2 = F.conv2d(img2, window, groups=channel)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=channel) - mu1_mu2
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def calc_ssim(pred, gt, border=4):
    pred_y = rgb_to_y_torch(pred)
    gt_y = rgb_to_y_torch(gt)
    if border > 0 and pred_y.size(-2) > border * 2 and pred_y.size(-1) > border * 2:
        pred_y = pred_y[:, :, border:-border, border:-border]
        gt_y = gt_y[:, :, border:-border, border:-border]
    window_size = min(11, pred_y.size(-2), pred_y.size(-1))
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return torch.tensor(1.0, device=pred.device)
    sigma = 1.5 if window_size == 11 else 1.5 * window_size / 11.0
    return _ssim(pred_y, gt_y, window_size=window_size, sigma=sigma)
