from __future__ import print_function
import argparse
import os
import torch
import cv2
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform
from os import listdir
import math
from model_archs.model_factory import build_model_by_type, is_scdr_radr_model, normalize_model_type
from train_4x import build_arg_parser as build_train_arg_parser
from train_4x import model_kwargs_from_opt, normalize_options
from utils.checkpoint_utils import (
    _load_model_state_compatible,
    load_torch_file,
    resume_model_only_checkpoint,
    strip_module_prefix,
)


# ---load model architecture---
def get_model(model_type, opt=None):
    model_type = normalize_model_type(model_type)
    if is_scdr_radr_model(model_type):
        train_parser = build_train_arg_parser()
        train_opt = train_parser.parse_args([])
        if opt is not None:
            for key, value in vars(opt).items():
                if hasattr(train_opt, key):
                    setattr(train_opt, key, value)
        train_opt.model_type = "scdr_radr"
        train_opt = normalize_options(train_opt)
        return build_model_by_type("scdr_radr", scdr_radr_kwargs=model_kwargs_from_opt(train_opt))
    if model_type == 'ttst':
        from model_archs.TTST_arc import TTST
        return TTST()
    elif model_type == 'edsr':
        from model_archs.edsr import EDSR
        return EDSR()
    elif model_type == 'rcan':
        from model_archs.rcan import RCAN
        return RCAN()
    elif model_type == 'han':
        from model_archs.han import HAN
        return HAN()
    elif model_type in ('nlsa', 'nlsn'):
        from model_archs.nlsn import NLSN
        return NLSN()
    elif model_type == 'hsenet':
        from model_archs.hsenet import HSENET
        return HSENET()
    elif model_type == 'haunet':
        from model_archs.haunet import HAUNet
        return HAUNet()
    elif model_type == 'transenet':
        from model_archs.transenet import TransENet
        return TransENet()
    elif model_type == 'hat_l':
        try:
            from model_archs.hat_arch import HAT_L
            return HAT_L()
        except ImportError:
            print("Warning: HAT_L not found, falling back to HAT")
            from model_archs.hat_arch import HAT
            return HAT()
    else:
        raise ValueError(f"Model type '{model_type}' is not supported check model_archs folder.")


import glob
import numpy as np
import socket
import time
import imageio
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

# Test settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='training batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')

# 将 data_dir 设置为包含所有数据集的父目录
parser.add_argument('--data_dir', type=str, default='../TTST_datasets/test/')

parser.add_argument('--model_type', type=str, default='scdr_radr')
parser.add_argument('--pretrained_sr', default='saved_models/scdr_radr/scdr_radr_best.pth',
                    help='sr pretrained base model')
parser.add_argument('--save_folder', default='results/', help='Location to save checkpoint models')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id to use')
parser.add_argument('--half', action='store_true', help='use CUDA autocast fp16 inference')
parser.add_argument('--val_save_images', type=int, default=1, help='1: save SR images; 0: metrics only')
parser.add_argument('--use_ema', type=int, default=1,
                    help='1: load ema_model from SCDR_RADR checkpoint when available; 0: load raw model')
parser.add_argument('--use_radr', type=int, default=1)
parser.add_argument('--radr_layer_mode', type=str, default='last', choices=['none', 'last', 'all'])
parser.add_argument('--radr_hidden_dim', type=int, default=32)
parser.add_argument('--radr_lambda', type=float, default=0.05)
parser.add_argument('--radr_tau', type=float, default=0.50)
parser.add_argument('--radr_init_bias', type=float, default=-4.0)
parser.add_argument('--radr_detach_feat', type=int, default=1)
parser.add_argument('--radr_use_correction', type=int, default=1)
parser.add_argument('--radr_corr_hidden_dim', type=int, default=64)
parser.add_argument('--radr_corr_lambda', type=float, default=0.1)
parser.add_argument('--radr_corr_scale', type=float, default=0.2)
parser.add_argument('--radr_corr_init_std', type=float, default=0.0005)
parser.add_argument('--radr_corr_detach_residual', type=int, default=1)
parser.add_argument('--radr_corr_gate_mode', type=str, default='sqrt',
                    choices=['ueff', 'sqrt', 'binary', 'none'])
parser.add_argument('--use_scdr', type=int, default=1)
parser.add_argument('--scdr_alpha', type=float, default=0.49)
parser.add_argument('--scdr_return_routes', type=int, default=0)
parser.add_argument('--scdr_detach_routes', type=int, default=0)
parser.add_argument('--use_scdr_adapter', type=int, default=1)
parser.add_argument('--scdr_adapter_hidden_dim', type=int, default=32)
parser.add_argument('--scdr_adapter_scale', type=float, default=0.05)
parser.add_argument('--scdr_adapter_init_std', type=float, default=1e-5)
parser.add_argument('--radr_corr_feature_mode', type=str, default='shortcut_xatd',
                    choices=['shortcut_xatd', 'shortcut_only', 'shortcut_xwin', 'shortcut_xaca'])
parser.add_argument('--radr_corr_train_feature_modes', type=str, default='')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
cuda = opt.gpu_mode
print(opt)

current_time = time.strftime("%H-%M-%S")
pth_name = os.path.splitext(os.path.basename(opt.pretrained_sr))[0]
opt.save_folder = os.path.join(opt.save_folder, opt.model_type, pth_name, current_time) + '/'

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)

transform = transform.Compose([transform.ToTensor(), ])


def rgb_to_y(img):
    # img: uint8 RGB, HWC
    img = img.astype(np.float32)
    y = 16.0 + (65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]) / 255.0
    return y


def calc_psnr_y(pred, gt, border=4):
    pred_y = rgb_to_y(pred)
    gt_y = rgb_to_y(gt)
    if border > 0:
        pred_y = pred_y[border:-border, border:-border]
        gt_y = gt_y[border:-border, border:-border]
    mse = np.mean((pred_y - gt_y) ** 2)
    if mse == 0: return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim_matlab(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim_y(pred, gt, border=4):
    pred_y = rgb_to_y(pred)
    gt_y = rgb_to_y(gt)
    if border > 0:
        pred_y = pred_y[border:-border, border:-border]
        gt_y = gt_y[border:-border, border:-border]
    return ssim_matlab(pred_y, gt_y)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %f M' % (num_params / 1e6))


def load_scdr_radr_weights(model, ckpt_path, use_ema=1):
    if use_ema:
        checkpoint = load_torch_file(ckpt_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'ema_model' in checkpoint:
            state = strip_module_prefix(checkpoint['ema_model'])
            target_model = model.module if hasattr(model, 'module') else model
            _load_model_state_compatible(target_model, state, logger=print, label='ema_model_as_model')
            print('Loaded SCDR_RADR EMA weights from checkpoint.')
            print(
                'Resumed EMA weights {} at epoch {}, best_psnr {:.4f}, best_ssim {:.4f}, best_epoch {}'.format(
                    ckpt_path,
                    int(checkpoint.get('epoch', 0)),
                    float(checkpoint.get('best_psnr', 0.0)),
                    float(checkpoint.get('best_ssim', 0.0)),
                    int(checkpoint.get('best_epoch', 0)),
                )
            )
            return
        print('[Warning] --use_ema=1 but checkpoint has no ema_model; falling back to raw model weights.')
    resume_model_only_checkpoint(model, ckpt_path, logger=print)


if opt.gpu_mode and torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)
    device = 'cuda:{}'.format(opt.gpu_id)
else:
    device = 'cpu'
print('===> Building model ', opt.model_type)
model = get_model(opt.model_type, opt)
print('---------- Networks architecture -------------')
print_network(model)

model_name = os.path.join(opt.pretrained_sr)
if os.path.exists(model_name):
    if is_scdr_radr_model(opt.model_type):
        load_scdr_radr_weights(model, model_name, use_ema=opt.use_ema)
    else:
        state_dict = torch.load(model_name, map_location=device)
        from collections import OrderedDict

        if isinstance(state_dict, dict):
            state_dict = state_dict.get('model', state_dict.get('state_dict', state_dict.get('model_state_dict',
                                                                                             state_dict.get('params',
                                                                                                            state_dict))))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    print('Pre-trained SR model is loaded.')
else:
    print('No pre-trained model!!!!')

if device != 'cpu':
    model = model.cuda(opt.gpu_id)
    if opt.gpus > 1 and not is_scdr_radr_model(opt.model_type):
        model = torch.nn.DataParallel(model, device_ids=gpus_list)
else:
    model = model.cpu()


def eval(dataset_name, folder_name):
    if folder_name in ['', '/']:
        print(f'===> Loading val datasets from {dataset_name} root directory (No Subfolders)')
    else:
        print(f'===> Loading val datasets: {dataset_name} -> {folder_name}')

    LR_dir = os.path.join(opt.data_dir, dataset_name, 'LR', folder_name)
    GT_dir = os.path.join(opt.data_dir, dataset_name, 'GT', folder_name)

    # 结果按数据集划分文件夹
    save_dir = os.path.join(opt.save_folder, dataset_name, folder_name) if folder_name else os.path.join(
        opt.save_folder, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    LR_image = []
    for ext in exts:
        LR_image.extend(glob.glob(os.path.join(LR_dir, ext)))
    LR_image = sorted([p for p in LR_image if os.path.isfile(p)])

    if len(LR_image) == 0:
        print(f'[Warning] No images found in: {LR_dir}')
        return None, None, 0

    model.eval()

    psnr_list = []
    ssim_list = []

    for i, img_path in enumerate(LR_image):
        lr = Image.open(img_path).convert('RGB')
        lr = transform(lr).unsqueeze(0).to(device)

        with torch.no_grad():
            t0 = time.time()
            with torch.cuda.amp.autocast(enabled=bool(opt.half and device != 'cpu')):
                prediction = model(lr)
            t1 = time.time()

        if isinstance(prediction, dict):
            prediction = prediction.get("sr", prediction.get("final_sr"))
        elif isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        prediction = prediction.cpu().data[0].numpy().astype(np.float32)
        prediction = (prediction * 255.0).clip(0, 255)
        prediction = prediction.transpose(1, 2, 0)
        prediction = np.round(prediction).astype(np.uint8)

        # print("===> Processing image: %s || Timer: %.4f sec." % (img_path, (t1 - t0)))

        # 保存结果
        if opt.val_save_images:
            save_name = os.path.splitext(os.path.basename(img_path))[0] + '.png'
            save_fn = os.path.join(save_dir, save_name)
            Image.fromarray(prediction).save(save_fn)

        # 读取对应 GT
        gt_name = os.path.basename(img_path)
        gt_path = os.path.join(GT_dir, gt_name)

        if not os.path.isfile(gt_path):
            print(f'[Warning] GT not found: {gt_path}, skip metric for this image.')
            continue

        gt = Image.open(gt_path).convert('RGB')
        gt = np.array(gt).astype(np.uint8)

        # 防止尺寸不一致
        if gt.shape != prediction.shape:
            print(f'[Warning] Shape mismatch: pred {prediction.shape}, gt {gt.shape}, skip metric.')
            continue

        cur_psnr = calc_psnr_y(prediction, gt)
        cur_ssim = calc_ssim_y(prediction, gt)
        print("===> Processing image: %s  || PSNR: %.4f dB || SSIM: %.4f" % (
            img_path, cur_psnr, cur_ssim))
        psnr_list.append(cur_psnr)
        ssim_list.append(cur_ssim)

    if len(psnr_list) == 0:
        return None, None, 0

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    display_name = folder_name.strip('/') if folder_name else "Root Directory"
    print(
        f'====> Folder [{display_name}] Average PSNR: {avg_psnr:.4f} dB | Average SSIM: {avg_ssim:.4f} | Images: {len(psnr_list)}')

    # 将每个数据集的结果统一写入一个 metrics 文件，或按数据集分开写
    metrics_path = os.path.join(opt.save_folder, 'metrics.txt')
    with open(metrics_path, 'a', encoding='utf-8') as f:
        f.write(f'[{dataset_name}] {display_name}: PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}, N={len(psnr_list)}\n')

    return avg_psnr, avg_ssim, len(psnr_list)


if __name__ == '__main__':
    AID_class_name = ['Airport/', 'BareLand/', 'BaseballField/', 'Beach/', 'Bridge/', 'Center/', 'Church/',
                      'Commercial/', 'DenseResidential/',
                      'Desert/', 'Farmland/', 'Forest/', 'Industrial/', 'Meadow/', 'MediumResidential/', 'Mountain/',
                      'Park/', 'Parking/', 'Playground/',
                      'Pond/', 'Port/', 'RailwayStation/', 'Resort/', 'River/', 'School/', 'SparseResidential/',
                      'Square/', 'Stadium/', 'StorageTanks/', 'Viaduct/']

    if not os.path.exists(opt.data_dir):
        print(f"Error: Data directory {opt.data_dir} does not exist!")
        exit()

    # 获取 test 目录下的所有子文件夹（即每个数据集的根目录）
    datasets = [d for d in os.listdir(opt.data_dir) if os.path.isdir(os.path.join(opt.data_dir, d))]
    print(f"Found datasets to evaluate: {datasets}")

    # ================= 新增：全局统计变量 =================
    global_psnr_sum = 0.0
    global_ssim_sum = 0.0
    global_count = 0
    dataset_avg_psnr_sum = 0.0
    dataset_avg_ssim_sum = 0.0
    dataset_avg_count = 0
    # ====================================================

    for dataset_name in datasets:
        print(f"\n{'=' * 50}")
        print(f"Starting Evaluation for Dataset: {dataset_name}")
        print(f"{'=' * 50}")

        LR_base_dir = os.path.join(opt.data_dir, dataset_name, 'LR')

        if not os.path.exists(LR_base_dir):
            print(f"[Warning] LR directory not found for {dataset_name} at {LR_base_dir}. Skipping...")
            continue

        # 检查是否存在 AID 的特色子文件夹
        is_aid_dataset = any(os.path.exists(os.path.join(LR_base_dir, c)) for c in AID_class_name[:3])

        if is_aid_dataset:
            print(f"===> Detected AID structure for {dataset_name}. Using AID_class_name list.")
            folder_list = AID_class_name
        else:
            print(f"===> Flat directory structure detected for {dataset_name}.")
            folder_list = ['']  # 空字符串代表直接读取 LR_base_dir 根目录

        # 初始化当前数据集的统计变量
        dataset_psnr_sum = 0.0
        dataset_ssim_sum = 0.0
        dataset_count = 0

        for folder in folder_list:
            avg_psnr, avg_ssim, count = eval(dataset_name=dataset_name, folder_name=folder)
            if count > 0:
                dataset_psnr_sum += avg_psnr * count
                dataset_ssim_sum += avg_ssim * count
                dataset_count += count

        # 打印并保存当前数据集的总计结果
        if dataset_count > 0:
            dataset_avg_psnr = dataset_psnr_sum / dataset_count
            dataset_avg_ssim = dataset_ssim_sum / dataset_count

            print(f'\n---------- {dataset_name} Overall Results ----------')
            print(f'Overall Average PSNR: {dataset_avg_psnr:.4f} dB')
            print(f'Overall Average SSIM: {dataset_avg_ssim:.4f}')
            print(f'Total Images: {dataset_count}')

            metrics_path = os.path.join(opt.save_folder, 'metrics.txt')
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(f'\n---------- {dataset_name} Overall Results ----------\n')
                f.write(f'Overall Average PSNR: {dataset_avg_psnr:.4f} dB\n')
                f.write(f'Overall Average SSIM: {dataset_avg_ssim:.4f}\n')
                f.write(f'Total Images: {dataset_count}\n\n')

            # ================= 新增：累加到全局统计变量 =================
            global_psnr_sum += dataset_psnr_sum
            global_ssim_sum += dataset_ssim_sum
            global_count += dataset_count
            dataset_avg_psnr_sum += dataset_avg_psnr
            dataset_avg_ssim_sum += dataset_avg_ssim
            dataset_avg_count += 1
            # =========================================================

    # ================= 新增：计算并输出所有数据集的最终总指标 =================
    if global_count > 0:
        global_avg_psnr = global_psnr_sum / global_count
        global_avg_ssim = global_ssim_sum / global_count
        dataset_mean_psnr = dataset_avg_psnr_sum / dataset_avg_count if dataset_avg_count > 0 else None
        dataset_mean_ssim = dataset_avg_ssim_sum / dataset_avg_count if dataset_avg_count > 0 else None

        print(f"\n{'#' * 50}")
        print(f"🌟 FINAL GLOBAL RESULTS (Across all datasets) 🌟")
        print(f"{'#' * 50}")
        if dataset_mean_psnr is not None:
            print(f'Dataset Average PSNR: {dataset_mean_psnr:.4f} dB')
            print(f'Dataset Average SSIM: {dataset_mean_ssim:.4f}')
        print(f'Image-weighted Overall Average PSNR: {global_avg_psnr:.4f} dB')
        print(f'Image-weighted Overall Average SSIM: {global_avg_ssim:.4f}')
        print(f'Total Evaluated Images: {global_count}')
        print(f"{'#' * 50}\n")

        metrics_path = os.path.join(opt.save_folder, 'metrics.txt')
        with open(metrics_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'#' * 50}\n")
            f.write(f"FINAL GLOBAL RESULTS (Across all datasets)\n")
            f.write(f"{'#' * 50}\n")
            if dataset_mean_psnr is not None:
                f.write(f'Dataset Average PSNR: {dataset_mean_psnr:.4f} dB\n')
                f.write(f'Dataset Average SSIM: {dataset_mean_ssim:.4f}\n')
            f.write(f'Image-weighted Overall Average PSNR: {global_avg_psnr:.4f} dB\n')
            f.write(f'Image-weighted Overall Average SSIM: {global_avg_ssim:.4f}\n')
            f.write(f'Total Evaluated Images: {global_count}\n')
            f.write(f"{'#' * 50}\n")
