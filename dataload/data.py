"""SCDR_RADR dataset factory, wrapping DatasetFromFolder / DatasetFromFolderEval."""
from os.path import join

from torchvision.transforms import Compose, ToTensor

from dataload.dataset import DatasetFromFolder, DatasetFromFolderEval


def transform():
    # 中文注释: 默认只做 ToTensor (将 [0,255] PIL 图转为 [0,1] FloatTensor)。
    # SCDR_RADR already handles mean/img_range internally, so no extra Normalize is needed.
    return Compose([ToTensor()])


def get_training_set(data_dir, upscale_factor, patch_size, data_augmentation,
                     use_radr_label=False, radr_label_index=""):
    """约定: data_dir 下并列 GT/ 与 LR/。"""
    hr_dir = join(data_dir, "GT")
    lr_dir = join(data_dir, "LR")
    return DatasetFromFolder(
        hr_dir,
        lr_dir,
        patch_size,
        upscale_factor,
        data_augmentation,
        transform=transform(),
        use_radr_label=use_radr_label,
        radr_label_index=radr_label_index,
    )


def get_eval_set(data_dir, upscale_factor):
    hr_dir = join(data_dir, "GT")
    lr_dir = join(data_dir, "LR")
    return DatasetFromFolderEval(hr_dir, lr_dir, upscale_factor, transform=transform())
