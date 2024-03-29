# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F
from einops import rearrange


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 4, 1, 2, 3)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 4, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3, 4], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1, 1], device=x.device)
    translation_z = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1, 1], device=x.device)
    grid_batch, grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        torch.arange(x.size(4), dtype=torch.long, device=x.device),
        indexing="ij"
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    grid_z = torch.clamp(grid_z + translation_z + 1, 0, x.size(4) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 4, 1).contiguous()[grid_batch, grid_x, grid_y, grid_z].permute(0, 4, 1, 2, 3)
    return x


def rand_cutout(x, ratio=0.5, apply_ratio=1.0):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5), int(x.size(4) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1, 1], device=x.device)
    offset_z = torch.randint(0, x.size(4) + (1 - cutout_size[2] % 2), size=[x.size(0), 1, 1, 1], device=x.device)
    grid_batch, grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[2], dtype=torch.long, device=x.device),
        indexing="ij"
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    grid_z = torch.clamp(grid_z + offset_z - cutout_size[2] // 2, min=0, max=x.size(4) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), x.size(4), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y, grid_z] = 0

    apply_mask = (torch.rand(x.size(0), dtype=x.dtype, device=x.device) > apply_ratio).to(x.dtype)
    apply_mask = rearrange(apply_mask, "b -> b () () ()")
    mask = (mask + apply_mask).clamp(max=1)

    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
