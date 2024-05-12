import torch
import math

def calculate_psnr(sr, hr, scale=2, rgb_range=1.0, rgb_channel=False):
    if hr.nelement() == 1: return 0

    diff = (sr - hr).data.div_(rgb_range)

    if diff.size(1) > 1 and not rgb_channel:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    valid = diff[..., scale:-scale, scale:-scale]
    mse = valid.pow(2).mean()

    return torch.as_tensor(-10 * math.log10(mse))