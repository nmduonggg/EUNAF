import torch
import math
import numpy as np

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

# def calculate_psnr(sr, hr, scale=2, rgb_range=1.0, rgb_channel=False):
#     # img1 and img2 have range [0, 255]
    
#     sr = sr.squeeze(0).permute(1,2,0).cpu().numpy()
#     hr = hr.squeeze(0).permute(1,2,0).cpu().numpy()
    
#     img1 = (sr*255).astype(np.float64)
#     img2 = (hr*255).astype(np.float64)
#     mse = np.mean((img1 - img2)**2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(255.0 / math.sqrt(mse))