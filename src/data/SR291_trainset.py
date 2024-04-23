import numpy as np
import torch
import os
import skimage.color as sc
import skimage.transform as st
import imageio
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import Dataset

class SR291_trainset(Dataset):
    def __init__(self, root, max_load=291, lr_patch_size=21, scale=2, style='RGB', rgb_range=1.0):
        super(SR291_trainset, self).__init__()

        file_list = os.listdir(root)

        self.root = root
        self.repeat = 1024
        if max_load > 0:
            self.N_raw_image = min(max_load, 291)
        else:
            self.N_raw_image = 291
        self.N = self.N_raw_image * self.repeat

        self.X, self.Y = [], []
        self.lr_patch_size = lr_patch_size
        self.scale = scale
        self.style = style

        for file_name in tqdm.tqdm(file_list, total=len(file_list)):
            Y_im_file_name = root + file_name
            Y_data = imageio.imread(Y_im_file_name)

            ih, iw, ic = Y_data.shape
            nih = ih - ih % scale
            niw = iw - iw % scale

            Y_data = Y_data[:nih, :niw, :]
            X_data = st.resize(Y_data, (nih//scale, niw//scale))

            if style == 'RGB':
                if X_data.ndim == 2:
                    X_data = np.stack((X_data,)*3, axis=-1)
            elif style == 'Y':
                if X_data.ndim == 2:
                    X_data = np.expand_dims(X_data, axis=2)
                
                if X_data.shape[2] == 3:
                    X_data = np.expand_dims(sc.rgb2ycbcr(X_data)[:, :, 0], 2)
            else:
                print('[ERRO] unknown style; should be Y or RGB')
                assert(0)

            X_data = np.ascontiguousarray(np.transpose(X_data, (2,0,1)))
            X_data = torch.Tensor(X_data).float().mul_(rgb_range/255.)
            self.X.append(X_data)

            if style == 'RGB':
                if Y_data.ndim == 2:
                    Y_data = np.stack((Y_data,)*3, axis=-1)
            elif style == 'Y':
                if Y_data.ndim == 2:
                    Y_data = np.expand_dims(Y_data, axis=2)
                
                if Y_data.shape[2] == 3:
                    Y_data = np.expand_dims(sc.rgb2ycbcr(Y_data)[:, :, 0], 2)
            else:
                print('[ERRO] unknown style; should be Y or RGB')
                assert(0)

            Y_data = np.ascontiguousarray(np.transpose(Y_data, (2,0,1)))
            Y_data = torch.Tensor(Y_data).float().mul_(rgb_range/255.)
            self.Y.append(Y_data)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        im_idx = idx % self.N_raw_image
        im_hr = self.Y[im_idx]
        im_lr = self.X[im_idx]

        im_lr_patch, im_hr_patch = get_patch(im_lr, im_hr, self.lr_patch_size, self.scale)

        if np.random.uniform(0,1) < 0.5:
            im_lr_patch = transforms.functional.hflip(im_lr_patch)
            im_hr_patch = transforms.functional.hflip(im_hr_patch)

        if np.random.uniform(0,1) < 0.5:
            im_lr_patch = transforms.functional.vflip(im_lr_patch)
            im_hr_patch = transforms.functional.vflip(im_hr_patch)
            
        if np.random.uniform(0,1) < 0.5:
            im_lr_patch = transforms.functional.rotate(im_lr_patch, 90)
            im_hr_patch = transforms.functional.rotate(im_hr_patch, 90)

        return im_lr_patch, im_hr_patch