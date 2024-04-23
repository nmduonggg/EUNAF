import numpy as np
import torch
import os
import skimage.color as sc
import imageio
from torch.utils.data import Dataset

class SetN_testset(Dataset):
    def __init__(self, root, scale=2, style='RGB', rgb_range=1.0):
        super(SetN_testset, self).__init__()
        self.X, self.Y = [], []
        X_root = root + 'LR_bicubic/X' + str(scale) + '/'
        Y_root = root + 'HR/'
        X_image_list = os.listdir(X_root)
        Y_image_list = os.listdir(Y_root)
        X_image_list.sort()
        Y_image_list.sort()

        for (X_image_file, Y_image_file) in zip(X_image_list, Y_image_list):
            X_image = imageio.imread(X_root + X_image_file)

            if style == 'RGB':
                if X_image.ndim == 2:
                    X_image = np.stack((X_image,)*3, axis=-1)
            elif style == 'Y':
                if X_image.ndim == 2:
                    X_image = np.expand_dims(X_image, axis=2)

                if X_image.shape[2] == 3:
                    X_image = np.expand_dims(sc.rgb2ycbcr(X_image)[:, :, 0], 2)
            else:
                print('[ERRO] unknown style for DIV2K; should be Y or RGB')
                assert(0)

            X_image = np.ascontiguousarray(X_image.transpose((2, 0, 1)))
            X_image = torch.from_numpy(X_image).float()
            X_image.mul_(rgb_range / 255.)

            ic, ih, iw = X_image.shape

            self.X.append(X_image)

            Y_image = imageio.imread(Y_root + Y_image_file)

            if style == 'RGB':
                if Y_image.ndim == 2:
                    Y_image = np.stack((Y_image,)*3, axis=-1)
            elif style == 'Y':
                if Y_image.ndim == 2:
                    Y_image = np.expand_dims(Y_image, axis=2)

                if Y_image.shape[2] == 3:
                    Y_image = np.expand_dims(sc.rgb2ycbcr(Y_image)[:, :, 0], 2)
            else:
                print('[ERRO] unknown style for DIV2K; should be Y or RGB')
                assert(0)
                
            Y_image = np.ascontiguousarray(Y_image.transpose((2, 0, 1)))
            Y_image = torch.from_numpy(Y_image).float()
            Y_image.mul_(rgb_range / 255.)

            Y_image = Y_image[:,:ih*scale,:iw*scale]

            self.Y.append(Y_image)

        self.N = len(self.X)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]