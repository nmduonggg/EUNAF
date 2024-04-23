import tqdm
from torch.utils.data import Dataset
from data.common import load_image_as_Tensor, get_patch

class DIV2K_validset(Dataset):
    def __init__(self, root, scale=2, style='RGB', rgb_range=1.0):
        super(DIV2K_validset, self).__init__()

        self.N_raw_image = 100 #801-810
        self.N = self.N_raw_image

        self.X, self.Y = [], []
        self.scale = scale
        
        for i in tqdm.tqdm(range(self.N_raw_image)):
            X_im_file_name = root + 'DIV2K_valid_LR_bicubic/X' + str(scale) + '/' + ('%04dx%d.png' % (i+801,scale))
            X_data = load_image_as_Tensor(X_im_file_name, style, rgb_range)
            self.X.append(X_data)

            Y_im_file_name = root + 'DIV2K_valid_HR/' + ('%04d.png' % (i+801))
            Y_data = load_image_as_Tensor(Y_im_file_name, style, rgb_range)
            self.Y.append(Y_data)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        im_lr = self.X[idx]
        im_hr = self.Y[idx]

        return im_lr, im_hr