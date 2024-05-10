import os
import torch
import shutil
import numpy as np
import scipy.stats as stats
import torch.nn.functional as F

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def confidence_interval_mean(estimated_mean, estimated_stddev, sample_size, confidence_level):
    """
    Calculate the confidence interval for the population mean.
    
    Parameters:
        estimated_mean (float): The estimated mean of the sample.
        estimated_stddev (float): The estimated standard deviation of the sample.
        sample_size (int): The size of the sample.
        confidence_level (float): The desired confidence level (e.g., 0.95 for 95% confidence).
    
    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    # Calculate the critical value based on the confidence level
    z_critical = stats.norm.ppf((1 + confidence_level) / 2)  # Two-tailed test
    
    # Calculate the margin of error
    margin_of_error = z_critical * (abs(estimated_stddev) / (sample_size ** 0.5))
    
    # Calculate the confidence interval
    lower_bound = estimated_mean - margin_of_error
    upper_bound = estimated_mean + margin_of_error
    
    return lower_bound, upper_bound

def resize_image_tensor(lr_image, hr_image, scale, rgb_range):
    hr_height, hr_width = hr_image.shape[2:4]
    lr_height, lr_width = lr_image.shape[2:4]
    if lr_height * scale != hr_height or lr_width * scale != lr_width:
        hr_width = lr_width * scale
        hr_height = lr_height * scale
        hr_image = F.interpolate(hr_image, size=(hr_height, hr_width), mode='bicubic')
        hr_image.clamp(min=0, max = rgb_range)
    return hr_image

def crop_cpu(img, crop_sz, step):
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=[]
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            lr_list.append(crop_img)
    h=x + crop_sz
    w=y + crop_sz
    return lr_list, num_h, num_w, h, w

def combine(sr_list, num_h, num_w, h, w, patch_size, step, scale):
    index=0
    sr_img = np.zeros((h*scale, w*scale, 3), 'float32')
    for i in range(num_h):
        for j in range(num_w):
            sr_img[i*step*scale:i*step*scale+patch_size*scale, j*step*scale:j*step*scale+patch_size*scale,:] += sr_list[index]
            index+=1
    sr_img=sr_img.astype('float32')

    for j in range(1,num_w):
        sr_img[:,j*step*scale:j*step*scale+(patch_size-step)*scale,:]/=2

    for i in range(1,num_h):
        sr_img[i*step*scale:i*step*scale+(patch_size-step)*scale,:,:]/=2
    return sr_img

class LrScheduler:
    def __init__(self, optimizer, base_lr, lr_decay_ratio, epoch_step):
        self.base_lr = base_lr
        self.lr_decay_ratio = lr_decay_ratio
        self.epoch_step = epoch_step
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (self.lr_decay_ratio ** (epoch // self.epoch_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if epoch%self.epoch_step==0:
                print('[INFO] Setting learning_rate to %.2E'%lr)
                
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False