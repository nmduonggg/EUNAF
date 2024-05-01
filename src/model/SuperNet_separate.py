import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

import math

class SuperNet_separate(nn.Module):
    def __init__(self, scale, input_channel, nblocks):
        super(SuperNet_separate, self).__init__()

        self.nblocks = nblocks
        self.input_channel = input_channel
        self.scale = scale
        
        # heads
        self.heads = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1)
        )
        # body
        self.body = nn.ModuleList([
            BasicBlock(32, self.scale) for _ in range(4)])
        # tail
        self.tails = nn.ModuleList([
            UpSampler(32, self.scale, self.input_channel) for _ in range(4)
        ])
        self.mask_predictors = MaskPredictor(input_channel=32, nblocks=4)
        self.gumbel_softmax = GumbelSoftmax()
        
    def predict_mask(self, x):
        return self.mask_predictors(x)
    
    def forward(self, x):
        x = self.heads(x)
        masks = self.mask_predictors(x.clone().detach())
        
        out_h, out_w = x.size(2)*self.scale, x.size(3)*self.scale
        out_means = []
        
        outs = []
        for i in range(self.nblocks):
            shortcut = x.clone()
            x = self.body[i](x)
            _, _, h, w = x.shape
            x = x + shortcut
            if i==self.nblocks-1:
                outs.append(self.tails[i](x))
            else:
                outs.append(self.tails[i](x.clone().detach()))
            
        return [outs, masks]
    
    def fuse_2_blocks(self, x, idxs, keep):
        """Fuse 2 blocks theoretically

        Args:
            idxs (list): List of indices of blocks
            keep (float): keep rate of 1st image

        Returns:
            out: fused image
        """
        assert 0 <= keep <= 1
        assert len(idxs)==2
        with torch.no_grad():
            yfs, masks = self.forward(x)
        # percentile filter - get the r% pixels with highest uncertainty
        hard_mask = masks[idxs[0]].clone().cpu().numpy()
        hard_mask = (hard_mask > np.percentile(hard_mask, keep*100)).astype(int)
        hard_mask = torch.tensor(hard_mask)
            
        hm = hard_mask.to(yfs[0].device)
        y = yfs[idxs[0]] * (1-hm) + yfs[idxs[1]] * hm
        
        return y
        
class BasicBlock(nn.Module):
    def __init__(self, channels, scale=2):
        super(BasicBlock, self).__init__()
        self.channels = channels
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU(True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1))

    def forward(self, x):
        B, C, H, W = x.size()
        shortcut = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + shortcut
        x = F.relu(x)
        
        return x
class UpSampler(nn.Module):
    """Upsamler = Conv + PixelShuffle
    This class is hard-code for scale factor of 2"""
    def __init__(self, n_features, scale, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_features, scale*scale*n_features, 3, 1, 1))
        self.dropout = nn.Dropout(p=0.01)
        self.shuffler = nn.PixelShuffle(scale)
        self.finalizer = nn.Conv2d(n_features, out_channel, 1, 1, 0)
        
    def forward(self, x):
        # x = self.dropout(x)
        x = self.conv1(x)
        x = self.shuffler(x)
        x = self.finalizer(x)
        return x
        
class GumbelSoftmax(nn.Module):
    '''
        Gumbel softmax gate
    '''
    def __init__(self, tau=1):
        super(GumbelSoftmax, self).__init__()
        self.tau = tau
        self.sigmoid = nn.Sigmoid()
        
    def gumbel_sample(self, template_tensor, eps=1e-8):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumbel_samples_tensor = torch.log(uniform_samples_tensor+eps) - torch.log(1-uniform_samples_tensor+eps)
        return gumbel_samples_tensor
    
    def gumbel_softmax(self, logits):
        """draw a sample from gumbel-softmax distribution
        """
        gsamples = self.gumbel_sample(logits.data)
        logits = logits + Variable(gsamples)
        soft_samples = self.sigmoid(logits / self.tau)
        
        return soft_samples, logits
    
    def forward(self, logits):
        if not self.training:
            out_hard = (logits>=0).float()
            return out_hard
        
        out_soft, prob_soft = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard
    
class MaskPredictor(nn.Module):
    def __init__(self, input_channel, nblocks):
        super().__init__()
        self.nblocks=nblocks
        self.heads = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1)
        )
        self.predictors = nn.ModuleList([
            MaskBlock(32) for _ in range(nblocks)
        ])
        
    def forward(self, x):
        x = self.heads(x)
        masks = [self.predictors[i](x) for i in range(self.nblocks)]
        return masks
    

class MaskBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, 1, 1, 1), nn.ReLU(),
            nn.Conv2d(channels//2, channels//4, 3, 1, 1, 1), nn.ReLU(),
        )
        self.upsampling = UpSampler(channels//4, 2, 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.upsampling(x)
        return x
    