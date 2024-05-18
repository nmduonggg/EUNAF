import torch.nn as nn
from model import common
import torch

import functools
import torch.nn as nn
import torch.nn.functional as FW
import torch


class FSRCNN_net(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FSRCNN_net, self).__init__()
        
        self.input_channels = input_channels = args.input_channel
        self.upscale = upscale = args.scale[0] if type(args.scale)==tuple else args.scale    # [HxW] or [HxH]
        self.nf = nf = args.n_feats
        self.s = s = 12
        self.m = m = args.n_resblocks
        
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=nf, kernel_size=5, stride=1, padding=2),
            nn.PReLU())

        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=nf, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=nf, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.body_conv = torch.nn.ModuleList(self.layers)

        # Deconvolution
        self.tail_conv = nn.ConvTranspose2d(in_channels=nf, out_channels=input_channels, kernel_size=9,
                                            stride=upscale, padding=3, output_padding=1)


        common.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        out = self.tail_conv(fea)
        return out
    
class EUNAF_FSRCNN(FSRCNN_net):
    def __init__(self, args, conv=common.default_conv):
        super(EUNAF_FSRCNN, self).__init__(args, conv=conv)
        self.n_estimators = min(args.n_estimators, self.m//2)
        self.predictors = self.init_intermediate_out(self.n_estimators-1, conv, out_channels=args.input_channel)
        self.estimators = self.init_intermediate_out(self.n_estimators, conv, out_channels=args.input_channel, last_act=False)

        common.initialize_weights([self.predictors, self.estimators], 0.1)
        
    def get_n_estimators(self):
        return self.n_estimators
        
    def init_intermediate_out(self, num_blocks, conv, out_channels=1, is_estimator=False, last_act=False):
        
        interm_predictors = nn.ModuleList()
        
        for _ in range(num_blocks):
            if is_estimator:
                m_tail = [
                    conv(12, 12, 3), nn.LeakyReLU(0.1),
                    conv(12, out_channels*self.upscale*self.upscale, 3),
                    nn.PixelShuffle(self.upscale), nn.LeakyReLU(0.1),
                    conv(out_channels, out_channels, 1)
                ]
            else:
                m_tail = [
                    nn.ConvTranspose2d(in_channels=self.s, out_channels=self.input_channels, kernel_size=9,
                                            stride=self.upscale, padding=3, output_padding=1)
                ]
            common.initialize_weights(m_tail, 0.1)
            if last_act: m_tail.append(nn.ELU())
            interm_predictors.append(nn.Sequential(*m_tail))
            
        return interm_predictors
    
    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if 'predictors' not in n and 'estimators' not in n:
                p.requires_grad = False
            else:
                print(n, end="; ")

    def forward(self, x):
        fea = self.head_conv(x)
        outs, masks = list(), list()
        for i, b in enumerate(self.body_conv):
            fea = b(fea)
            
            if i==len(self.body_conv)-3:    # before increase feature
                tmp_out = self.predictors[0](fea)
                outs.append(tmp_out)
                
                for j in range(self.n_estimators):
                    m = self.estimators[j](fea)
                    masks.append(m)
        
        out = self.tail_conv(fea)
        outs.append(out)
        return outs, masks
    
    def eunaf_forward(self, x):
        fea = self.head_conv(x)
        outs, masks = list(), list()
        for i, b in enumerate(self.body_conv):
            fea = b(fea)
            
            if i==len(self.body_conv)-3:    # before increase feature
                tmp_out = self.predictors[0](fea)
                outs.append(tmp_out)
                
                for j in range(self.n_estimators):
                    m = self.estimators[j](fea)
                    masks.append(m)
        
        out = self.tail_conv(fea)
        outs.append(out)
        return outs, masks