import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from model import common

class EResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out
    
class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)
        
class UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale, multi_scale,
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale,
                 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out

class Block(nn.Module):
    def __init__(self, nf,
                 group=1):
        super(Block, self).__init__()

        self.b1 = EResidualBlock(nf, nf, group=group)
        self.c1 = BasicBlock(nf*2, nf, 1, 1, 0)
        self.c2 = BasicBlock(nf*3, nf, 1, 1, 0)
        self.c3 = BasicBlock(nf*4, nf, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        

class CARN_M(nn.Module):
    def __init__(self, args):
        super(CARN_M, self).__init__()
        self.in_nc = in_nc = args.input_channel
        self.out_nc = out_nc = args.input_channel
        self.nf = nf = args.n_feats
        self.scale = scale = args.scale[0] if type(args.scale)==tuple else args.scale
        multi_scale=False
        
        self.group = group = 4

        self.rgb_range = rgb_range = args.rgb_range # = 1.0
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.entry = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.b1 = Block(nf, group=group)
        self.b2 = Block(nf, group=group)
        self.b3 = Block(nf, group=group)
        self.c1 = BasicBlock(nf*2, nf, 1, 1, 0)
        self.c2 = BasicBlock(nf*3, nf, 1, 1, 0)
        self.c3 = BasicBlock(nf*4, nf, 1, 1, 0)
        
        self.upsample = UpsampleBlock(nf, scale=scale, 
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = nn.Conv2d(nf, out_nc, 3, 1, 1)
                
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
    
class EUNAF_CARN(CARN_M):
    def __init__(self, args, conv=common.default_conv):
        super(EUNAF_CARN, self).__init__(args)
        self.n_estimators = 4
        self.predictors = self.init_intermediate_out(self.n_estimators-1, conv, out_channels=args.input_channel)
        self.estimators = self.init_intermediate_out(self.n_estimators, conv, out_channels=args.input_channel, last_act=False)
        
    def init_intermediate_out(self, num_blocks, conv, out_channels=1, last_act=False):
        interm_predictors = nn.ModuleList()
        for _ in range(num_blocks):
            m_tail = [
                conv(self.nf, out_channels*self.scale*self.scale, 3),
                nn.PixelShuffle(self.scale), nn.ReLU(),
                conv(out_channels, out_channels, 1)
            ]
            if last_act: m_tail.append(nn.ELU())
            interm_predictors.append(nn.Sequential(*m_tail))
            
        return interm_predictors
    
    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if 'predictors' not in n and 'estimators' not in n:
                p.requires_grad = False
            if p.requires_grad:
                print(n, end=' ')
                
    def eunaf_forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x



        b1 = self.b1(o0)
        
        ee0 = self.predictors[0](b1.clone().detach())
        ee0 = self.add_mean(ee0)
        
        # get uncertainty mask
        masks = list()
        for j in range(4):
            mask = self.estimators[j](b1.clone().detach())
            masks.append(mask)
        
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        
        ee1 = self.predictors[1](b2.clone().detach())
        ee1 = self.add_mean(ee1)
        
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        
        ee2 = self.predictors[2](b3.clone().detach())
        ee2 = self.add_mean(ee2)
        
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)
        
        outs = [ee0, ee1, ee2, out]

        return outs, masks
    
    def eunaf_forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        ee0 = self.predictors[0](o0.clone().detach())
        ee0 = self.add_mean(ee0)
        
        # get uncertainty mask
        masks = list()
        for j in range(4):
            mask = self.estimators[j](o0.clone().detach())
            masks.append(mask)

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        ee1 = self.predictors[1](o1.clone().detach())
        ee1 = self.add_mean(ee1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        ee2 = self.predictors[2](o2.clone().detach())
        ee2 = self.add_mean(ee2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)
        
        outs = [ee0, ee1, ee2, out]

        return outs, masks