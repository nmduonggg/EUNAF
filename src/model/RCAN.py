## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common

import torch.nn as nn

def make_model(args, parent=False):
    return RCAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()
        
        self.n_resgroups = n_resgroups = args.n_resgroups
        self.n_resblocks = n_resblocks = args.n_resblocks
        self.n_feats = n_feats = args.n_feats
        self.kernel_size = kernel_size = 3
        self.reduction = reduction = args.reduction 
        self.scale = scale = args.scale[0] if type(args.scale)==tuple else args.scale    # [HxW] or [HxH]
        self.act = act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        
        # define head module
        modules_head = [conv(args.input_channel, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.input_channel, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.ModuleList(modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

class EUNAF_RCAN(RCAN):
    def __init__(self, args, conv=common.default_conv):
        super(EUNAF_RCAN, self).__init__(args, conv=conv) 
        self.n_estimators = min(args.n_estimators, self.n_resgroups//2)
        self.predictors = self.init_intermediate_out(self.n_estimators - 1, conv, out_channels=args.input_channel)
        self.estimators = self.init_intermediate_out(self.n_estimators, conv, out_channels=args.input_channel, last_act=False)
        
    def get_n_estimators(self):
        return self.n_estimators
        
    def init_intermediate_out(self, num_blocks, conv, out_channels=1, last_act=False):
        
        interm_predictors = nn.ModuleList()
        for _ in range(num_blocks):
            m_tail = [
                common.Upsampler(conv, self.scale, self.n_feats, act=False),
                conv(self.n_feats, out_channels, self.kernel_size)
            ]
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
        x = self.sub_mean(x)
        x = self.head(x)

        for i in range(self.n_groups):
            res = self.body[i](x) if i==0 else self.body[i](res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)
        
        outs = [torch.zeros_like(x) for _ in range(self.n_resgroups-1)] + [x]
        masks = [torch.zeros_like(x) for _ in range(self.n_resgroups)]
        
        return outs, masks
    
    def eunaf_forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        shortcut = x
        
        # enauf frame work start here
        outs = list() 
        masks = list()
            
        for i in range(self.n_resgroups):
            x = self.body[i](x) 
            
            if i==self.n_resgroups-1:
                x += shortcut
                x = self.tail(x) 
                out = self.add_mean(x)
                outs.append(out)
                
            else:
                if i > (self.n_resgroups - self.n_estimators) - 1:
                    tmp_x = (x + shortcut).clone().detach()
                    tmp_x = self.predictors[i - self.n_resgroups + self.n_estimators](tmp_x) 
                    out = self.add_mean(tmp_x) 
                    outs.append(out)
                elif i== (self.n_resgroups - self.n_estimators) - 1:  # last block before intermediate predictors
                    for j in range(self.n_estimators): 
                        mask = self.estimators[j](x.clone().detach())
                        mask = self.add_mean(mask)
                        masks.append(mask)  
                    
        return outs, masks