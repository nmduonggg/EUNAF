import torch.nn.functional as F
import torch.nn as nn
import torch
from model import common


def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels==0).sum() > 0):
        gumbels = torch.rand_like(x)
        
    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)
    
    return x

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
    
    
class SMM(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1, padding=1, 
                 bias=False, reduction=16):
        super().__init__()
        # spatial mask
        self.spa_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 4, 2, 3, 1, 1),
        )

        # body
        self.body = SMB(in_channels, out_channels, kernel_size, stride, padding, bias, n_layers=2)

        # CA layer
        self.ca = CALayer(out_channels, reduction)

        self.tau = 1
        
    def _set_tau(self, tau):
        self.tau = tau
        
    def forward(self, x):
        spa_mask = self.spa_mask(x)
        
        if self.training:
            spa_mask = gumbel_softmax(spa_mask, 1, self.tau)
            out, ch_mask = self.body([x, spa_mask[:, 1:, ...]])
            out = self.ca(out) + x
        else:
            spa_mask = spa_mask.softmax(1).round()
            out, ch_mask = self.body([x, spa_mask[:, 1:, ...]])
            out = self.ca(out) + x
        return out, spa_mask[:, 1:, ...], ch_mask
        
    
class SMB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, n_layers=2):
        super().__init__()
        self.ch_mask = nn.Parameter(torch.rand(1, out_channels, n_layers, 2))
        self.tau=1
        self.n_layers = n_layers
        self.relu = nn.ReLU(True)
        
        # body
        body = []
        body.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        for _ in range(self.n_layers-1):
            body.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias))
        self.body = nn.Sequential(*body)
        
        # collect 
        self.collect = nn.Conv2d(out_channels*self.n_layers, out_channels, 1, 1, 0)
        
    def forward(self, x):
        '''
        x[0]: input feature (B, C, H, W)
        x[1]: spatial mask (B, 1, H, W)
        '''
        spa_mask = x[1] # gumbel softmax already
        out = []
        fea = x[0]
        
        if self.training:
            ch_mask = gumbel_softmax(self.ch_mask, 3, self.tau)
            for i in range(self.n_layers):
                if i==0:
                    fea = self.body[i](fea)
                    fea = fea * ch_mask[:, :, i:i+1, 1:] * spa_mask + fea * ch_mask[:, :, i:i+1, :1] # 0: dense, 1: sparse
                else:
                    fea_d = self.body[i](fea * ch_mask[:, :, i-1:i, :1])
                    fea_s = self.body[i](fea * ch_mask[:, :, i-1:i, 1:])
                    fea = fea_d * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea_d * ch_mask[:, :, i:i + 1, :1] + fea_s * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea_s * ch_mask[:, :, i:i + 1, :1] * spa_mask
                fea = self.relu(fea)
                out.append(fea)
            
            out = self.collect(torch.cat(out, 1))
        
        else:
            ch_mask = self.ch_mask.softmax(3).round()
            # ch_mask = (ch_mask[:, :, :, 1:] > ch_mask[:, :, :, :1]).float() # sparse mask
            for i in range(self.n_layers):
                if i==0:
                    fea = self.body[i](fea)
                    fea = fea * ch_mask[:, :, i:i+1, 1:] * spa_mask + fea * ch_mask[:, :, i:i+1, :1] # 0: dense, 1: sparse
                else:
                    fea_d = self.body[i](fea * ch_mask[:, :, i-1:i, :1])
                    fea_s = self.body[i](fea * ch_mask[:, :, i-1:i, 1:])
                    fea = fea_d * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea_d * ch_mask[:, :, i:i + 1, :1] + fea_s * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea_s * ch_mask[:, :, i:i + 1, :1] * spa_mask
                fea = self.relu(fea)
                out.append(fea)
            
            out = self.collect(torch.cat(out, 1))
            
        return out, ch_mask
    
class SMSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()
        self.n_resblocks = n_resblocks = args.n_resblocks
        self.n_feats = n_feats =  args.n_feats
        self.reduction = reduction = args.reduction
        self.kernel_size = kernel_size = 3
        self.scale = scale = args.scale[0] if type(args.scale)==tuple else args.scale    # [HxW] or [HxH]
        self.act = act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        
        # head
        modules_head = [conv(args.input_channel, n_feats, kernel_size)]
        
        # body
        modules_body = [
            SMM(
                n_feats, n_feats, kernel_size, reduction=reduction) for _ in range(self.n_resblocks)]
        
        # collect
        self.collect = nn.Sequential(
            nn.Conv2d(n_feats*n_resblocks, n_feats, 1, 1, 0), nn.ReLU(True), 
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )
        
        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.input_channel, kernel_size)]
        
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
        self.head = nn.Sequential(*modules_head)
        self.body = nn.ModuleList(modules_body)
        self.tail = nn.Sequential(*modules_tail)
        
        self.density = list()
        
    def reset_density(self):
        self.density = []
        
    def forward(self, x):
        x0 = x
        x = self.head(x0)
        fea = x
        
        sparsity = []
        out_fea = []
        for i in range(4):
            fea, _spa_mask, _ch_mask = self.body[i](fea)
            round_spa, round_ch = _spa_mask.round(), _ch_mask.round()
            out_fea.append(fea)
            sparsity.append((_spa_mask * _ch_mask[..., 1].view(1, -1, 1, 1) + torch.ones_like(_spa_mask) * _ch_mask[..., 0].view(1, -1, 1, 1)).float())
            self.density.append(torch.mean((round_spa * round_ch[..., 1].view(1, -1, 1, 1) + torch.ones_like(round_spa) * round_ch[..., 0].view(1, -1, 1, 1)).float()))
        out_fea = self.collect(torch.cat(out_fea, 1)) + x
        sparsity = torch.cat(sparsity, 0)
        
        x = self.tail(out_fea) + F.interpolate(x0, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        return [x, sparsity]
    
class EUNAF_SMSR(SMSR):
    def __init__(self, args, conv=common.default_conv):
        super().__init__(args, conv=conv)
        self.n_estimators = min(args.n_estimators, self.n_resblocks // 2)
        self.predictors = self.init_intermediate_out(self.n_estimators-1, conv, out_channels=args.input_channel, is_estimator=False)
        self.estimators = self.init_intermediate_out(self.n_estimators, conv, out_channels=args.input_channel, is_estimator=True, last_act=False)
        
    def get_n_estimators(self):
        return self.n_estimators
        
    def init_intermediate_out(self, num_blocks, conv, out_channels=1, is_estimator=False, last_act=False):
        
        # get the number of collect features
        start = self.n_resblocks - self.n_estimators + 1
        end = self.n_resblocks + 1
        feat_range = range(start, end)
        
        interm_predictors = nn.ModuleList()
        for i in range(num_blocks):
            collect_feat = feat_range[i]
            
            m_tail = [
                conv(self.n_feats, out_channels*self.scale*self.scale, self.kernel_size),
                nn.PixelShuffle(self.scale),
                conv(out_channels, out_channels, 1)
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
        x0 = x
        x = self.head(x0)
        fea = x
        
        # sparsity = []
        out_fea = []
        for i in range(self.n_resblocks):
            fea, _spa_mask, _ch_mask = self.body[i](fea)
            round_spa, round_ch = _spa_mask.round(), _ch_mask.round()
            out_fea.append(fea)
            # sparsity.append((_spa_mask * _ch_mask[..., 1].view(1, -1, 1, 1) + torch.ones_like(_spa_mask) * _ch_mask[..., 0].view(1, -1, 1, 1)).float())
            # self.density.append(torch.mean((round_spa * round_ch[..., 1].view(1, -1, 1, 1) + torch.ones_like(round_spa) * round_ch[..., 0].view(1, -1, 1, 1)).float()))
        out_fea = self.collect(torch.cat(out_fea, 1)) + x
        # sparsity = torch.cat(sparsity, 0)
        
        x = self.tail(out_fea) + F.interpolate(x0, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        outs = [torch.zeros_like(x) for _ in range(self.n_estimators-1)] + [x]
        masks = [torch.zeros_like(x) for _ in range(self.n_estimators)]
        
        return [outs], masks
    
    def eunaf_forward(self, x):
        x0 = x
        x = self.head(x0)
        fea = x
        
        # enauf frame work start here
        outs = list() 
        masks = list()
        
        out_fea = list()
        # sparsity = list()
        for i in range(self.n_resblocks):
            fea, _spa_mask, _ch_mask = self.body[i](fea)
            round_spa, round_ch = _spa_mask.round(), _ch_mask.round()
            out_fea.append(fea)
            # sparsity.append((_spa_mask * _ch_mask[..., 1].view(1, -1, 1, 1) + torch.ones_like(_spa_mask) * _ch_mask[..., 0].view(1, -1, 1, 1)).float())
            # self.density.append(torch.mean((round_spa * round_ch[..., 1].view(1, -1, 1, 1) + torch.ones_like(round_spa) * round_ch[..., 0].view(1, -1, 1, 1)).float()))
            
            if i==self.n_resblocks-1:
                out_fea = self.collect(torch.cat(out_fea, 1)) + x
                # sparsity = torch.cat(sparsity, 0)
                out = self.tail(out_fea) + F.interpolate(x0, scale_factor=self.scale, mode='bicubic', align_corners=False)
                outs.append(out)
                
            else:
                if i > (self.n_resblocks - self.n_estimators) - 1:
                    tmp_out = self.predictors[i - self.n_resblocks + self.n_estimators](fea)  + F.interpolate(x0, scale_factor=self.scale, mode='bicubic', align_corners=False)
                    outs.append(tmp_out)
                    
                elif i== (self.n_resblocks - self.n_estimators) - 1:  # last block before intermediate predictors
                    for j in range(self.n_estimators): 
                        mask = self.estimators[j](fea)
                        mask = self.add_mean(mask)
                        masks.append(mask)  
                    
        return outs, masks