import torch.nn.functional as F
import torch.nn as nn
import torch


def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels==0).sum() > 0):
        gumbels = torch.rand_like(x)
        
    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)
    
    return x

class SuperNet_SMSR(nn.Module):
    def __init__(self, scale, input_channel, nblocks):
        super().__init__()
        self.nblocks=nblocks
        self.scale=scale
        self.input_channel=input_channel
        
        self.smsr = SMSR(scale, input_channel, nblocks)
        self.upsamplers = nn.ModuleList([
            UpSampler(scale, 32, 3) for _ in range(nblocks-1)
        ])
        self.mask_predictor = MaskPredictor(scale, 32, nblocks)
        
    def forward(self, x): 
        head_fea, final_out, _, outs = self.smsr(x, intermediate_collect=True)
        masks = self.mask_predictor(head_fea.clone().detach()) 
        
        final_outs = list()
        for i in range(self.nblocks-1):
            final_outs.append(self.upsamplers[i](outs[i].clone().detach()))
        final_outs.append(final_out)   
        
        return [final_outs, masks]
        
        
class UpSampler(nn.Module):
    def __init__(self, scale, input_channel, out_channel):
        super().__init__()
        n_feats = input_channel
        self.upsampling = nn.Sequential(
            nn.Conv2d(n_feats, scale*scale*n_feats, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, out_channel, 1, 1, 0)
        )
    
    def forward(self, x):
        return self.upsampling(x)
    
class MaskPredictor(nn.Module):
    def __init__(self, scale, input_channel, nblocks):
        super().__init__()
        self.nblocks=nblocks
        self.heads = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, 1, 1)
        )
        self.predictors = nn.ModuleList([
            MaskBlock(32, scale) for _ in range(nblocks)
        ])
        
    def forward(self, x):
        x = self.heads(x)
        masks = [self.predictors[i](x) for i in range(self.nblocks)]
        return masks
    
class MaskBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, 1, 1, 1), nn.ReLU(),
            nn.Conv2d(channels//2, channels//4, 3, 1, 1, 1), nn.ReLU(),
        )
        self.upsampling = UpSampler(scale, channels//4, 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.upsampling(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        
        # global average pooling: feature -> point
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
    
class SMSR(nn.Module):
    def __init__(self, scale, input_channel, nblocks):
        super().__init__()
        
        print("[INFO] Use an fully dense SMSR version")
        
        n_feats =  32
        kernel_size = 3
        self.scale = scale
        self.nblocks = nblocks
        
        # head
        self.head = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, 1, 1),
            nn.Conv2d(64, 32, 1, 1, 0)
        )
        
        # body
        modules_body = [SMM(n_feats, n_feats, kernel_size) for _ in range(self.nblocks)]
        
        # collect
        self.collect = nn.Sequential(
            nn.Conv2d(32*4, 32, 1, 1, 0), nn.ReLU(True), 
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        
        # tail modules
        modules_tail = [
            nn.Conv2d(n_feats, self.scale*self.scale*n_feats, 3, 1, 1),
            nn.PixelShuffle(self.scale),
            nn.Conv2d(n_feats, input_channel, 1, 1, 0)
        ]
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.density = []
        
    def reset_density(self):
        self.density = []
        
    def forward(self, x, intermediate_collect=False):
        x0 = x
        x = self.head(x0)
        head_fea = x.clone().detach()
        fea = x
        
        sparsity = []
        out_fea = []
        outs = list()
        for i in range(4):
            fea, _spa_mask, _ch_mask = self.body[i](fea)
            if intermediate_collect:
                outs.append(fea.clone().detach())
            round_spa, round_ch = _spa_mask.round(), _ch_mask.round()
            out_fea.append(fea)
            sparsity.append((_spa_mask * _ch_mask[..., 1].view(1, -1, 1, 1) + torch.ones_like(_spa_mask) * _ch_mask[..., 0].view(1, -1, 1, 1)).float())
            self.density.append(torch.mean((round_spa * round_ch[..., 1].view(1, -1, 1, 1) + torch.ones_like(round_spa) * round_ch[..., 0].view(1, -1, 1, 1)).float()))
        out_fea = self.collect(torch.cat(out_fea, 1)) + x
        sparsity = torch.cat(sparsity, 0)
        
        x = self.tail(out_fea) + F.interpolate(x0, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        if intermediate_collect:
            return [head_fea, x, sparsity, outs]
        return [head_fea, x, sparsity]
        
        
class SMM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        # spatial mask
        self.spa_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(in_channels//4, in_channels//4, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(in_channels//4, 2, 1, 1, 0)
        )

        # body
        self.body = SMB(in_channels, out_channels, kernel_size, stride, padding, bias, n_layers=4)

        # CA layer
        self.ca = ChannelAttention(out_channels)

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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, n_layers= 4):
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