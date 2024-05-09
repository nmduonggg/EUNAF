import functools
import torch.nn as nn
import torch.nn.functional as F
from model import common



def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return layers


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        common.initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, args, conv=common.default_conv):
        super(MSRResNet, self).__init__()
        
        self.in_nc = in_nc = args.input_channel
        self.out_nc = out_nc = args.input_channel 
        self.nf = nf = args.n_feats 
        self.upscale = scale = args.scale[0] if type(args.scale)==tuple else args.scale    # [HxW] or [HxH]
        self.nb = nb = args.n_resblocks
        self.n_estimators = n_estimators = args.n_estimators
        self.kernel_size = kernel_size = 3

        m_head = [conv(args.input_channel, nf, kernel_size)]
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        m_body = make_layer(basic_block, nb)

        # upsampling
        m_tail = [
            common.Upsampler(conv, scale, nf, act='lrelu'),
            conv(nf, nf, kernel_size), nn.LeakyReLU(0.1, True), 
            conv(nf, out_nc, kernel_size)
        ]

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.ModuleList(m_body)
        self.tail = nn.Sequential(*m_tail)
        
        common.initialize_weights([self.head, self.body, self.tail], 0.1)

    def forward(self, x):
        fea = self.lrelu(self.head(x))
        
        for i in enumerate(self.nb):
            out = self.body[i](fea) if i == 0 else self.body[i](out) 

        out = self.tail(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out
    
class EUNAF_MSRResNet(MSRResNet):
    def __init__(self, args, conv=common.default_conv):
        super(EUNAF_MSRResNet, self).__init__(args, conv=conv) 
        self.n_estimators = min(args.n_estimators, self.nb // 2)
        self.predictors = self.init_intermediate_out(self.n_estimators-1, conv, out_channels=args.input_channel)
        self.estimators = self.init_intermediate_out(self.n_estimators, conv, out_channels=args.input_channel, last_act=False)
            
    def get_n_estimators(self):
        return self.n_estimators
        
    def init_intermediate_out(self, num_blocks, conv, out_channels=1, last_act=False):
        
        interm_predictors = nn.ModuleList()
        
        for _ in range(num_blocks):
            m_tail = [
                common.Upsampler(conv, self.upscale, self.nf, act='lrelu'),
                conv(self.nf, self.out_nc, self.kernel_size),
            ]
            if last_act: m_tail.append(nn.ELU())
            interm_predictors.append(nn.Sequential(*m_tail))
            
        return interm_predictors
    
    def forward(self, x):
        fea = self.lrelu(self.head(x))
        
        for i in range(self.nb):
            out = self.body[i](fea) if i == 0 else self.body[i](out) 

        out = self.tail(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        
        outs = [torch.zeros_like(out) for _ in range(self.n_estimators-1)] + [out]
        masks = [torch.zeros_like(out) for _ in range(self.n_estimators)]
        
        return outs, masks
    
    def eunaf_forward(self, x):
        fea = self.lrelu(self.head(x))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False).clone()
        
        outs = list() 
        masks = list()
        
        for i in range(self.nb):
            fea = self.body[i](fea) 
            
            if i==self.nb - 1:
                out = self.tail(fea)
                out += base
                outs.append(out)
                
            else:
                if i > (self.nb - self.n_estimators)-1 :
                    tmp_fea = fea.clone().detach() 
                    tmp_out = self.predictors[i - self.nb + self.n_estimators](tmp_fea)
                    outs.append(tmp_out + base)
                elif i==(self.nb - self.n_estimators)-1 :
                    for j in range(self.n_estimators):
                        mask = self.estimators[j](fea.clone().detach())
                        masks.append(mask)
                        
        return outs, masks