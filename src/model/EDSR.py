from model import common

import torch
import torch.nn as nn

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        self.n_resblocks = n_resblocks = args.n_resblocks
        self.n_feats = n_feats = args.n_feats
        self.kernel_size = kernel_size = 3 
        self.input_channel = args.input_channel
        self.scale = scale = args.scale[0] if type(args.scale)==tuple else args.scale    # [HxW] or [HxH]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.input_channel, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.input_channel, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.ModuleList(m_body)   
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        print('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
                        print(f'Skip load state dict for {name}')
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
                

class EUNAF_EDSR(EDSR):
    def __init__(self, args, conv=common.default_conv):
        super(EUNAF_EDSR, self).__init__(args, conv=conv)
        self.n_estimators = min(args.n_estimators, self.n_resblocks // 2)
        self.predictors = self.init_intermediate_out(self.n_estimators-1, conv, out_channels=args.input_channel)
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
            if 'predictors' not in n and 'estimators' not in n and 'align_biases' not in n:
                p.requires_grad = False
            else:
                print(n, end="; ")
                
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        for i in range(self.n_estimators):
            res = self.body[i](x) if i==0 else self.body[i](res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)
        
        outs = [torch.zeros_like(x) for _ in range(self.n_estimators - 1)] + [x]
        masks = [torch.zeros_like(x) for _ in range(self.n_estimators)]
        
        return outs, masks
    
    def eunaf_forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        shortcut = x
        
        # enauf frame work start here
        outs = list() 
        masks = list()
            
        for i in range(self.n_resblocks):
            x = self.body[i](x) 
            
            if i==self.n_resblocks-1:
                x += shortcut
                x = self.tail(x) 
                out = self.add_mean(x)
                outs.append(out)
                
            else:
                if i > (self.n_resblocks - self.n_estimators) - 1:
                    tmp_x = (x + shortcut).clone().detach()
                    tmp_x = self.predictors[i - self.n_resblocks + self.n_estimators](tmp_x) 
                    out = self.add_mean(tmp_x) 
                    outs.append(out)
                elif i== (self.n_resblocks - self.n_estimators) - 1:  # last block before intermediate predictors
                    for j in range(self.n_estimators): 
                        mask = self.estimators[j](x.clone().detach())
                        mask = self.add_mean(mask)
                        masks.append(mask)  
                    
        return outs, masks
    
        
    
    
