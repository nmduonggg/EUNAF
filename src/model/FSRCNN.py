import torch.nn as nn
from model import common
import torch

class FSRCNN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FSRCNN, self).__init__()

        self.input_channels = input_channels = args.input_channel
        self.upscale = upscale = args.scale[0] if type(args.scale)==tuple else args.scale    # [HxW] or [HxH]
        self.m = m = args.n_resblocks
        self.n_estimators = n_estimators = args.n_estimators
        self.nf = nf = args.n_feats
        self.sf = sf = 12 # squeezed feats

        self.head = nn.Sequential(
            conv(args.input_channel, nf, 5),
            nn.PReLU(),
            conv(nf, sf, 1), nn.PReLU()
        )

        self.layers = []
        for _ in range(m):
            self.layers.append(
                conv(sf, sf, 3))
            
        self.body = nn.ModuleList(self.layers)

        # Deconvolution
        self.tail = nn.Sequential(
            nn.PReLU(),
            conv(sf, nf, 1), nn.PReLU(),
            nn.ConvTranspose2d(in_channels=nf, out_channels=input_channels, kernel_size=9, stride=upscale, padding=3, output_padding=1))

        common.initialize_weights([self.head, self.body, self.tail], 0.1)

    def forward(self, x):
        fea = self.head(x)
        for b in body:
            fea = self.b(fea)
        out = self.tail_conv(fea)
        return out
    
class EUNAF_FSRCNN(FSRCNN):
    def __init__(self, args, conv=common.default_conv):
        super(EUNAF_FSRCNN, self).__init__(args, conv=conv)
        self.n_estimators = min(args.n_estimators, self.m // 2)
        self.predictors = self.init_intermediate_out(self.n_estimators-1, conv, out_channels=args.input_channel)
        self.estimators = self.init_intermediate_out(self.n_estimators, conv, out_channels=args.input_channel, last_act=False)

        common.initialize_weights([self.predictors, self.estimators], 0.1)
        
    def get_n_estimators(self):
        return self.n_estimators
        
    def init_intermediate_out(self, num_blocks, conv, out_channels=1, last_act=False):
        interm_predictors = nn.ModuleList()
        for _ in range(num_blocks):
            m_tail = [
                nn.PReLU(),
                nn.ConvTranspose2d(in_channels=self.sf, out_channels=out_channels, kernel_size=9, stride=self.upscale, padding=3, output_padding=1)
            ]
            if last_act: m_tail.append(nn.ELU())
            interm_predictors.append(nn.Sequential(*m_tail))
            
        return interm_predictors
    
    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if 'predictors' not in n and 'estimators' not in n and 'tail' not in n:
                p.requires_grad = False
            else:
                print(n, end="; ")

    def forward(self, x):
        fea = self.head(x)
        for b in body:
            fea = self.b(fea)
        out = self.tail_conv(fea)
        
        outs = [torch.zeros_like(out) for _ in range(self.n_estimators-1)] + [out]
        masks = [torch.zeros_like(out) for _ in range(self.n_estimators-1)]
        
        return out
    
    def eunaf_forward(self, x):
        fea = self.head(x)
        outs = list()
        masks = list()
        
        for i in range(self.m):
            fea = self.body[i](fea)
            
            if i==self.m - 1:
                out = self.tail(fea)
                outs.append(out)
                
            else:
                if i > (self.m - self.n_estimators)-1:
                    tmp_fea = fea
                    tmp_out = self.predictors[i - self.m + self.n_estimators](tmp_fea)
                    outs.append(tmp_out)
                elif i==(self.m - self.n_estimators)-1:
                    for j in range(self.n_estimators):
                        mask = self.estimators[j](fea)
                        masks.append(mask)
                        
        return outs, masks