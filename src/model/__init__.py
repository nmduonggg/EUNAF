from .EDSR import EUNAF_EDSR
from .RCAN import EUNAF_RCAN
from .SRResNet import EUNAF_MSRResNet
def config(args):
    arch = args.core.split("-")
    name = arch[0]
    if name=='EUNAF_EDSR':
        return EUNAF_EDSR(args)
    elif name=='EUNAF_RCAN':
        return EUNAF_RCAN(args)
    elif name=='EUNAF_SRResNet':
        return EUNAF_MSRResNet(args)
    
    else:
        assert(0), 'No configuration found'