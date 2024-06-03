from .EDSR import EUNAF_EDSR
from .RCAN import EUNAF_RCAN
from .SRResNet import EUNAF_MSRResNet
from .SMSR import EUNAF_SMSR
from .SRResNet import EUNAF_MSRResNet
from .FSRCNN import EUNAF_FSRCNN
from .CARN import EUNAF_CARN
from .SRResNet_1est import EUNAF_MSRResNet_1est


def config(args):
    arch = args.core.split("-")
    name = arch[0]
    if name=='EUNAF_EDSR':
        return EUNAF_EDSR(args)
    elif name=='EUNAF_RCAN':
        return EUNAF_RCAN(args)
    elif name=='EUNAF_SRResNet':
        return EUNAF_MSRResNet(args)
    elif name=='EUNAF_SMSR':
        return EUNAF_SMSR(args)
    elif name=='EUNAF_SRResNet':
        return EUNAF_MSRResNet(args)
    elif name=='EUNAF_FSRCNN':
        return EUNAF_FSRCNN(args)
    elif name=='EUNAF_CARN':
        return EUNAF_CARN(args)
    elif name=='EUNAF_SRResNet_1est':
        return EUNAF_MSRResNet_1est(args)
    
    
    else:
        assert(0), 'No configuration found'