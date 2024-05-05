from .SuperNet_separate import SuperNet_separate
from .SuperNet_UDL import SuperNet_udl
from .SuperNet_SMSR import SuperNet_SMSR
from .EDSR import EUNAF_EDSR
from .RCAN import EUNAF_RCAN

def config(args):
    arch = args.core.split("-")
    name = arch[0]

    if name=='SuperNet_udl':
        return SuperNet_udl(args.scale, args.input_channel, args.nblocks)
    elif name=='SuperNet_separate':
        return SuperNet_separate(args.scale, args.input_channel, args.nblocks)
    elif name=='SuperNet_SMSR':
        return SuperNet_SMSR(args.scale, args.input_channel, args.nblocks)
    elif name=='EUNAF_EDSR':
        return EUNAF_EDSR(args)
    elif name=='EUNAF_RCAN':
        return EUNAF_RCAN(args)
    
    else:
        assert(0), 'No configuration found'