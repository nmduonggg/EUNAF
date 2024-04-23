from .SuperNet_separate import SuperNet_separate
from .SuperNet_UDL import SuperNet_udl

def config(args):
    arch = args.core.split("-")
    name = arch[0]

    if name=='SuperNet_udl':
        return SuperNet_udl(args.scale, args.input_channel, args.nblocks)
    elif name=='SuperNet_separate':
        return SuperNet_separate(args.scale, args.input_channel, args.nblocks)
    else:
        assert(0), 'No configuration found'