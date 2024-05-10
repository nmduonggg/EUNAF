import os

def set_template(args):
    if args.template == 'SuperNet_udl':
        print('[INFO] Template found (UDL-like SR)')
        args.style='Y'
        args.rgb_range=1.0
        args.core='SuperNet_udl'
        args.weight = os.path.join(args.cv_dir, '_best.t7')
    elif args.template == 'SuperNet_udl_RGB':
        print('[INFO] Template found (UDL-like SR)')
        args.style='RGB'
        args.rgb_range=1.0
        args.input_channel=3
        args.core='SuperNet_udl'
        args.weight = os.path.join(args.cv_dir, '_best.t7')
    elif  args.template == 'SuperNet_separate':
        print('[INFO] Template found (Separate SR)')
        args.style='Y'
        args.rgb_range=1.0
        args.core='SuperNet_separate'
        args.weight = os.path.join(args.cv_dir, '_best.t7')
    elif  args.template == 'SuperNet_separate_RGB':
        print('[INFO] Template found (Separate SR)')
        args.style='RGB'
        args.rgb_range=1.0
        args.input_channel=3
        args.core='SuperNet_separate'
        args.weight = os.path.join(args.cv_dir, '_best.t7')
    elif  args.template == 'EUNAF_EDSRx2_bl':
        print('[INFO] Template found (Separate SR)')
        args.style='RGB'
        args.rgb_range=1.0
        args.input_channel=3
        args.core='EUNAF_EDSR'
        args.scale=2
        args.weight = "./checkpoints/EUNAF_EDSRx2_bl_nblock1/_best.t7"
        print(vars(args))
    elif  args.template == 'EUNAF_EDSRx3_bl':
        print('[INFO] Template found (Separate SR)')
        args.style='RGB'
        args.rgb_range=1.0
        args.input_channel=3
        args.core='EUNAF_EDSR'
        args.scale=3
        args.weight = "./checkpoints/EUNAF_EDSRx2_bl_nblock1/_best.t7"
        print(vars(args))
    elif  args.template == 'EUNAF_RCANx2':
        print('[INFO] Template found (Separate SR)')
        args.style='RGB'
        args.rgb_range=1.0
        args.input_channel=3
        args.core='EUNAF_RCAN'
        args.scale=2
        args.n_resgroups = 10
        args.n_resblocks = 4
        args.reduction=16
        args.n_feats=64
        args.weight = "/mnt/disk1/nmduong/FusionNet/Supernet-SR/src/checkpoints/EUNAF_EDSRx2_bl_nblock1/_best.t7"
        print(vars(args))
    elif  args.template == 'EUNAF_RCANxN':
        print('[INFO] Template found (Separate SR)')
        args.style='RGB'
        args.rgb_range=1.0
        args.input_channel=3
        args.core='EUNAF_RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 4
        args.reduction=16
        args.n_feats=64
        args.weight = "/mnt/disk1/nmduong/FusionNet/Supernet-SR/src/checkpoints/EUNAF_EDSRx2_bl_nblock1/_best.t7"
        print(vars(args))
    elif  args.template == 'EUNAF_SRResNetxN':
        print('[INFO] Template found (SRResNet SR)')
        args.style='RGB'
        args.rgb_range=1.0
        args.input_channel=3
        args.core='EUNAF_SRResNet'
        args.n_resblocks = 16
        args.reduction=16
        args.n_feats=64
        args.weight = "/mnt/disk1/nmduong/FusionNet/Supernet-SR/src/checkpoints/EUNAF_EDSRx2_bl_nblock1/_best.t7"
        print(vars(args))
    else:
        print('[ERRO] Template not found')
        assert(0)
