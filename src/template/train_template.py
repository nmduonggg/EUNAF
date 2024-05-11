import time

def set_template(args):
    if  args.template == 'EUNAF_EDSRx2_bl':
        print('[INFO] Template found (SMSR SR)')
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=16
        args.epoch_step=30
        args.val_each=1
        args.loss='L1'
        args.style='RGB'
        args.trainset_tag='DIV2K'
        args.trainset_patch_size=48
        args.trainset_dir='../../data/DIV2K/'
        args.rgb_range=1.0
        args.scale=2
        args.core='EUNAF_EDSR'
        args.n_feats=64
        args.input_channel=3
        args.res_scale=1.0
        print(vars(args))
    elif  args.template == 'EUNAF_EDSRx3_bl':
        print('[INFO] Template found (SMSR SR)')
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=16
        args.epoch_step=30
        args.val_each=1
        args.loss='L1'
        args.style='RGB'
        args.trainset_tag='DIV2K'
        args.trainset_patch_size=48
        args.trainset_dir='../../data/DIV2K/'
        args.rgb_range=1.0
        args.scale=3
        args.core='EUNAF_EDSR'
        args.n_feats=64
        args.input_channel=3
        args.res_scale=1.0
        print(vars(args))
    elif  args.template == 'EUNAF_EDSRx4_bl':
        print('[INFO] Template found (SMSR SR)')
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=16
        args.epoch_step=30
        args.val_each=1
        args.loss='L1'
        args.style='RGB'
        args.trainset_tag='DIV2K'
        args.trainset_patch_size=48
        args.trainset_dir='../../data/DIV2K/'
        args.rgb_range=1.0
        args.scale=4
        args.core='EUNAF_EDSR'
        args.n_feats=64
        args.input_channel=3
        args.res_scale=1.0
        print(vars(args))
    elif  args.template == 'EUNAF_SMSRxN':
        print('[INFO] Template found (SMSR SR)')
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=16
        args.epoch_step=30
        args.val_each=1
        args.loss='L1'
        args.style='RGB'
        args.trainset_tag='DIV2K'
        args.trainset_patch_size=48
        args.trainset_dir='../../data/DIV2K/'
        args.rgb_range=1.0
        args.core='EUNAF_SMSR'
        args.n_feats=64
        args.reduction=16
        args.input_channel=3
        args.res_scale=1.0
        print(vars(args))
    elif  args.template == 'EUNAF_RCANxN':
        print('[INFO] Template found (EUNAF RCAN SR)')
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=16
        args.epoch_step=30
        args.val_each=1
        args.max_epochs=1000
        args.loss='L1'
        args.style='RGB'
        args.trainset_tag='DIV2K'
        args.trainset_patch_size=48
        args.trainset_dir='../../data/DIV2K/'
        args.rgb_range=1.0
        args.core='EUNAF_RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 4
        args.reduction=16
        args.n_feats=64
        args.input_channel=3
        args.res_scale=1.0
        print(vars(args))
    elif  args.template == 'EUNAF_SRResNetxN':
        print('[INFO] Template found (EUNAF SRResNet SR)')
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=16
        args.epoch_step=30
        args.val_each=1
        args.loss='L1'
        args.style='RGB'
        args.trainset_tag='DIV2K'
        args.trainset_patch_size=48
        args.trainset_dir='../../data/DIV2K/'
        args.rgb_range=1.0
        args.core='EUNAF_SRResNet'
        args.n_resblocks = 16
        args.reduction=16
        args.n_feats=64
        args.input_channel=3
        args.res_scale=1.0
        print(vars(args))
    else:
        print('[ERRO] Template not found')
        assert(0)
