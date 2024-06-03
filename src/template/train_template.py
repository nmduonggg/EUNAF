import time

def set_template(args):
    if  args.template == 'EUNAF_SRResNetxN':
        print('[INFO] Template found (EUNAF SRResNet SR)')
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=128
        args.batch_size_test=128
        args.epoch_step=30
        args.val_each=1
        args.val_each_step=3000
        args.loss='L1'
        args.style='RGB'
        args.trainset_tag='LQGT'
        args.testset_tag='LQGT'
        args.trainset_patch_size=32
        args.trainset_dir='../../data/DIV2K/TMP/DIV2K_scale_sub/'
        args.testset_dir='../../data/DIV2K/TMP/DIV2K_valid_HR_sub/'
        args.rgb_range=1.0
        args.core='EUNAF_SRResNet'
        args.n_resblocks = 16
        args.reduction=16
        args.n_feats=64
        args.input_channel=3
        args.res_scale=1.0
        args.phase='train'
        args.n_estimators=3
        print(vars(args))
    elif  args.template == 'EUNAF_SRResNetxN_1est':
        print('[INFO] Template found (EUNAF SRResNet SR)')
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=128
        args.batch_size_test=128
        args.epoch_step=30
        args.val_each=1
        args.val_each_step=3000
        args.loss='L1'
        args.style='RGB'
        args.trainset_tag='LQGT'
        args.testset_tag='LQGT'
        args.trainset_patch_size=32
        args.trainset_dir='../../data/DIV2K/TMP/DIV2K_scale_sub/'
        args.testset_dir='../../data/DIV2K/TMP/DIV2K_valid_HR_sub/'
        args.rgb_range=1.0
        args.core='EUNAF_SRResNet_1est'
        args.n_resblocks = 16
        args.reduction=16
        args.n_feats=64
        args.input_channel=3
        args.res_scale=1.0
        args.phase='train'
        args.n_estimators=3
        print(vars(args))
    elif  args.template == 'EUNAF_FSRCNNxN':
        print('[INFO] Template found (EUNAF FSRCNN SR)')
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=16
        args.epoch_step=30
        args.val_each=1
        args.loss='L1'
        args.style='RGB'
        args.trainset_tag='DIV2K'
        args.trainset_patch_size=32
        args.trainset_dir='../../data/DIV2K/'
        args.rgb_range=1.0
        args.core='EUNAF_FSRCNN'
        args.n_resblocks = 4
        args.reduction=16
        args.n_feats=56
        args.input_channel=3
        args.res_scale=1.0
        print(vars(args))
    elif  args.template == 'EUNAF_CARNxN':
        print('[INFO] Template found (EUNAF FSRCNN SR)')
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=16
        args.epoch_step=30
        args.val_each=1
        args.loss='L1'
        args.style='RGB'
        args.trainset_tag='DIV2K'
        args.trainset_patch_size=32
        args.trainset_dir='../../data/DIV2K/'
        args.rgb_range=1.0
        args.core='EUNAF_CARN'
        args.n_resblocks=4
        args.reduction=16
        args.n_feats=64
        args.input_channel=3
        args.res_scale=1.0
        print(vars(args))
    else:
        print('[ERRO] Template not found')
        assert(0)
