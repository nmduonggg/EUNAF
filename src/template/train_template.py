import time

def set_template(args):
    if  args.template == 'SuperNet_separate':
        print('[INFO] Template found (Separate SR)')
        args.lr=1e-4
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=256
        args.epoch_step=100
        args.val_each=2
        args.max_epochs=1000
        args.loss='L1'
        # args.max_load=0
        args.style='Y'
        args.trainset_tag='SR291B'
        args.trainset_patch_size=21
        args.trainset_dir='/mnt/disk1/nmduong/FusionNet/data/2x/'
        args.testset_tag='Set14B'
        args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/set14_dnb/2x/'
        args.rgb_range=1.0
        args.scale=2
        args.core='SuperNet_separate'
        # args.weight='/mnt/disk1/nmduong/FusionNet/fusion-net/checkpoints/SUPERNET_KUL/SuperNet_kulnblock-1_lbda0.0_gamma0.2_den2.0/_last.t7'
        print(vars(args))
    elif  args.template == 'SuperNet_separate_RGB':
        print('[INFO] Template found (Separate SR)')
        args.lr=1e-4
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=128
        args.epoch_step=100
        args.val_each=2
        args.max_epochs=1000
        args.loss='L1'
        args.style='RGB'
        args.trainset_tag='DIV2K'
        args.trainset_patch_size=21
        args.trainset_dir='/mnt/disk1/nmduong/FusionNet/data/DIV2K/'
        args.testset_tag='DIV2K-valid'
        args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/DIV2K/'
        args.rgb_range=1.0
        args.scale=2
        args.core='SuperNet_separate'
        args.input_channel=3
        print(vars(args))
    elif  args.template == 'SuperNet_udl':
        print('[INFO] Template found (UDL-like SR)')
        args.lr=1e-4
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=256
        args.epoch_step=100
        args.val_each=2
        args.max_epochs=1000
        args.loss='L1'
        # args.max_load=0
        args.style='Y'
        args.trainset_tag='SR291B'
        args.trainset_patch_size=21
        args.trainset_dir='/mnt/disk1/nmduong/FusionNet/data/2x/'
        args.testset_tag='Set14B'
        args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/set14_dnb/2x/'
        args.rgb_range=1.0
        args.core='SuperNet_udl'
        # args.weight='/mnt/disk1/nmduong/FusionNet/fusion-net/checkpoints/SUPERNET_UDL/SuperNet_udlnblock-1_lbda0.0_gamma0.2_den1.0/_last.t7'
        print(vars(args))
    elif  args.template == 'SuperNet_udl_RGB':
        print('[INFO] Template found (UDL-like SR)')
        args.lr=1e-4
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=256
        args.epoch_step=100
        args.val_each=2
        args.max_epochs=1000
        args.loss='L1'
        # args.max_load=0
        args.style='RGB'
        args.trainset_tag='DIV2K'
        args.trainset_patch_size=21
        args.trainset_dir='../../data/DIV2K/'
        args.testset_tag='DIV2K-valid'
        args.testset_dir='../../data/DIV2K'
        args.rgb_range=1.0
        args.core='SuperNet_udl'
        args.input_channel=3 
        print(vars(args))
    elif  args.template == 'SuperNet_SMSR_RGB':
        print('[INFO] Template found (SMSR SR)')
        args.lr=1e-4
        args.lr_decay_ratio=0.5
        args.weight_decay=0
        args.batch_size=128
        args.epoch_step=100
        args.val_each=2
        args.max_epochs=1000
        args.loss='L1'
        args.style='RGB'
        args.trainset_tag='DIV2K'
        args.trainset_patch_size=21
        args.trainset_dir='/mnt/disk1/nmduong/FusionNet/data/DIV2K/'
        args.testset_tag='DIV2K-valid'
        args.testset_dir='/mnt/disk1/nmduong/FusionNet/data/DIV2K/'
        args.rgb_range=1.0
        args.scale=2
        args.core='SuperNet_SMSR'
        args.input_channel=3
        print(vars(args))
    elif  args.template == 'EUNAF_EDSRx2_bl':
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
    elif  args.template == 'EUNAF_RCANx2':
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
        args.scale=2
        args.core='EUNAF_RCAN'
        args.reduction=16
        args.n_feats=64
        args.input_channel=3
        args.res_scale=1.0
        print(vars(args))
    else:
        print('[ERRO] Template not found')
        assert(0)
