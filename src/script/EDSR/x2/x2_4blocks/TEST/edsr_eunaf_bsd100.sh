python test_eunaf.py \
    --template EUNAF_EDSRx2_bl \
    --testset_tag BSD100 \
    --N 100 \
    --testset_dir /mnt/disk1/nmduong/FusionNet/data/BSD100/image_SRF_2/ \
    --n_resblocks 4 \
    --train_stage 2 \
    --scale 2 \
    --eval_tag psnr \
    --visualize
    