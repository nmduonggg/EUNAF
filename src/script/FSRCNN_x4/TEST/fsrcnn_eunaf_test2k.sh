python test_eunaf_by_patches_2block.py \
    --template EUNAF_FSRCNNxN \
    --testset_tag Test2K \
    --N 100 \
    --testset_dir ../../data/test2k/ \
    --train_stage 0 \
    --n_resblocks 16 \
    --n_estimators 4 \
    --scale 4 \
    --eval_tag ssim \
    --rgb_channel
    # --visualize