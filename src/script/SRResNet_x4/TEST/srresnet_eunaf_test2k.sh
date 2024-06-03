python test_eunaf_by_patches_1est.py \
    --template EUNAF_SRResNetxN \
    --testset_tag Test2K \
    --N 100 \
    --testset_dir ../../data/test2k/ \
    --train_stage 1 \
    --n_resblocks 16 \
    --n_estimators 3 \
    --scale 4 \
    --eval_tag ssim \
    --rgb_channel \
    # --visualize