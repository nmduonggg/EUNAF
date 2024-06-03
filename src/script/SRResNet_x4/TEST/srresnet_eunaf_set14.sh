python test_eunaf_by_patches.py \
    --template EUNAF_SRResNetxN \
    --testset_tag Set14RGB \
    --N 14 \
    --testset_dir ../../data/ \
    --train_stage 1\
    --n_resblocks 16 \
    --n_estimators 3 \
    --scale 4 \
    --eval_tag ssim \
    --rgb_channel \
    # --visualize \
