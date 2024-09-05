python test_eunaf_by_patches_O.py \
    --template EUNAF_SRResNetxN_O \
    --testset_tag Test4K \
    --N 100 \
    --testset_dir ../../data/test4k/ \
    --train_stage 1 \
    --n_resblocks 4 \
    --n_estimators 3 \
    --scale 4 \
    --eval_tag psnr \
    --rgb_channel \
    --backbone_name srresnet \
    # --visualize