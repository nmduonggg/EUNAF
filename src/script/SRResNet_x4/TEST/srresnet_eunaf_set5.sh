python test_eunaf_by_patches.py \
    --template EUNAF_SRResNetxN \
    --testset_tag Set5RGB \
    --N 5 \
    --testset_dir ../../data/ \
    --train_stage 0 \
    --n_resblocks 16 \
    --n_estimators 4 \
    --scale 4 \
    --eval_tag psnr \
    --rgb_channel
    # --visualize