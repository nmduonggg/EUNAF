python test_eunaf.py \
    --template EUNAF_SRResNetxN \
    --testset_tag Set14RGB \
    --N 14 \
    --testset_dir ../../data/ \
    --train_stage 0 \
    --n_resblocks 16 \
    --n_estimators 4 \
    --scale 4 \
    --eval_tag ssim \
    --rgb_channel
    --weight './checkpoints/EUNAF_EDSRx2_bl_x2_nb16_nf64_st1/_best.t7' \
    # --visualize \