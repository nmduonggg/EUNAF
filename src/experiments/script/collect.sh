python collect_psnr_unc.py \
    --template EUNAF_SRResNetxN \
    --testset_tag Test2K \
    --N 171256 \
    --testset_dir ../../data/PATCHES_RGB/8x8/Test2K/ \
    --analyze_dir ./experiments/ANALYZE/ \
    --train_stage 2 \
    --n_resblocks 16 \
    --n_estimators 4 \
    --scale 4 \
    --eval_tag psnr \
    --rgb_channel \
    --weight './checkpoints/jointly_nofreeze/EUNAF_SRResNetxN_x4_nb16_nf64_st0'