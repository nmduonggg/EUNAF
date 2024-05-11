python test_eunaf.py \
    --template EUNAF_EDSRx4_bl \
    --testset_tag Set5RGB \
    --N 5 \
    --testset_dir ../../data/ \
    --train_stage 1 \
    --n_resblocks 16 \
    --n_estimators 4 \
    --scale 4 \
    --eval_tag psnr \
    # --visualize