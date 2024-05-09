python test_eunaf.py \
    --template EUNAF_EDSRx3_bl \
    --testset_tag Urban100 \
    --N 100 \
    --testset_dir ../../data/Urban100/ \
    --train_stage 1 \
    --n_resblocks 16 \
    --n_estimators 4 \
    --scale 3 \
    --eval_tag ssim \
    # --visualize