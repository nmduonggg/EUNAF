python test_eunaf.py \
    --template EUNAF_EDSRx4_bl \
    --testset_tag Urban100 \
    --N 100 \
    --testset_dir ../../data/Urban100/ \
    --train_stage 2 \
    --n_resblocks 16 \
    --n_estimators 4 \
    --scale 4 \
    --eval_tag ssim \
    # --visualize