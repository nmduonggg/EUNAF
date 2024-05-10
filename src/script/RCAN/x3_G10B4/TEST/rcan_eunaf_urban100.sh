python test_eunaf.py \
    --template EUNAF_RCANxN \
    --testset_tag Urban100 \
    --N 100 \
    --testset_dir ../../data/Urban100/ \
    --train_stage 2 \
    --n_estimators 4 \
    --scale 3 \
    --eval_tag ssim \
    --weight './checkpoints/EUNAF_RCANx2_x2_nb4_nf64_ng10_st1/_best.t7' \
    # --visualize