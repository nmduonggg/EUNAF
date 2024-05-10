python test_eunaf.py \
    --template EUNAF_RCANxN \
    --testset_tag Set5RGB \
    --N 5 \
    --testset_dir ../../data/ \
    --train_stage 2 \
    --n_estimators 4 \
    --scale 4 \
    --eval_tag psnr \
    --weight './checkpoints/EUNAF_RCANx2_x2_nb4_nf64_ng10_st1/_best.t7' \
    # --visualize