python train_eunaf_no_freeze_srr.py \
    --template EUNAF_RCANxN \
    --N 14 \
    --scale 2 \
    --train_stage 1 \
    --max_epochs 1000 \
    --lr 0.0001 \
    --testset_tag='Set14RGB' \
    --testset_dir='../../data/' \
    --trainset_preload 400 \
    --n_estimators 4 \
    # --weight './checkpoints/EUNAF_RCANx2_x2_nb4_nf64_ng10_st0/_best.t7' \
    # --wandb \
    # --lr 0.00
    # --max_load 1000