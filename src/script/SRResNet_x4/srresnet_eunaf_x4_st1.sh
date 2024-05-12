python train_eunaf.py \
    --template EUNAF_SRResNetxN \
    --N 14 \
    --scale 4 \
    --train_stage 1 \
    --max_epochs 1000 \
    --lr 0.0001 \
    --testset_tag='Set14RGB' \
    --testset_dir='../../data/' \
    --trainset_preload 400 \
    --rgb_channel \
    --n_estimators 4 \
    --weight './checkpoints/EUNAF_SRResNetxN_x2_nb16_nf64_st0/_best.t7' \
    # --wandb \
    # --lr 0.00
    # --max_load 1000