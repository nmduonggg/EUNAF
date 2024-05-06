python train_eunaf.py \
    --template EUNAF_EDSRx2_bl \
    --N 14 \
    --scale 2 \
    --train_stage 2 \
    --n_resblock 8 \
    --lr 0.00001 \
    --testset_tag='Set14RGB' \
    --testset_dir='../../data/' \
    --weight './checkpoints/EUNAF_EDSRx2_bl_x2_nb8_nf64_st1/_best.t7' \
    --trainset_preload 400 \
    # --wandb \
    # --lr 0.00
    # --max_load 1000