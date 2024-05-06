python train_eunaf.py \
    --template EUNAF_EDSRx2_bl \
    --N 14 \
    --scale 2 \
    --train_stage 1 \
    --n_resblocks 4 \
    --testset_tag='Set14RGB' \
    --testset_dir='../../data/' \
    --weight './checkpoints/EUNAF_EDSRx2_bl_x2_nb4_nf64_st0/_best.t7' \
    --lr 0.0001 \
    # --wandb \
    # --lr 0.00
    # --max_load 1000