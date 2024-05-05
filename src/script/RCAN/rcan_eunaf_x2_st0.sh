python train_eunaf.py \
    --template EUNAF_RCANx2 \
    --N 14 \
    --scale 2 \
    --train_stage 0 \
    --n_resblocks 2 \
    --n_resgroups 4 \
    --lr 0.0001 \
    # --wandb \
    # --weight '/mnt/disk1/nmduong/FusionNet/Supernet-SR/src/checkpoints/EUNAF_EDSRx2_bl_x2_nb4_nf64_st0/_best.t7' \
    # --wandb \
    # --lr 0.00
    # --max_load 1000