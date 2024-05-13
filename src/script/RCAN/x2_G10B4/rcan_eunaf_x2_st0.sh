python train_eunaf_no_freeze.py \
    --template EUNAF_RCANxN \
    --N 14 \
    --scale 2 \
    --train_stage 0 \
    --max_epochs 1000 \
    --lr 0.0001 \
    --testset_tag='Set14RGB' \
    --testset_dir='../../data/' \
    --n_estimators 4 \
    --trainset_preload 400 \
    --weight '/mnt/disk1/nmduong/FusionNet/Supernet-SR/src/checkpoints/EUNAF_EDSRx2_bl_x2_nb4_nf64_st0/_best.t7' \
    # --wandb \
    # --lr 0.00
    # --max_load 1000