python train_eunaf_no_freeze.py \
    --template EUNAF_FSRCNNxN \
    --N 14 \
    --scale 4 \
    --train_stage 1 \
    --max_epochs 600 \
    --lr 0.0002 \
    --testset_tag='Set14RGB' \
    --testset_dir='../../data/' \
    --n_estimators 4 \
    --trainset_preload 100 \
    --rgb_channel \
    --weight '/mnt/disk1/nmduong/FusionNet/Supernet-SR/src/checkpoints/EUNAF_EDSRx2_bl_x2_nb4_nf64_st0/_best.t7' \
    # --wandb \
    
    # --lr 0.00
    # --max_load 1000