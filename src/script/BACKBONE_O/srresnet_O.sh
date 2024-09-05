python train_backbone.py \
    --template EUNAF_SRResNetxN_O \
    --N 14 \
    --scale 4 \
    --train_stage 0 \
    --max_epochs 30 \
    --lr 0.0008 \
    --testset_tag='Set14RGB' \
    --testset_dir='../../data/' \
    --n_estimators 3 \
    --trainset_preload 200 \
    --rgb_channel \
    --weight '/mnt/disk1/nmduong/FusionNet/Supernet-SR/src/checkpoints/Backbone-O/1est/EUNAF_SRResNetxN_O_x4_nb16_nf64_st0/_best.t7' \
    # --wandb \