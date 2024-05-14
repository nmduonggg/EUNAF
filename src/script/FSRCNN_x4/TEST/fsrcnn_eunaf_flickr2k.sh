python test_eunaf_by_patches.py \
    --template EUNAF_FSRCNNxN \
    --testset_tag Flickr2K \
    --N 100 \
    --testset_dir ../../data/Flikr2K/Flickr2K/ \
    --train_stage 2 \
    --n_resblocks 16 \
    --n_estimators 4 \
    --scale 4 \
    --eval_tag ssim \
    --rgb_channel \
    # --wandb \
    # --visualize