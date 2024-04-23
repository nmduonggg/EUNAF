python train.py \
    --template SuperNet_separate_RGB \
    --N 100 \
    --cv_dir checkpoints/SUPERNET_SEP_x3 \
    --nblocks 4 \
    --scale 3
    # --wandb \
    # --nblocks 1 \
    # --lr 0.00
    # --max_load 1000