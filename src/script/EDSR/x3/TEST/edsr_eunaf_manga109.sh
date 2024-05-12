python test_eunaf.py \
    --template EUNAF_EDSRx3_bl \
    --testset_tag Manga109 \
    --N 109 \
    --testset_dir ../../data/Manga109_new/ \
    --train_stage 1 \
    --n_resblocks 16 \
    --n_estimators 4 \
    --scale 3 \
    --eval_tag psnr \
    # --visualize