# Train YOWO
/home/seokhoon/anaconda3/envs/yowo_pytorch_1.10/bin/python train.py \
        --cuda \
        -d aihub_park \
        -v yowo_nano \
        --num_workers 0 \
        --eval_epoch 1 \
        --inferred_tiling
        # --eval \
        # --fp16 \
