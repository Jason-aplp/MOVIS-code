python main.py \
    -t \
    --base configs/3d_mix.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 1 \
    --finetune_from sd-image-conditioned-v2.ckpt