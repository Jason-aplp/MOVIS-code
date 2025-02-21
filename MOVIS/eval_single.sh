# azimuth angle rotate counterclockwise
python eval_single.py \
    --input_image 'assets/SUNRGBD/example_0/image.png' \
    --input_depth 'assets/SUNRGBD/example_0/depth.npy' \
    --input_mask 'assets/SUNRGBD/example_0/mask.png' \
    --azimuth 80 \
    --elevation 0 \
    --output_path 'test.png'