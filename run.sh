ulimit -n 65536

# vnice.sh [this command] single or multi-gpu? 

# train with rgb (photometric rendering)
# bash tools/dist_train.sh configs/gaussianflowocc_with_rgb.py 4
# python tools/test.py configs/gaussianflowocc_with_rgb.py work_dirs/gaussianflowocc_with_rgb/epoch_18_ema.pth --eval mIoU 

# train with 24 instead of 18 epochs
# bash tools/dist_train.sh configs/gaussianflowocc_epoch24.py 4

# train with 30 epochs instead of 18 epochs
# CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29766 bash tools/dist_train.sh configs/gaussianflowocc_epoch30.py 4

# train with no depth supervision
# bash tools/dist_train.sh configs/gaussianflowocc_no_depth.py 4
# CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29766 bash tools/dist_train.sh configs/gaussianflowocc_no_depth.py 4 --resume-from work_dirs/gaussianflowocc_no_depth/epoch_12.pth

# train with 24 instead of 18 epochs (with rgb)
# bash tools/dist_train.sh configs/gaussianflowocc_with_rgb_epoch24.py 4

# train with no temporal supervision
# CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29766 bash tools/dist_train.sh configs/gaussianflowocc_no_temporal.py 4