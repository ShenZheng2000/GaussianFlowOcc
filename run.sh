ulimit -n 65536

# vnice.sh [this command] single or multi-gpu? 

# train with rgb (photometric rendering)
# bash tools/dist_train.sh configs/gaussianflowocc_with_rgb.py 4
# python tools/test.py configs/gaussianflowocc_with_rgb.py work_dirs/gaussianflowocc_with_rgb/epoch_18_ema.pth --eval mIoU 

# # train with 24 instead of 18 epochs
# bash tools/dist_train.sh configs/gaussianflowocc_epoch24.py 4

# # train with 30 epochs instead of 18 epochs
# CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_train.sh configs/gaussianflowocc_epoch30.py 4

# TOOD: need debug this to save rendered depth and semantics!
python tools/test.py configs/gaussianflowocc.py work_dirs/gaussianflowocc/epoch_18_ema.pth \
    --eval mIoU \
    --save-occ-path occ/gaussianflowocc