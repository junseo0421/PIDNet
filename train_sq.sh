CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29505 debug.py \
 --epochs 500 \
 --lr 5.e-4 \
 --scale_range [0.75,1.25] \
 --crop_size [1024,1024] \
 --batch_size 8 \
