# source 60.3
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/imagenet_c/source.yaml
# tent 56.0
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/imagenet_c/tent.yaml
# cotta 55.5
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/imagenet_c/cotta.yaml DETERMINISM True
# rmt 48.2 
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/imagenet_c/rmt.yaml DETERMINISM True 
# obao 48.3
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/imagenet_c/obao.yaml DETERMINISM True
# cmae 51.6 
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/imagenet_c/cmae.yaml DETERMINISM True MODEL.ARCH VIT_B_224_MAE 
# DDA 57.4
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/imagenet_c/dda.yaml DETERMINISM True
# SDA 55.8
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg cfgs/vit/imagenet_c/sda.yaml DETERMINISM True
# DPCore 47.6
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/imagenet_c/dpcore.yaml DETERMINISM True 
# Ours
CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/imagenet_c/das.yaml DETERMINISM True
