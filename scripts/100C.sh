# source 44.0 
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar100_c/source.yaml
# cotta 42.7
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar100_c/cotta.yaml DETERMINISM True
# obao 33.9
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar100_c/obao.yaml DETERMINISM True
# cmae 38.2
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar100_c/cmae.yaml DETERMINISM True MODEL.ARCH VIT_B_224_MAE
# rem 36.4 
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar100_c/rem.yaml DETERMINISM True
# SDA 42.9
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar100_c/sda.yaml DETERMINISM True
# Ours
CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar100_c/das.yaml DETERMINISM True