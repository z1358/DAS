# source 19.2 
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar10_c/source.yaml 
# tent 16.1 
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar10_c/tent.yaml DETERMINISM True
# cotta 20.4 
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar10_c/cotta.yaml DETERMINISM True
# cmae 12.6 
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar10_c/cmae.yaml DETERMINISM True MODEL.ARCH VIT_B_224_MAE
# obao 12.0
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg cfgs/vit/cifar10_c/obao.yaml DETERMINISM True 
# rem 11.1 
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar10_c/rem.yaml DETERMINISM True
# SDA 16.8
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/cifar10_c/sda.yaml DETERMINISM True
# Ours
CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg cfgs/vit/cifar10_c/das.yaml DETERMINISM True