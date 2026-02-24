# Official Implement of CVPR 2026 paper "Dance Across Shifts: Forward-Facilitation Continual Test-Time Adaptation through Dynamic Style Bridging".
## Preparation

Please create and activate the following conda envrionment.

```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate tta_ft
# We run the code under torch version 2.1.0 with CUDA-12.1
```
We recommend referring to [this repository](https://github.com/mariodoebler/test-time-adaptation/tree/main?tab=readme-ov-file#classification) to obtain the required datasets and source domain models. After downloading, please modify `_C.DATA_DIR` in `conf.py` accordingly.

## Experiment Execution

You can use the provided configuration files to run experiments. Simply execute the following Python script with the corresponding configuration file:

```
# Tested on RTX4090
CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/vit/[cifar10_c/cifar100_c/imagenet_c]/[source/tent/cotta/cmae/dda/sda/obao/dpcore/das].yaml
```
Alternatively, you can execute the provided shell script:
```
bash scripts/IN-C.sh
```


## Citation
Please cite our work if you find it useful.
```bibtex
```

## Acknowledgement 
+ Online Test-time Adaptation code is heavily used. [official](https://github.com/mariodoebler/test-time-adaptation/tree/main) 
+ GIFT_CL [official](https://github.com/Luo-Jiaming/GIFT_CL) 
+ OBAO [official](https://github.com/z1358/OBAO)
+ Robustbench [official](https://github.com/RobustBench/robustbench) 
