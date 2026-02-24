import os
import torch
import logging
import numpy as np
import methods
import wandb
import time
from models.model import get_model
from utils.misc import print_memory_info, rm_substr_from_state_dict
from utils.eval_utils import get_accuracy, eval_domain_dict
from utils.registry import ADAPTATION_REGISTRY
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes, ckpt_path_to_domain_seq

logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_from_args(description)

    if cfg.USE_WANDB:
        run_name = f"{cfg.MODEL.ADAPTATION}/{cfg.SETTING}/{cfg.MODEL.ARCH}"
        # wandb init
        wandb.init(
            project="test-time-adaptation-under-dynamic-environments",
            mode="offline",
            name=run_name,
            config={
                "adaptation": cfg.MODEL.ADAPTATION,
                "setting": cfg.SETTING,
                "batch_size": cfg.TEST.BATCH_SIZE,
                "lr": cfg.OPTIM.LR,
                "optimizer": cfg.OPTIM.METHOD,
                "moment": cfg.M_TEACHER.MOMENTUM,
            },
            group=cfg.MODEL.ADAPTATION,  # 使用适应方法作为组
            tags=[cfg.SETTING, cfg.CORRUPTION.DATASET]  # 添加有用的标签
        )

    valid_settings = ["reset_each_shift",           # reset the model state after the adaptation to a domain
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                      "mixed_domains",              # consecutive test samples are likely to originate from different domains
                      "correlated",                 # sorted by class label
                      "mixed_domains_correlated",   # mixed domains + sorted by class label
                      "gradual_correlated",         # gradual domain shifts + sorted by class label
                      "reset_each_shift_correlated",
                      "continual_tta_noniid",      # non-iid continual test-time adaptation. In fact, it is correlated
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # get the base model and its corresponding input pre-processing (if available)
    base_model, model_preprocess = get_model(cfg, num_classes, device)

    if cfg.TEST.LR_DECAY_WITH_BS:
        if cfg.CORRUPTION.DATASET in ["imagenet_c", "cifar10_c", "cifar100_c"]:
            base_bs = 50
        elif cfg.CORRUPTION.DATASET in ["imagenet_d", "imagenet_d109", "ccc"]:
            base_bs = 64
        elif cfg.CORRUPTION.DATASET == "domainnet126":
            base_bs = 128
        else:
            raise ValueError(f"Unknown dataset: {cfg.CORRUPTION.DATASET}")
        cfg.defrost()
        cfg.OPTIM.LR *= cfg.TEST.BATCH_SIZE / base_bs
        logger.info("Using learning rate decay with batch size.")
        logger.info("Adjusted learning rate to: %.7f with current batch size %d", cfg.OPTIM.LR, cfg.TEST.BATCH_SIZE)

    # append the input pre-processing to the base model
    base_model.model_preprocess = model_preprocess

    # load the model checkpoint
    if cfg.MODEL.load_ckpt and len(cfg.MODEL.CKPT_PATH) > 0:
        # load from local path
        checkpoint = torch.load(cfg.MODEL.CKPT_PATH)
        checkpoint = rm_substr_from_state_dict(checkpoint['model'], 'module.')
        missing_keys, unexpected_keys = base_model.load_state_dict(checkpoint, strict=False)
        if len(missing_keys) > 0:
            logger.info(f"Missing keys in state_dict: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.info(f"Unexpected keys in state_dict: {unexpected_keys}")
        logger.info(f"Successfully loaded model checkpoint from {cfg.MODEL.CKPT_PATH}")
        cfg.defrost()
        cfg.MODEL.load_ckpt = False
        cfg.freeze()

    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET == "domainnet126":
        # extract the domain sequence for a specific checkpoint.
        domain_sequence = ckpt_path_to_domain_seq(ckpt_path=cfg.MODEL.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in ["imagenet_d", "imagenet_d109"] and not cfg.CORRUPTION.TYPE[0]:
        # domain_sequence = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        domain_sequence = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        domain_sequence = cfg.CORRUPTION.TYPE
    logger.info(f"Using {cfg.CORRUPTION.DATASET} with the following domain sequence: {domain_sequence}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    domain_seq_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else domain_sequence

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in ["cifar10_c", "cifar100_c", "imagenet_c"] and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
        model = torch.nn.DataParallel(model)

    start_time = time.time()

    domain_dict = {}
    errs_dict = {}
    global_idx = 0
    # start evaluation
    for i_dom, domain_name in enumerate(domain_seq_loop):
        if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")

        for severity in severities:
            test_data_loader = get_test_loader(
                setting=cfg.SETTING,
                adaptation=cfg.MODEL.ADAPTATION,
                dataset_name=cfg.CORRUPTION.DATASET,
                preprocess=model_preprocess,
                data_root_dir=cfg.DATA_DIR,
                domain_name=domain_name,
                domain_names_all=domain_sequence,
                severity=severity,
                num_examples=cfg.CORRUPTION.NUM_EX,
                rng_seed=cfg.RNG_SEED,
                use_clip=cfg.MODEL.USE_CLIP,
                n_views=cfg.TEST.N_AUGMENTATIONS,
                delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                cfg=cfg
            )

            if i_dom == 0:
                # Note that the input normalization is done inside of the model
                logger.info(f"Using the following data transformation:\n{test_data_loader.dataset.transform}")

            # evaluate the model
            acc_dict, domain_dict, num_samples, global_idx = get_accuracy(
                model,
                data_loader=test_data_loader,
                dataset_name=cfg.CORRUPTION.DATASET,
                domain_name=domain_name,
                setting=cfg.SETTING,
                domain_dict=domain_dict,
                print_every=cfg.PRINT_EVERY,
                device=device,
                global_idx=global_idx,
                use_wandb=cfg.USE_WANDB,
                cfg=cfg
            )

            # save the error rates for each key
            for key, acc in acc_dict.items():
                err = 1. - acc
                if key not in errs_dict:
                    errs_dict[key] = []
                errs_dict[key].append(err)

                logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][key={key}][#samples={num_samples}]: {err:.2%}")
    
    # print the mean error for each key
    for key, errors in errs_dict.items():
        logger.info(f"Key {key} mean error: {np.mean(errors):.2%}")

    if "mixed_domains" in cfg.SETTING and len(domain_dict.values()) > 0:
        # print detailed results for each domain
        eval_domain_dict(domain_dict, domain_seq=domain_sequence)

    for key, errors in errs_dict.items():
        formatted_errs = [f"{err:.2%}" for err in errors]
        error_string = " ".join(formatted_errs)
        logger.info(f"all error for key {key}: {error_string}")

    if cfg.TEST.DEBUG:
        print_memory_info()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total used time: {elapsed_time:.2f} seconds.")

    if cfg.USE_WANDB:
        wandb.finish()

if __name__ == '__main__':
    evaluate('"Evaluation.')
