import os
import logging
import random
import numpy as np
import time
import webdataset as wds

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from typing import Union
from conf import complete_data_dir_path, generalization_dataset_names, ds_name2pytorch_ds_name
from datasets.imagelist_dataset import ImageList, FGVCAircraft
from datasets.imagenet_subsets import create_imagenet_subset
from datasets.corruptions_datasets import create_cifarc_dataset, create_imagenetc_dataset
from datasets.imagenet_d_utils import create_symlinks_and_get_imagenet_visda_mapping
from datasets.imagenet_dict import map_dict
from augmentations.transforms_adacontrast import get_augmentation_versions, get_augmentation
from augmentations.transforms_augmix import AugMixAugmenter


logger = logging.getLogger(__name__)


def identity(x):
    return x


def get_transform(dataset_name: str, adaptation: str, preprocess: Union[transforms.Compose, None], use_clip: bool, n_views: int = 64, cfg=None, arch=None):
    """
    Get the transformation pipeline
    Note that the data normalization is done within the model
    Input:
        dataset_name: Name of the dataset
        adaptation: Name of the adaptation method
        preprocess: Input pre-processing from restored model (if available)
        use_clip: If the underlying model is based on CLIP
        n_views Number of views for test-time augmentation
    Returns:
        transforms: The data pre-processing (and augmentation)
    """
    if arch is None:
        arch = cfg.MODEL.ARCH
    if use_clip:
        if adaptation in ["tpt", "vte"]:
            base_transform = transforms.Compose([preprocess.transforms[0], preprocess.transforms[1]])
            preproc = transforms.Compose([transforms.ToTensor()])  # the input normalization is done within the model
            use_augmix = True if dataset_name in generalization_dataset_names else False
            transform = AugMixAugmenter(base_transform, preproc, dataset_name=dataset_name,
                                        n_views=n_views-1, use_augmix=use_augmix)
        else:
            transform = preprocess

    elif adaptation in ["memo", "ttaug"]:
        base_transform = transforms.Compose([preprocess.transforms[0], preprocess.transforms[1]]) if preprocess else None
        preproc = transforms.Compose([transforms.ToTensor()])
        transform = AugMixAugmenter(base_transform, preproc, dataset_name=dataset_name, n_views=n_views, use_augmix=True)

    elif adaptation == "adacontrast":
        # adacontrast requires specific transformations
        if dataset_name in ["cifar10", "cifar100", "cifar10_c", "cifar100_c"]:
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=(32, 32), crop_size=32)
        elif dataset_name in ["imagenet_c", "ccc"]:
            # note that ImageNet-C and CCC are already resized and centre cropped (to size 224)
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=(224, 224), crop_size=224)
        elif dataset_name == "domainnet126":
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=(256, 256), crop_size=224)
        else:
            resize_size = 256
            crop_size = 224
            # try to get the correct resize & crop size from the pre-process
            if isinstance(preprocess, transforms.Compose):
                for transf in preprocess.transforms:
                    if isinstance(transf, transforms.Resize):
                        resize_size = transf.size
                    elif isinstance(transf, (transforms.CenterCrop, transforms.RandomCrop, transforms.RandomResizedCrop)):
                        crop_size = transf.size

            # use classical ImageNet transformation procedure
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=resize_size, crop_size=crop_size)
    elif '384' in arch:
        transform = transforms.Compose([transforms.Resize(size=(384, 384)),
                                        transforms.ToTensor()])
    elif '224' in arch and "imagenet" not in dataset_name:
        transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor()])
    else:
        # create non-method specific transformation
        if dataset_name in ["cifar10", "cifar100"]:
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name in ["cifar10_c", "cifar100_c"]:
            transform = None
        elif dataset_name in ["imagenet_c", "ccc"]:
            # note that ImageNet-C and CCC are already resized and centre cropped (to size 224)
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name == "domainnet126":
            transform = get_augmentation(aug_type="test", res_size=(256, 256), crop_size=224)
        else:
            if preprocess:
                # set transform to the corresponding input transformation of the restored model
                transform = preprocess
            else:
                # use classical ImageNet transformation procedure
                transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()])

    return transform


def get_test_loader(setting: str, adaptation: str, dataset_name: str, preprocess: Union[transforms.Compose, None],
                    data_root_dir: str, domain_name: str, domain_names_all: list, severity: int, num_examples: int,
                    rng_seed: int, use_clip: bool, n_views: int = 64, delta_dirichlet: float = 0.,
                    batch_size: int = 128, shuffle: bool = False, workers: int = 4, cfg=None):
    """
    Create the test data loader
    Input:
        setting: Name of the considered setting
        adaptation: Name of the adaptation method
        dataset_name: Name of the dataset
        preprocess: Input pre-processing from restored model (if available)
        data_root_dir: Path of the data root directory
        domain_name: Name of the current domain
        domain_names_all: List containing all domains
        severity: Severity level in case of corrupted data
        num_examples: Number of test samples for the current domain
        rng_seed: A seed number
        use_clip: If the underlying model is based on CLIP
        n_views: Number of views for test-time augmentation
        delta_dirichlet: Parameter of the Dirichlet distribution
        batch_size: The number of samples to process in each iteration
        shuffle: Whether to shuffle the data. Will destroy pre-defined settings
        workers: Number of workers used for data loading
    Returns:
        test_loader: The test data loader
    """

    # Fix seed again to ensure that the test sequence is the same for all methods
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    data_dir = complete_data_dir_path(data_root_dir, dataset_name)
    transform = get_transform(dataset_name, adaptation, preprocess, use_clip, n_views, cfg=cfg)

    # create the test dataset
    if domain_name == "none":
        test_dataset, _ = get_source_loader(dataset_name, adaptation, preprocess,
                                            data_root_dir, batch_size, use_clip, n_views,
                                            train_split=False, percentage=cfg.SOURCE.PERCENTAGE, workers=workers, use_synthetic=cfg.SOURCE.USE_SYN)
    else:
        if dataset_name in ["cifar10_c", "cifar100_c"]:
            test_dataset = create_cifarc_dataset(dataset_name=dataset_name,
                                                severity=severity,
                                                data_dir=data_dir,
                                                corruption=domain_name,
                                                corruptions_seq=domain_names_all,
                                                transform=transform,
                                                setting=setting)
            if cfg.DATA_Paired:
                ########################## floaders concat syn ##########################
                # Syn_data1_dir = os.path.join(data_dir, 'cifar100c_images', domain_name, str(severity))
                # test_dataset = SyntheticDataset(root_dir=Syn_data1_dir, transform=transform, dataset_name=dataset_name, DATA_Paired=cfg.DATA_Paired)
                # Syn_data2_dir = os.path.join(data_dir, 'cifar100c_images_Syn', domain_name, str(severity))
                # test_dataset_syn = SyntheticDataset(root_dir=Syn_data2_dir, transform=transform, dataset_name=dataset_name, DATA_Paired=cfg.DATA_Paired)
                # test_dataset = PairedImageDataset(test_dataset, test_dataset_syn)
                ########################## floaders concat syn ##########################

                ########################## npy concat syn ##########################
                folder_name = "cifar100c_images_Syn" if dataset_name == "cifar100_c" else "cifar10c_images_Syn"
                Syn_data2_dir = os.path.join(data_dir, folder_name, domain_name, str(severity))
                model_arch = cfg.MODEL.ARCH if cfg is not None else None
                if model_arch is not None:
                    if 'vit' in model_arch or 'VIT' in model_arch or '224' in model_arch:
                        img_size = 224
                    else:
                        img_size = 32
                else:
                    img_size = 32
                transform_syn = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor()])
                test_dataset_syn = SyntheticCifarDataset(root_dir=Syn_data2_dir, transform=transform_syn)
                test_dataset = PairedCifarDataset(test_dataset, test_dataset_syn)
                ########################## npy concat syn ##########################

                

        elif dataset_name == "imagenet_c":
            test_dataset = create_imagenetc_dataset(n_examples=num_examples,
                                                    severity=severity,
                                                    data_dir=data_dir,
                                                    corruption=domain_name,
                                                    corruptions_seq=domain_names_all,
                                                    transform=transform,
                                                    setting=setting)
            if cfg.DATA_Paired:
                data_dir_syn = os.path.join(data_root_dir, "ImageNet-C-Syn")
                transform_syn = transforms.Compose([
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
                test_dataset_syn = create_imagenetc_dataset(n_examples=num_examples,
                                                    severity=severity,
                                                    data_dir=data_dir_syn,
                                                    corruption=domain_name,
                                                    corruptions_seq=domain_names_all,
                                                    transform=transform_syn,
                                                    setting=setting)
                test_dataset = PairedImageDataset(test_dataset, test_dataset_syn)

        elif dataset_name in ["imagenet_k", "imagenet_r", "imagenet_a", "imagenet_v2"]:
            test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

        elif dataset_name == "ccc":
            # url = os.path.join(dset_path, "serial_{{00000..99999}}.tar") Uncoment this to use a local copy of CCC
            # domain_name = "baseline_20_transition+speed_1000_seed_44" # choose from: baseline_<0/20/40>_transition+speed_<1000/2000/5000>_seed_<43/44/45>
            url = f'https://mlcloud.uni-tuebingen.de:7443/datasets/CCC/{domain_name}/serial_{{00000..99999}}.tar'
            test_dataset = (wds.WebDataset(url)
                    .decode("pil")
                    .to_tuple("input.jpg", "output.cls")
                    .map_tuple(transform, identity)
            )
        elif dataset_name in ["imagenet_d", "imagenet_d109", "domainnet126"]:
            # create the symlinks needed for imagenet-d variants
            if dataset_name in ["imagenet_d", "imagenet_d109"]:
                for dom_name in domain_names_all:
                    if not os.path.exists(os.path.join(data_dir, dom_name)):
                        logger.info(f"Creating symbolical links for ImageNet-D {dom_name}...")
                        domainnet_dir = os.path.join(complete_data_dir_path(data_root_dir, "domainnet126"), dom_name)
                        create_symlinks_and_get_imagenet_visda_mapping(domainnet_dir, map_dict)

            # prepare a list containing all paths of the image-label-files
            if "mixed_domains" in setting:
                data_files = [os.path.join("datasets", f"{dataset_name}_lists", dom_name + "_list.txt") for dom_name in domain_names_all]
            else:
                data_files = [os.path.join("datasets", f"{dataset_name}_lists", domain_name + "_list.txt")]

            test_dataset = ImageList(image_root=data_dir,
                                     label_files=data_files,
                                     transform=transform)

        elif dataset_name in generalization_dataset_names:
            if not os.path.exists(data_dir):
                # create the corresponding torchvision dataset name
                ds_name = ds_name2pytorch_ds_name(dataset_name)
                # use torchvision to download the data
                eval(f"torchvision.datasets.{ds_name}")(root=data_root_dir, download=True)

            if dataset_name == "fgvc_aircraft":
                test_dataset = FGVCAircraft(image_root=data_dir, transform=transform, split="test")
            else:
                data_list_paths = [os.path.join("datasets", f"other_lists", f"split_zhou_{dataset_name}.json")]
                test_dataset = ImageList(image_root=data_dir, label_files=data_list_paths, transform=transform, split="test")

        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported!")

    try:
        # shuffle the test sequence; deterministic behavior for a fixed random seed
        random.shuffle(test_dataset.samples)

        # randomly subsample the dataset if num_examples is specified
        if num_examples != -1:
            num_samples_orig = len(test_dataset)
            # logger.info(f"Changing the number of test samples from {num_samples_orig} to {num_examples}...")
            test_dataset.samples = random.sample(test_dataset.samples, k=min(num_examples, num_samples_orig))

        # prepare samples with respect to the considered setting
        if "mixed_domains" in setting:
            logger.info(f"Successfully mixed the file paths of the following domains: {domain_names_all}")

        if "correlated" in setting:
            # sort the file paths by label
            if delta_dirichlet > 0.:
                logger.info(f"Using Dirichlet distribution with delta={delta_dirichlet} to temporally correlated samples by class labels...")
                test_dataset.samples = sort_by_dirichlet(delta_dirichlet, samples=test_dataset.samples)
            else:
                # sort the class labels by ascending order
                logger.info(f"Sorting the file paths by class labels...")
                test_dataset.samples.sort(key=lambda x: x[1])
    except AttributeError:
        logger.warning("Attribute 'samples' is missing. Continuing without shuffling, sorting or subsampling the files...")

    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)


def get_source_loader(dataset_name: str, adaptation: str, preprocess: Union[transforms.Compose, None],
                      data_root_dir: str, batch_size: int, use_clip: bool = False, n_views: int = 64,
                      train_split: bool = True, ckpt_path: str = None, num_samples: int = -1,
                      percentage: float = 1.0, workers: int = 4, use_synthetic=False, model_arch=None, arch=None):
    """
    Create the source data loader
    Input:
        dataset_name: Name of the dataset
        adaptation: Name of the adaptation method
        preprocess: Input pre-processing from restored model (if available)
        data_root_dir: Path of the data root directory
        batch_size: The number of samples to process in each iteration
        use_clip: If the underlying model is based on CLIP
        n_views: Number of views for test-time augmentation
        train_split: Whether to use the training or validation split
        ckpt_path: Path to a checkpoint which determines the source domain for DomainNet-126
        num_samples: Number of source samples used during training
        percentage: (0, 1] Percentage of source samples used during training
        workers: Number of workers used for data loading
    Returns:
        source_dataset: The source dataset
        source_loader: The source data loader
    """

    # create the correct source dataset name
    src_dataset_name = dataset_name.split("_")[0] if dataset_name != "ccc" else "imagenet"
    if use_synthetic:
        # 检查路径是否以 "Syn_datasets" 结尾
        if not data_root_dir.endswith("Syn_datasets"):
            data_root_dir = os.path.join(data_root_dir, "Syn_datasets")
    # complete the data root path to the full dataset path
    data_dir = complete_data_dir_path(data_root_dir, dataset_name=src_dataset_name, is_synthetic=use_synthetic)

    # get the data transformation
    if use_synthetic:
        transform = None
    else:
        transform = get_transform(src_dataset_name, adaptation, preprocess, use_clip, n_views, arch=model_arch) # Compose(ToTensor()) for 100-C

    # create the source dataset
    if use_synthetic:
        source_dataset = SyntheticDataset(root_dir=data_dir, transform=transform, dataset_name=dataset_name, model_arch=model_arch)
    elif dataset_name in ["cifar10", "cifar10_c"]:
        source_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                                      train=train_split,
                                                      download=True,
                                                      transform=transform)
    elif dataset_name in ["cifar100", "cifar100_c"]:
        source_dataset = torchvision.datasets.CIFAR100(root=data_dir,
                                                       train=train_split,
                                                       download=True,
                                                       transform=transform)
        # print(source_dataset.class_to_idx)
        # {'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14, 'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19, 'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 'dinosaur': 29, 'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33, 'fox': 34, 'girl': 35, 'hamster': 36, 'house': 37, 'kangaroo': 38, 'keyboard': 39, 'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 'lobster': 45, 'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 'mouse': 50, 'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64, 'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79, 'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84, 'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91, 'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}
    elif dataset_name in ["imagenet", "imagenet_c", "imagenet_k", "ccc"]:
        split = "train" if train_split else "val"
        source_dataset = torchvision.datasets.ImageNet(root=data_dir,
                                                       split=split,
                                                       transform=transform)
    elif dataset_name in ["domainnet126"]:
        src_domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
        source_data_list = [os.path.join("datasets", f"{dataset_name}_lists", f"{src_domain}_list.txt")]
        source_dataset = ImageList(image_root=data_dir,
                                   label_files=source_data_list,
                                   transform=transform)
        logger.info(f"Loading source data from list: {source_data_list[0]}")
    elif dataset_name in ["imagenet_r", "imagenet_a", "imagenet_v2", "imagenet_d", "imagenet_d109"]:
        split = "train" if train_split else "val"
        source_dataset = create_imagenet_subset(data_dir=data_dir,
                                                test_dataset_name=dataset_name,
                                                split=split,
                                                transform=transform)
    else:
        raise ValueError("Dataset not supported.")

    if percentage < 1.0 or num_samples >= 0:    # reduce the number of source samples
        assert percentage > 0.0, "The percentage of source samples has to be in range 0.0 < percentage <= 1.0"
        if src_dataset_name in ["cifar10", "cifar100"] and not use_synthetic:
            nr_src_samples = source_dataset.data.shape[0]
            nr_reduced = min(num_samples, nr_src_samples) if num_samples > 0 else int(np.ceil(nr_src_samples * percentage))
            inds = random.sample(range(0, nr_src_samples), nr_reduced)
            source_dataset.data = source_dataset.data[inds]
            source_dataset.targets = [source_dataset.targets[k] for k in inds]
        else:
            nr_src_samples = len(source_dataset.samples)
            nr_reduced = min(num_samples, nr_src_samples) if num_samples > 0 else int(np.ceil(nr_src_samples * percentage))
            source_dataset.samples = random.sample(source_dataset.samples, nr_reduced)

        logger.info(f"Number of images in source loader: {nr_reduced}/{nr_src_samples} \t Reduction factor = {nr_reduced / nr_src_samples:.4f}")

    # create the source data loader
    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers,
                                                drop_last=False)
    logger.info(f"Number of images and batches in source loader: #img = {len(source_dataset)} #batches = {len(source_loader)}")
    return source_dataset, source_loader


def sort_by_dirichlet(delta_dirichlet: float, samples: list):
    """
    Adapted from: https://github.com/TaesikGong/NOTE/blob/main/learner/dnn.py
    Sort classes according to a dirichlet distribution
    Input:
        delta_dirichlet: Parameter of the distribution
        samples: List containing all data sample pairs (file_path, class_label)
    Returns:
        samples_sorted: List containing the temporally correlated samples
    """

    N = len(samples)
    samples_sorted = []
    class_labels = np.array([val[1] for val in samples])
    num_classes = int(np.max(class_labels) + 1)
    dirichlet_numchunks = num_classes

    time_start = time.time()
    time_duration = 120  # seconds until program terminates if no solution was found

    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
    min_size = -1
    min_size_thresh = 10
    while min_size < min_size_thresh:  # prevent any chunk having too less data
        idx_batch = [[] for _ in range(dirichlet_numchunks)]
        idx_batch_cls = [[] for _ in range(dirichlet_numchunks)]  # contains data per each class
        for k in range(num_classes):
            idx_k = np.where(class_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(delta_dirichlet, dirichlet_numchunks))

            # balance
            proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

            # store class-wise data
            for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                idx_j.append(idx)

        # exit loop if no solution was found after a certain while
        if time.time() > time_start + time_duration:
            raise ValueError(f"Could not correlated sequence using dirichlet value '{delta_dirichlet}'. Try other value!")

    sequence_stats = []

    # create temporally correlated sequence
    for chunk in idx_batch_cls:
        cls_seq = list(range(num_classes))
        np.random.shuffle(cls_seq)
        for cls in cls_seq:
            idx = chunk[cls]
            samples_sorted.extend([samples[i] for i in idx])
            sequence_stats.extend(list(np.repeat(cls, len(idx))))

    return samples_sorted


class SyntheticDataset(torchvision.datasets.ImageFolder):
    """
    Generic dataset class for synthetic image datasets that directly extends ImageFolder.
    Can optionally use CSV metadata for additional information.
    
    Args:
        root_dir (str): Directory containing the dataset organized in class folders
        csv_file (str, optional): Path to CSV file with additional metadata
        transform (callable, optional): Transform to apply to images
        dataset_name (str): Name of the dataset ('cifar10', 'cifar100', etc.)
    """
    # FIXME 传入arch
    def __init__(self, root_dir, csv_file=None, transform=None, dataset_name="cifar100", model_arch=None, DATA_Paired=False):
        self.csv_file = csv_file
        self.dataset_name = dataset_name.lower()
        self.DATA_Paired = DATA_Paired

        if model_arch is not None:
            if 'vit' in model_arch or 'VIT' in model_arch:
                img_size = 384 if '384' in model_arch else 224
            else:
                img_size = 32
        else:
            img_size = 32
            
        # Set default transforms based on dataset
        if transform is None:
            if "cifar" in self.dataset_name:
                # CIFAR images are 32x32
                # transform = transforms.Compose([
                #     transforms.Resize((40, 40)),
                #     transforms.RandomCrop(32, padding=4),
                #     transforms.RandomHorizontalFlip(p=0.5),
                #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                #     transforms.RandomApply([transforms.RandomRotation(15)], p=0.3),
                #     transforms.ToTensor(),
                #     transforms.RandomErasing(p=0.2)
                # ])

                transform = transforms.Compose([
                    # transforms.Resize((32, 32)),
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    # transforms.Normalize(
                    #     (0.5071, 0.4867, 0.4408), 
                    #     (0.2675, 0.2565, 0.2761)
                    # ) if self.dataset_name == "cifar100" else 
                    # transforms.Normalize(
                    #     (0.4914, 0.4822, 0.4465), 
                    #     (0.2470, 0.2435, 0.2616)
                    # )  # CIFAR10 stats
                ])
            elif "imagenet" in self.dataset_name:
                # ImageNet images are typically resized to 224x224
                transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # transforms.Normalize(
                    #     mean=[0.485, 0.456, 0.406],
                    #     std=[0.229, 0.224, 0.225]
                    # )
                ])
            else:
                # Generic default transform
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        else:
            transform = transform

        # Initialize the ImageFolder parent class
        super(SyntheticDataset, self).__init__(
            root=root_dir,
            transform=transform
        )
        
        # Display dataset information
        print(f"Loaded {self.dataset_name} dataset from {root_dir}")
        print(f"Found {len(self.classes)} classes: {self.classes}")
        print(f"Total images: {len(self.samples)}")
        
        # Load additional metadata if provided
        self.has_metadata = False
        if csv_file and os.path.exists(csv_file):
            self._load_metadata()

    def _load_metadata(self):
        """Load additional metadata from CSV file"""
        self.metadata = pd.read_csv(self.csv_file)
        print(f"Loaded metadata from {self.csv_file}")
        
        # Create a mapping from image path to its metadata
        self.metadata_map = {}
        
        # Check if image paths in CSV are relative or absolute
        if 'image' in self.metadata.columns:
            for idx, row in self.metadata.iterrows():
                img_path = row['image']
                # If it's a relative path, we need to adjust
                if not os.path.isabs(img_path):
                    img_path = os.path.abspath(img_path)
                
                self.metadata_map[img_path] = idx
        
        # Verify we can map images from samples to metadata
        found = 0
        for path, _ in self.samples:
            if path in self.metadata_map:
                found += 1
        
        if found > 0:
            self.has_metadata = True
            print(f"Successfully mapped {found}/{len(self.samples)} images to metadata")
        else:
            print("Warning: Could not map any images to metadata. Check paths in CSV.")
    
    def __getitem__(self, idx):
        """
        Get item with optional metadata.
        Returns (image, label) or (image, label, text) if metadata is available
        """
        # Get image and label from parent class
        image, label = super(SyntheticDataset, self).__getitem__(idx)
        if self.DATA_Paired:
            path, _ = self.samples[idx]
            return image, label, _, path
        # If we have metadata, add it to the returned tuple
        if self.has_metadata and self.csv_file:
            path, _ = self.samples[idx]
            
            if path in self.metadata_map:
                metadata_idx = self.metadata_map[path]
                metadata_row = self.metadata.iloc[metadata_idx]
                
                # Add any metadata columns you want to return
                if 'text' in metadata_row:
                    return image, label, metadata_row['text']
        
        return image, label
    
    def get_class_names(self):
        """Returns the list of class names"""
        return self.classes

    def find_classes(self, directory):
        """重写find_classes方法，对imagenet数据集进行特殊处理    
        对所有类别文件夹均为数字命名的数据集，按数字顺序排序并重新映射类别索引。
        否则，使用父类的默认实现。
        """
        # 获取所有子文件夹
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        if "imagenet" in self.dataset_name or all(cls.isdigit() for cls in classes):
            # 按数字顺序排序文件夹
            classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))],
                           key=lambda x: int(x))
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            return classes, class_to_idx
        else:
            # 其他数据集使用默认实现 例如类名作为目录名
            return super().find_classes(directory)

class PairedImageDataset(Dataset):
    """
    一个包装类，用于将两个目录结构相同的图像数据集配对。
    它假设 dataset1[i] 和 dataset2[i] 是对应的样本。

    Args:
        real_dataset (Dataset): 包含真实图像的数据集。
                                __getitem__ 应返回 (sample, target, domain, path)。
        synth_dataset (Dataset): 包含合成图像的数据集。
                                 __getitem__ 的结构应与 real_dataset 相同。
    """
    def __init__(self, real_dataset, synth_dataset):
        super().__init__()
        self.real_dataset = real_dataset
        self.synth_dataset = synth_dataset
        self.transform = real_dataset.transform 
        
        # 确保两个数据集的大小相同
        assert len(self.real_dataset) == len(self.synth_dataset), \
            "真实数据集和合成数据集的大小必须相同！"

    def __getitem__(self, index):
        # 1. 从真实数据集中获取数据
        # 返回值: real_sample, target, domain, real_path
        real_sample, target, domain, real_path = self.real_dataset[index]
        
        # 2. 从合成数据集中获取对应的数据
        # 因为目录结构和样本顺序都相同，所以使用相同的索引即可
        # 我们只需要合成样本本身，所以用 _ 忽略其他返回值
        synth_sample, _, _, syn_path = self.synth_dataset[index]
        
        # 3. 将真实样本和合成样本组合在一起返回
        # 其他元数据（target, domain, path）可以沿用真实数据集的
        return real_sample, target, domain, real_path, synth_sample, syn_path

    def __len__(self):
        return len(self.real_dataset)
    
class PairedCifarDataset(Dataset):
    def __init__(self, cifar_dataset, syn_dataset):
        # assert len(cifar_dataset) == len(syn_dataset), "样本数量不一致"
        self.cifar_dataset = cifar_dataset
        self.syn_dataset = syn_dataset
        self.transform = cifar_dataset.transform 

    def __getitem__(self, idx):
        # 加载npy数据集的样本
        img, label, domain = self.cifar_dataset[idx]  # (img, label, shift, ...)
        # 加载syn数据集的样本
        syn_img, syn_path = self.syn_dataset[idx]  # 你可以根据SyntheticDataset的__getitem__返回内容调整
        return img, label, domain, syn_path, syn_img, syn_path

    def __len__(self):
        return len(self.syn_dataset)
    
class SyntheticCifarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        print('loading synthetic cifar dataset from:', root_dir)
        # 收集所有图片路径，按文件名排序
        self.img_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.img_paths.append(os.path.join(root, file))
        # 按文件名排序，确保0000.JPEG, 0001.JPEG, ...顺序一致
        self.img_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.length = len(self.img_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 你可以根据需要返回label，这里假设没有label
        return image, img_path