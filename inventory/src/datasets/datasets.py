import torch
import random
from torchvision import transforms
from PIL import ImageFilter, Image

from .cifar10 import CIFAR10, CIFAR10Corners, CIFAR10_FED, CIFAR10_semi
from .cifar100 import CIFAR100, CIFAR100Corners, CIFAR100_FED, CIFAR100_semi

DATASET = {
    'cifar10': CIFAR10,
    'cifar100':CIFAR100,
    'cifar10_corners': CIFAR10Corners,
    'cifar10_semi':CIFAR10_semi,
    'cifar100_semi':CIFAR100_semi,
    'cifar10_Fed': CIFAR10_FED,
    'cifar100_Fed': CIFAR100_FED
}


def zscore_image(img_tensor):
    img_tensor -= img_tensor.mean([-1, -2], keepdim=True)
    img_tensor /= img_tensor.std([-1, -2], keepdim=True)
    return img_tensor

def get_image_datasets(cfg):

        # dataset_name,
        # default_augmentations='none',
        # img_list = None
        
    dataset_name = cfg.data_params.dataset
    default_augmentations=cfg.data_params.default_augmentations
    img_list = cfg.data_params.img_list
    load_transforms = TRANSFORMS[default_augmentations]
    train_transforms, test_transforms = load_transforms(
        cfg,
        dataset=dataset_name, 
    )

    if dataset_name in ['ImageNet', 'cifar10_Fed', 'cifar100_Fed', 'cifar10_semi', 'cifar100_semi']: 
        # print("################################################", dataset_name)  
        if cfg.system == "PretrainExpertSystem":
            train_dataset = DATASET[dataset_name](
                img_list= img_list,
                train=True,
                image_transforms=train_transforms
            )
            return train_dataset, None
        else:   
            train_dataset = DATASET[dataset_name](
                img_list= cfg.data_params.img_list_train,
                train=True,
                image_transforms=train_transforms
            )
            

            val_dataset = DATASET[dataset_name](
                img_list= cfg.data_params.img_list_val,
                train=False,
                image_transforms=test_transforms,
            )
            print(train_dataset, val_dataset)
            return train_dataset, val_dataset
    else:
        train_dataset = DATASET[dataset_name](
            # img_list= img_list,
            train=True,
            image_transforms=train_transforms
        )
        val_dataset = DATASET[dataset_name](
            train=False,
            image_transforms=test_transforms,
        )
        return train_dataset, val_dataset


def load_transforms_FED(cfg,dataset):
    if 'cifar' in dataset:
        train_transforms = transforms.Compose([
            # transforms.Resize(cfg.data_params.img_size),
            # transforms.CenterCrop(cfg.data_params.img_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.491, 0.482, 0.446],
            #                     std=[0.247, 0.243, 0.261]),
        ])
        test_transforms = transforms.Compose([
            # transforms.Resize(cfg.data_params.img_size),
            # transforms.CenterCrop(cfg.data_params.img_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.491, 0.482, 0.446],
            #                     std=[0.247, 0.243, 0.261]),
        ])
    else:
        return None, None

    return train_transforms, test_transforms


def load_image_transforms(cfg, dataset):
    if 'cifar' in dataset:
        train_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    elif dataset in ['mscoco'] or 'meta_' in dataset:
        train_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
    elif 'ImageNet' in dataset:
        train_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        return None, None

    return train_transforms, test_transforms


def load_default_transforms(cfg,dataset):
    if 'cifar' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(cfg.data_params.img_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                std=[0.247, 0.243, 0.261]),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                std=[0.247, 0.243, 0.261]),
        ])
    elif 'cifar10_Fed' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(cfg.data_params.img_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                std=[0.247, 0.243, 0.261]),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                std=[0.247, 0.243, 0.261]),
        ])

    else:
        return None, None
    
    return train_transforms, test_transforms

def load_rotation_transforms(cfg,dataset):
    if 'cifar' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomRotation((0,360)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                std=[0.247, 0.243, 0.261]),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                std=[0.247, 0.243, 0.261]),
        ])
    else:
        return None, None
    
    return train_transforms, test_transforms


def load_default_unnorm_transforms(cfg, dataset, **kwargs):
    if 'cifar' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.ToTensor()
    else:
        return None, None

    return train_transforms, test_transforms


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

TRANSFORMS = {
    True: load_default_transforms,
    False: load_image_transforms,
    'all': load_default_transforms,
    'all_unnorm': load_default_unnorm_transforms,
    'rotation':load_rotation_transforms,
    'none': load_image_transforms,
    'Fed': load_transforms_FED, 
}


