# import init_path
import sys
from collections import OrderedDict
from typing import Tuple
from argparse import Namespace
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
# from pyvrl.apis import train_network, get_root_logger, set_random_seed, test_network 
# from pyvrl.builder import build_model, build_dataset
from inventory.src.models import resnet_mssl, resnet_small
from inventory.src.datasets import datasets
from inventory.src.systems import image_systems
from inventory.scripts import run_image

# import _init_pathsl
import os
import re
# import mmcv
import argparse
import json
from dotmap import DotMap

from flwr.common import parameters_to_weights
import numpy as np



def load_model(cfg):
    
    """Load SSL model."""
   
    if cfg.model_params.resnet_small:
        print("Load Resnenet_small" )
        model = resnet_small.ResNet18(num_classes=cfg.model_params.out_dim)
    
    elif cfg.model_params.resnet_50:
        model = resnet_small.ResNet50(num_classes=cfg.model_params.out_dim)
    
    elif cfg.model_params.resnet_34:
        model = resnet_small.ResNet34(num_classes=cfg.model_params.out_dim)
    else:
        kwargs = {"low_dim": 128, "in_channel": 3, "width": 1, "type":'resnet18'}
        model = resnet_mssl.resnet18(**kwargs)
    return model

def load_data(cfg):
    """Load the data partition for a single client ID."""
    train_dataset, dummy_var = datasets.get_image_datasets(cfg)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg.optim_params.batch_size, 
                              shuffle=True, 
                              num_workers=cfg.data_loader_workers)

    # if cfg.loss_params.name == "rotnet":

    return train_loader, train_dataset

def load_test_data(cfg):
    
    dummy_var, val_dataset = datasets.get_image_datasets(cfg)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    
    return val_loader, val_dataset

def train_model_cl(model,train_loader, train_dataset, val_data, cfg):
    # model code
    run_image.run(cfg, train_loader, train_dataset, val_data, model)
   


def test_model_cl(model, test_dataset, cfg, distributed, logger):
    run_image.run(cfg, train_loader, train_dataset, val_data, model)

def tranfer_model_cl(model, train_loader, train_dataset, val_data, cfg, transfer=False):
    run_image.run(cfg, train_loader, train_dataset, val_data, model, transfer)
    


def mkdir_or_exist(dir_name, mode=0o777):
    # copied from https://github.com/open-mmlab/mmcv/blob/7e6f4624954d62e6900c3927c7608785cb4a593f/mmcv/utils/path.py#L26
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

# for debugging
if __name__ == '__main__':

    # This codie is just for testing
    cid_plus_one = str(1)

    # path to the configuration file
    path_to_config = '/home/data1/Fed_SSL_Image/FL_image/viewmaker/config/fed_Linear/resnet_18/w_momentum/my_idea/barlotwins/NIID/10C_10S_1LE_RAND_FedAvg_all_clients.json'
    
    # # path_to_config = './adc_viewmaker/config/image/transfer_expert_imagenet_simclr.json'
    with open(path_to_config) as f:
        x = json.load(f)
        config = DotMap(x)
    config.data_params.img_list = '/home/data_ssd/CIFAR-10_data/annotations_fed_non_iid_10/client_dist'+ str(1) +'.json'
    model_1 = load_model(config)
    
    
   
