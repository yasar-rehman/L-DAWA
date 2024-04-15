# import init_path
import sys
# sys.path.insert(0, ['/home/root/yasar/reproduce_papers', '/home/root/yasar/Dataset/'])
from typing import Tuple

# sys.path.remove(['/home/data1/FL_image_Reg'])
# sys.path.append(['/home/data1/Fed_SSL_Image/FL_image'])
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
# from pyvrl.apis import train_network, get_root_logger, set_random_seed, test_network 
# from pyvrl.builder import build_model, build_dataset
from viewmaker.src.models import resnet, resnet_mssl, resnet_small
from viewmaker.src.datasets import datasets
from viewmaker.src.systems import image_systems
from viewmaker.scripts import run_image
from viewmaker.src.utils import image_retrieval
from pytorch_lightning import seed_everything
# import _init_paths
import os
import re
# import mmcv
import argparse
import json
from dotmap import DotMap
import argparse
# from mmcv import Config
# from mmcv.runner import init_dist
# from mmcv.utils import collect_env



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
    if cfg.retrieval:
        train_dataset, val_dataset = datasets.get_image_datasets(cfg)
        train_loader = DataLoader(train_dataset, 
                                batch_size=cfg.optim_params.batch_size, 
                                shuffle=False, 
                                num_workers=cfg.data_loader_workers)
        val_loader = DataLoader(val_dataset, 
                                batch_size=cfg.optim_params.batch_size, 
                                shuffle=False, 
                                num_workers=cfg.data_loader_workers)
        return train_loader, train_dataset, val_loader, val_dataset

    else:
        train_dataset, dummy_var = datasets.get_image_datasets(cfg)
        train_loader = DataLoader(train_dataset, 
                                batch_size=cfg.optim_params.batch_size, 
                                shuffle=True, 
                                num_workers=cfg.data_loader_workers)
        return train_loader, train_dataset

def load_test_data(cfg):
    
    dummy_var, val_dataset = datasets.get_image_datasets(cfg)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    
    return val_loader

def train_model_cl(model,train_loader, train_dataset, val_data, cfg):
    # model code
    run_image.run(cfg, train_loader, train_dataset, val_data, model)
    # train_network(model,
    #     train_dataset,
    #     cfg,
    #     distributed=distributed,
    #     logger=logger
    # )ad_model(config)
def retrival_model_cl(model,train_loader, train_dataset, val_dataloader, val_dataset, cfg):
    aud_ret = image_retrieval.IMAGE_RETRIEVAL(model,train_loader, train_dataset, val_dataloader, val_dataset, cfg)
    # aud_ret.knn_monitor()
    aud_ret.extract_features()
    #
    aud_ret.topk_retrieval()

def test_model_cl(model, test_dataset, cfg, distributed, logger):
    run_image.run(cfg, train_loader, train_dataset, val_data, model)

def tranfer_model_cl(model, cfg, transfer=False):
    run_image.run(cfg, None, None, None, model, transfer)
    

def mkdir_or_exist(dir_name, mode=0o777):
    # copied from https://github.com/open-mmlab/mmcv/blob/7e6f4624954d62e6900c3927c7608785cb4a593f/mmcv/utils/path.py#L26
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='FedSSL_finetune')
    parser.add_argument('--rounds', 
                        default=200, 
                        type=str, 
                        help='pretrain epoch to be loaded')
    parser.add_argument('--config_path',
                        default=None,
                        type=str,
                        help='pretrain path to the configuration file')
    
    parser.add_argument('--exp_type',
                        default=None,
                        type=str,
                        help='Global checkpoint file')

    parser.add_argument('--num_gpu',
                        default=None,
                        type=int,
                        help='number of gpus')

    parser.add_argument('--client_num',
                        default=None,
                        type=int,
                        help='number of gpus')

    parser.add_argument('--backend',
                    default=None,
                    type=str,
                    help='type of backend to use')
    
    parser.add_argument('--finetune_type',
                    default='linear',
                    type=str,
                    help='type of finetuning')
    
    parser.add_argument('--lr',
                    default=0.01,
                    type=float,
                    help='learning rate')
    
    args = parser.parse_args()

    cid_plus_one = str(1)

    if args.config_path:
        with open(args.config_path) as f:
            x = json.load(f)
            config = DotMap(x)
    else:
        raise ValueError('Cannot find the configuration file')
    # path_to_config = '/home/data1/Fed_SSL_Image/FL_image/'\
    # 'viewmaker/config/simclr_fed_Linear/resnet_18/w_momentum/IID/Barlotwins/10C_10S_1LE_RAND_FedAvg_all_clients.json'
    if config.pretrain_model.checkpoint_name:
        config.pretrain_model.checkpoint_name = f"round-{args.rounds}-weights.npy.npz"
        print(f"Round {config.pretrain_model.checkpoint_name} is loaded")
    print(config)

    seed_everything(config.seed, workers=True)

    
    if args.lr != 0.01:
        config.exp_name = "Linear_" + "_".join(args.exp_type.strip().split('_')[0:]) + "_" + args.rounds + "_" + str(args.lr)
    else: 
        config.exp_name = "Linear_" + "_".join(args.exp_type.strip().split('_')[0:]) + "_" + args.rounds 

   
    config.finetune_type = args.finetune_type
    # print("This configuration file will now train the whole network", config.finetune_type)

    config.optim_params.learning_rate = args.lr

        
    # modify the parameters in the config file accroding the args file
    
    config.pretrain_model.exp_dir = os.path.join(config.pretrain_model.exp_dir, args.exp_type)
    
    config.num_gpu = args.num_gpu
    config.distributed_backend = args.backend
    # for transfer learning
    model_1 = load_model(config)


    Experiment_name_path = os.path.join(config.default_path, 
                                          config.Experiment_name)   

    mkdir_or_exist(Experiment_name_path)

    config.exp_base = Experiment_name_path
    exp_dir = os.path.join(config.exp_base, "experiments", config.exp_name)
    
    config.exp_dir = exp_dir
    config.metric_dir = os.path.join(exp_dir, "metrics")
    print(config.metric_dir)

    args.work_dir = config.metric_dir
    args.checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    mkdir_or_exist(args.checkpoint_dir)
    if config.retrieval:
               
                train_data_loader, train_dataset, val_data_loader, val_dataset = load_data(config)
                retrival_model_cl(model_1, 
                        train_data_loader, train_dataset, 
                        val_data_loader, val_dataset, 
                        config)

    else:
        tranfer_model_cl(model_1, config, transfer=True)
         



