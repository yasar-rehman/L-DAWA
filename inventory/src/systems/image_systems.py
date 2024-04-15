import os
import random
import dotmap
import numpy as np
from dotmap import DotMap
from collections import OrderedDict
from sklearn.metrics import f1_score
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torch.optim.lr_scheduler import LambdaLR
import torch.optim.lr_scheduler as lr_scheduler

from ..datasets import datasets
from ..models import resnet_small, resnet
from ..models.transfer import LogisticRegression
from ..objectives.memory_bank import MemoryBank
from ..objectives.adversarial import  AdversarialSimCLRLoss,  AdversarialNCELoss
from ..objectives.infonce import NoiseConstrastiveEstimation
from ..objectives.simclr import SimCLRObjective
from ..objectives.Barlotwins import BarlowtinsObjective
from ..objectives.rot import ROTObjective
from ..utils import utils
from flwr.common import parameter
from ..models import viewmaker

import logging
from logging import Formatter
# from logging.handlers import FileHandler
import datetime

import torch_dct as dct
import pytorch_lightning as pl
# import wandb

###################################################################################
def makedirs(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)

###################################################################################


def create_dataloader(dataset, config, batch_size, shuffle=True, drop_last=True):
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        pin_memory=True,
        drop_last=drop_last,
        num_workers=config.data_loader_workers,
    )
    return loader


class PretrainExpertSystem(pl.LightningModule):
    '''Pytorch Lightning System for self-supervised pretraining 
    with expert image views as described in Instance Discrimination 
    or SimCLR.
    '''

    def __init__(self, config, train_data, model):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.loss_name = self.config.loss_params.name
        self.t = self.config.loss_params.t

        default_augmentations = self.config.data_params.default_augmentations
        # DotMap is the default argument when a config argument is missing
        if default_augmentations == DotMap():
           default_augmentations = 'all'
        self.train_dataset = train_data
       
        self.model = model 
        

    def forward(self, img):
        return self.model(img)

    def get_losses_for_batch(self, emb_dict, train=True):
        if self.loss_name == 'nce':
            loss_fn = NoiseConstrastiveEstimation(emb_dict['indices'], emb_dict['img_embs_1'], self.memory_bank,
                                                  k=self.config.loss_params.k,
                                                  t=self.t,
                                                  m=self.config.loss_params.m)
            loss = loss_fn.get_loss()
        elif self.loss_name == 'simclr':
            if 'img_embs_2' not in emb_dict:
                raise ValueError(f'img_embs_2 is required for SimCLR loss')
            
            loss_fn = SimCLRObjective(emb_dict['img_embs_1'], emb_dict['img_embs_2'], t=self.t)
            loss = loss_fn.get_loss()

        elif self.loss_name == 'barlotwins':
            if 'img_embs_2' not in emb_dict:
                raise ValueError(f'img_embs_2 is required for SimCLR loss')

            loss_fn = BarlowtinsObjective(emb_dict['img_embs_1'], emb_dict['img_embs_2']) 
            loss = loss_fn.get_loss()

        elif self.loss_name == 'rotnet':
            loss_fn = ROTObjective(emb_dict['img_embs_1'], emb_dict['rot_labels'])
            loss = loss_fn.get_loss()
        else:
            raise Exception(f'Objective {self.loss_name} is not supported.')

        # if train:
        #     with torch.no_grad():
        #         if self.loss_name == 'nce':
        #             new_data_memory = loss_fn.updated_new_data_memory()
        #             self.memory_bank.update(emb_dict['indices'], new_data_memory)
        #         elif 'simclr' in self.loss_name:
        #             outputs_avg = (utils.l2_normalize(emb_dict['img_embs_1'], dim=1) + 
        #                            utils.l2_normalize(emb_dict['img_embs_2'], dim=1)) / 2.
        #             indices = emb_dict['indices']
        #             self.memory_bank.update(indices, outputs_avg)
        #         else:
        #             raise Exception(f'Objective {self.loss_name} is not supported.')
        # log the loss 
        
        return loss


    def configure_optimizers(self):
        encoder_params = self.model.parameters()

        # if self.config.optim_params.adam:
        #     optim = torch.optim.AdamW(encoder_params)
        # else:
        optim = torch.optim.SGD(
            encoder_params,
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
        )
        return [optim], []

    def training_step(self, batch, batch_idx):
        emb_dict = {}
        indices, img, img2, neg_img, labels = batch
        if self.loss_name == 'nce':
            emb_dict['img_embs_1'] = self.forward(img)
        elif 'simclr' in self.loss_name:
            emb_dict['img_embs_1'] = self.forward(img)
            emb_dict['img_embs_2'] = self.forward(img2)
        elif 'barlotwins' in self.loss_name:
            emb_dict['img_embs_1'] = self.forward(img)
            emb_dict['img_embs_2'] = self.forward(img2)
        elif 'rotnet' in self.loss_name:
            out = []
            rot1 = img.transpose(2,3).flipud().rot90(1,[2,3]) # rotate by 90 degree
            rot2 = img.fliplr().flipud().rot90(2,[2,3]) # rotate by 180 degree
            rot3 = img.flipud().transpose(2,3).rot90(3,[2,3]) # rotate by 270 degree

            rot_list = [img, rot1, rot2, rot3]
            
            rotlabels = torch.from_numpy(
            np.concatenate([img.size(0) *[i] for i in range(4)])).to(device='cuda')   
            for x in rot_list:
                logits_f = self.forward(x)
                out.append(logits_f)

            emb_dict['img_embs_1'] = torch.cat(out, axis=0) 
            emb_dict['rot_labels'] = rotlabels

        emb_dict['indices'] = indices
        emb_dict['labels'] = labels

        
        # if self.global_step % 10 == 0:
        #     # Log some example views. 
        #     _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=img.device)
        #     _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=img.device)
        #     images_tf = img[:32,:,:,:]
        #     images_tf = (images_tf * _IMAGENET_STD[None, :, None, None]) + _IMAGENET_MEAN[None, :, None, None]
        #     images_tf = F.interpolate(images_tf, size=128)

        #     images_tf_2 = img2[:32,:,:,:]
        #     images_tf_2 = (images_tf_2 * _IMAGENET_STD[None, :, None, None]) + _IMAGENET_MEAN[None, :, None, None]
        #     images_tf_2 = F.interpolate(images_tf_2, size=128)

        #     grid = torchvision.utils.make_grid(images_tf)
        #     self.logger.experiment.add_image('train/input_images_1', grid, self.global_step)

        #     grid = torchvision.utils.make_grid(images_tf_2)
        #     self.logger.experiment.add_image('train/input_images_2', grid, self.global_step)
        # print(emb_dict)
        return emb_dict


    def training_step_end(self, emb_dict):
        loss = self.get_losses_for_batch(emb_dict, train=True)
        metrics = {'loss': loss, 'temperature': self.t}
        self.log_dict(metrics)
       
        # self.main_logger(metrics)
        return {'loss': loss, 'log': metrics}
    
    def training_epoch_end(self, loss):
        final_value = [i["loss"].cpu().numpy().tolist() for i in loss]
        p_loss = np.mean(final_value)
        loss_dict_save = {"loss":p_loss}
        save_log(self.config, loss_dict_save)



    # def validation_step(self, batch, batch_idx):
    #     emb_dict = {}
    #     indices, img, img2, neg_img, labels, = batch
    #     if self.loss_name == 'nce':
    #         emb_dict['img_embs_1'] = self.forward(img)
    #     elif 'simclr' in self.loss_name:
    #         emb_dict['img_embs_1'] = self.forward(img)
    #         emb_dict['img_embs_2'] = self.forward(img2)

    #     emb_dict['indices'] = indices
    #     emb_dict['labels'] = labels
    #     img_embs = emb_dict['img_embs_1']
        
    #     loss = self.get_losses_for_batch(emb_dict, train=False)

    #     num_correct, batch_size = self.get_nearest_neighbor_label(img_embs, labels)
    #     output = OrderedDict({
    #         'val_loss': loss,
    #         'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
    #         'val_num_total': torch.tensor(batch_size, dtype=float, device=self.device),
    #     })
    #     return output

def save_log(config, metrics):
    exp_base = config.exp_base
    exp_dir = os.path.join(exp_base, "experiments", config.exp_name)
    config.exp_dir = exp_dir
    config.metric_dir = os.path.join(exp_dir, "metrics/")
    makedirs([config.metric_dir])
    
    log_file_name = os.path.join(config.metric_dir,
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_log.json')
    
    log_file_name_txt = os.path.join(config.metric_dir,'loss_log.txt')
    
    with open(log_file_name, "w") as f:
        json.dump(metrics, f)
    
    with open(log_file_name_txt, 'a') as f:
            f.write('{}\n'.format(str(metrics["loss"])))

    # logging.basicConfig(filemode='a',
    #                 datefmt='%H:%M:%S',
    #                 level=logging.INFO)
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # file_handler=logging.FileHandler(os.path.join(config.metric_dir,
    #     datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '.json'),
    # )
    # file_formatter=logging.Formatter("%(message)s")
    # file_handler.setFormatter(file_formatter)
    # logger.addHandler(file_handler)
    # logger.info(metrics)



class TransferExpertSystem(pl.LightningModule):

    def __init__(self, config, train_data, encoder):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.encoder = encoder
        self.load_pretrained_model()
        # self.train_dataset = train_data
        
        # print("#########################################")
        # print(self.config)
        # print("#################", self.config.finetune_type, "########################3")


        resnet = self.config.pretrain_model.resnet_version
        if resnet == 'resnet18':
            if self.config.model_params.use_prepool:
                if self.config.model_params.resnet_small:
                    print("######Resnet small is selected########")
                    # print("#################", self.config.finetune_type, "########################3")
                    num_features = 512 * 4 * 4
                else:
                    num_features = 512 * 7 * 7
            else:
                self.num_features = 512
        elif resnet == 'resnet50':
            if self.config.model_params.use_prepool:
                num_features = 2048 * 4 * 4
            else:
                num_features = 2048 * 1 * 1
        elif resnet == 'resnet34':
            if self.config.model_params.use_prepool:
                num_features = 512 * 4 * 4
            else:
                num_features = 512 * 7 * 7
        else:
            raise Exception(f'resnet {resnet} not supported.')

        
        # self.encoder = nn.Sequential(*list(self.encoder.children()))  # keep pooling layer
        # Freeze encoder for linear evaluation.
        ###################################################################
        # unfreeze this for full network fine-tunining

        if self.config.finetune_type == 'full':
            print("#########################################")
            print("using the full network finetuning")
            print("###########################################")
        else:
            print("#########################################")
            print("using the linear finetuning")
            print("###########################################")
            print(self.config.finetune_type) 
            self.encoder = self.encoder.eval()
            utils.frozen_params(self.encoder)
        ####################################################################
        default_augmentations = self.config.data_params.default_augmentations
        # if self.config.data_params.force_default_views or default_augmentations == DotMap():
        #    default_augmentations = 'all'

        self.train_dataset, self.val_dataset = datasets.get_image_datasets(self.config)
        # self.train_dataset, dummy_var = datasets.get_image_datasets(
        #     self.config)#
            # default_augmentations=default_augmentations,
        # )
        self.num_features = num_features
        self.model = self.create_model()


    def load_pretrained_model(self):
        base_dir = self.config.pretrain_model.exp_dir
        checkpoint_name = self.config.pretrain_model.checkpoint_name
        pretrain_weight_path = os.path.join(base_dir, checkpoint_name)
        # print("the pretrained weighted path is: ", pretrain_weight_path)
        if pretrain_weight_path.endswith('.npz'):
            print('load pretrain model')
            model_weights = np.load(pretrain_weight_path, allow_pickle=True)
            # print(model_weights['arr_0'].item())
            model_weights = model_weights['arr_0'].item()
            model_weights = parameter.parameters_to_weights(model_weights)
            print(len(self.encoder.state_dict().items()), len(model_weights))

            params_dict = zip(self.encoder.state_dict().keys(), model_weights)
            state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
            self.encoder.load_state_dict(state_dict, strict=True)
        else:
            # params_dict = zip(self.encoder.state_dict().keys(), model_weights)
            # state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
            # self.encoder.load_state_dict(state_dict, strict=True)
            model_weights = torch.load(pretrain_weight_path)
            weights = [val.cpu().numpy() for _, val in model_weights['state_dict'].items()]
            params_dict = zip(self.encoder.state_dict().keys(), weights)
            state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
            self.encoder.load_state_dict(state_dict, strict=True)

    def create_model(self):
        num_class = self.train_dataset.NUM_CLASSES
        model = LogisticRegression(self.num_features, num_class)
        return model

    def forward(self, img, unused_valid=None):
        del unused_valid
        batch_size = img.size(0)
        
        if self.config.model_params.resnet_small:
            if self.config.model_params.use_prepool:
                # print("########## Using the pre-pooling layers #############")
                # print(self.encoder(img, layer=5).shape)
                embs = self.encoder(img, layer=5)
            else:
                embs = self.encoder(img, layer=6)
        elif self.config.model_params.resnet_50:
            if self.config.model_params.use_prepool:
                # print("########## Using the pre-pooling layers #############")
                # print(self.encoder(img, layer=5).shape)
                embs = self.encoder(img, layer=5)
            else:
                embs = self.encoder(img, layer=6)
        
        elif self.config.model_params.resnet_34:
            if self.config.model_params.use_prepool:
                # print("########## Using the pre-pooling layers #############")
                # print(self.encoder(img, layer=5).shape)
                embs = self.encoder(img, layer=5)
            else:
                embs = self.encoder(img, layer=6)
        else:
            self.encoder_ = nn.Sequential(*list(self.encoder.children())[:-1])  # keep pooling layer
            embs = self.encoder_(img)
        # print(embs.shape)
        # print(embs)
        # embs = self.encoder(img)
        # print(embs.view(batch_size, -1).shape)
        return self.model(embs.view(batch_size, -1))

    def get_losses_for_batch(self, batch, valid=False):
        _, img, _, _, label = batch
        logits = self.forward(img, valid)
        if self.train_dataset.MULTI_LABEL:
            return F.binary_cross_entropy(torch.sigmoid(logits).view(-1), 
                                          label.view(-1).float())
        else:
            return F.cross_entropy(logits, label)

    def get_accuracies_for_batch(self, batch, valid=False):
        _, img, _, _, label = batch
        batch_size = img.size(0)
        logits = self.forward(img, valid)
        if self.train_dataset.MULTI_LABEL:
            preds = torch.round(torch.sigmoid(logits))
            preds = preds.long().cpu()
            num_correct = torch.sum(preds.cpu() == label.cpu(), dim=0)
            num_correct = num_correct.detach().cpu().numpy()
            num_total = batch_size
            return num_correct, num_total, preds, label.cpu()
        else:
            preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            preds = preds.long().cpu()
            num_correct = torch.sum(preds == label.long().cpu()).item()
            num_total = batch_size
            return num_correct, num_total

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        with torch.no_grad():
            if self.train_dataset.MULTI_LABEL:
                num_correct, num_total, _, _ = self.get_accuracies_for_batch(batch)
                num_correct = num_correct.mean()
            else:
                num_correct, num_total = self.get_accuracies_for_batch(batch)
            metrics = {
                'train_loss': loss,
                'train_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'train_num_total': torch.tensor(num_total, dtype=float, device=self.device),
                'train_acc': torch.tensor(num_correct / float(num_total), dtype=float, device=self.device)
            }
        self.log_dict(metrics)
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, valid=True)
        if self.train_dataset.MULTI_LABEL:  # regardless if binary or not
            num_correct, num_total, val_preds, val_labels = \
                self.get_accuracies_for_batch(batch, valid=True)
            return OrderedDict({
                'val_loss': loss,
                'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'val_num_total': torch.tensor(num_total, dtype=float, device=self.device),
                'val_acc': torch.tensor(num_correct / float(num_total), dtype=float, device=self.device),
                'val_pred_labels': val_preds.float(),
                'val_true_labels': val_labels.float(),
            })
        else:
            num_correct, num_total = self.get_accuracies_for_batch(batch, valid=True)
            return OrderedDict({
                'val_loss': loss,
                'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'val_num_total': torch.tensor(num_total, dtype=float, device=self.device),
                'val_acc': torch.tensor(num_correct / float(num_total), dtype=float, device=self.device),
            })

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            try:
                metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
            except:
                pass
        
        if self.train_dataset.MULTI_LABEL:
            num_correct = torch.stack([out['val_num_correct'] for out in outputs], dim=1).sum(1)
            num_total = torch.stack([out['val_num_total'] for out in outputs]).sum()
            val_acc = num_correct / float(num_total)
            metrics['val_acc'] = val_acc.mean()
            progress_bar = {'acc': val_acc.mean()}
            num_class = self.train_dataset.NUM_CLASSES
            for c in range(num_class):
                val_acc_c = num_correct[c] / float(num_total)
                metrics[f'val_acc_feat{c}'] = val_acc_c
            val_pred_labels = torch.cat([out['val_pred_labels'] for out in outputs], dim=0).numpy()
            val_true_labels = torch.cat([out['val_true_labels'] for out in outputs], dim=0).numpy()
        
            val_f1 = 0
            for c in range(num_class):
                val_f1_c = f1_score(val_true_labels[:, c], val_pred_labels[:, c])
                metrics[f'val_f1_feat{c}'] = val_f1_c
                val_f1 = val_f1 + val_f1_c
            val_f1 = val_f1 / float(num_class)
            metrics['val_f1'] = val_f1
            progress_bar['f1'] = val_f1
            
            error_log1 = {'val_loss': metrics['val_loss'], 'val_acc': val_acc}
            error_log1 = {k:v.item() for k,v in error_log1.items()}
            save_log_val(self.config,error_log1)

            return {'val_loss': metrics['val_loss'], 
                    'log': metrics,
                    'val_acc': val_acc, 
                    'val_f1': val_f1,
                    'progress_bar': progress_bar}
        else:
            num_correct = sum([out['val_num_correct'] for out in outputs])
            num_total = sum([out['val_num_total'] for out in outputs])
            val_acc = num_correct / float(num_total)
            # val_acc = torch.mean(self.all_gather(val_acc))
            # 
                # self.log("my_reduced_metric", mean, rank_zero_only=True)
            # print(num_total)
            metrics['val_acc'] = val_acc
            progress_bar = {'acc': val_acc}
            self.log_dict(metrics)
            # if self.trainer.is_global_zero:
            print('val_acc:', val_acc.item())
            error_log1 = {'val_loss': metrics['val_loss'], 'val_acc': val_acc}
            error_log1 = {k:v.item() for k,v in error_log1.items()}
            save_log_val(self.config, error_log1)
            
            return {'val_loss': metrics['val_loss'], 
                    'log': metrics, 
                    'val_acc': val_acc,
                    'progress_bar': progress_bar}

    def configure_optimizers(self):
        if self.config.finetune_type == 'full':
            params_iterator = list(self.model.parameters()) + list(self.encoder.parameters())
        else:
            params_iterator = self.model.parameters()
        # if self.config.optim_params.optimizer_Name == 'adam':
        #     # optim = torch.optim.Adam(params_iterator)
        #     optim = torch.optim.Adam(params_iterator, 
        #         lr=self.config.optim_params.learning_rate)#, 
        #         # weight_decay=self.config.optim_params.weight_decay)

        # elif self.config.optim_params.optimizer_Name == "LBFG":
        #     optim = torch.optim.LBFGS(params_iterator, 
        #         lr=self.config.optim_params.learning_rate, 
        #         max_iter=20)
        # else:
            # https://github.com/
            # p3i0t/SimCLR-CIFAR10/blob/2f449c2e39666a5c3439859347e3f1aced67b17d/
            # simclr_lin.py#L54

            # https://github.com/AidenDurrant
            # /SimCLR-Pytorch/blob/c3c6f65ebc31afbdc2c963c0cf5ad114f39a218e/src/optimisers.py#L7

            # self.config.linear_lr =  self.config.optim_params.learning_rate #* (self.config.optim_params.batch_size/256)
        optim = torch.optim.SGD(
            params_iterator,
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
            nesterov=True
        )
            
            # lr_decay = lr_scheduler.CosineAnnealingLR(optim, 
            #     self.config.num_epochs)
            # lr_decay = lr_scheduler.MultiStepLR(optim,
            #     milestones=[40, 60, 80], gamma=0.1)
            # print(lr)
        return [optim], []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size, 
                                 shuffle=False, drop_last=False)

def save_log_val(config, metrics):
    exp_base = config.exp_base
    exp_dir = os.path.join(exp_base, "experiments", config.exp_name)
    config.exp_dir = exp_dir
    config.metric_dir = os.path.join(exp_dir, "metrics/")
    makedirs([config.metric_dir])
    log_file_name = os.path.join(config.metric_dir, 'val_result'+'_log.json')
        # datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_log.json')
    with open(log_file_name, "w") as f:
        json.dump(metrics, f)
