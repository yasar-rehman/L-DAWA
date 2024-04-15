import os
import sys
from copy import deepcopy

from  ..src.systems import image_systems
from  ..src.utils.utils import load_json
from  ..src.utils.setup import process_config
from  ..src.utils.callbacks import MoCoLRScheduler
import random, torch, numpy

import pytorch_lightning as pl
# import wandb

torch.backends.cudnn.benchmark = True

SYSTEM = {
    # 'PretrainViewMakerSystem': image_systems.PretrainViewMakerSystem,
    'PretrainExpertSystem': image_systems.PretrainExpertSystem,
    # 'TransferViewMakerSystem': image_systems.TransferViewMakerSystem,
    'TransferExpertSystem': image_systems.TransferExpertSystem,
}

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def run(cfg, train_dataloader, train_dataset, val_dataloader, model, gpu_device=None, transfer=False):
    '''Run the Lightning system. 

    Args:
        args
            args.config_path: str, filepath to the config file
        gpu_device: str or None, specifies GPU device as follows:
            None: CPU (specified as null in config)
            'cpu': CPU
            '-1': All available GPUs
            '0': GPU 0
            '4': GPU 4
            '0,3' GPUs 1 and 3
            See the following for more options: 
            https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html
    '''

    config = cfg
    if gpu_device == 'cpu' or not gpu_device:
        gpu_device = None
    

    config = process_config(cfg)
    # Only override if specified.
    if gpu_device: config.gpu_device = gpu_device
    seed_everything(config.seed)
    SystemClass = SYSTEM[config.system]
    system = SystemClass(config, train_dataset, model)

    if config.num_gpu:
        avail_gpu = config.num_gpu
    else:
        avail_gpu = 1

    if config.optim_params.scheduler:
        lr_callback = globals()[config.optim_params.scheduler](
            initial_lr=config.optim_params.learning_rate,
            max_epochs=config.num_epochs,
            use_cosine_scheduler = config.use_cosine_scheduler, 
            schedule=(
                int(0.6*config.num_epochs),
                int(0.8*config.num_epochs),
            ), 
        )
        callbacks = lr_callback
    else:
        callbacks = []

    # TODO: adjust period for saving checkpoints.
    if config.system == 'PretrainExpertSystem':
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            os.path.join(config.exp_dir, 'checkpoints'),
            save_top_k=0,
            save_last=False,
            period=1,
        )
    elif config.system == 'TransferExpertSystem':
        "Training the model expert system. Will only save last checkpoint"
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            os.path.join(config.exp_dir, 'checkpoints'),
            # save_top_k=0,
            save_last=True,
            period=1,
        )
    if config.distributed_backend == "None":
        config.distributed_backend = None

    if config.optim_params.scheduler:
        call_backs=[callbacks, ckpt_callback]
    else:
        call_backs = [ckpt_callback]

    # wandb.init(project=args.wandb_proj_name, entity='stevenlau', name=config.exp_name, config=config, sync_tensorboard=True)
    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=avail_gpu,
        auto_select_gpus=False,
         # 'ddp' is usually faster, but we use 'dp' so the negative samples 
         # for the whole batch are used for the SimCLR loss
        distributed_backend=config.distributed_backend,
        max_epochs=config.num_epochs,
        min_epochs=config.num_epochs,
        checkpoint_callback=True,
        # checkpoint_callback=ckpt_callback,
        resume_from_checkpoint=config.ckpt or config.continue_from_checkpoint,
        profiler=config.profiler,
        precision=config.optim_params.precision or 32,
        callbacks= call_backs, 
        val_check_interval=config.val_check_interval or 1.0,
        limit_val_batches=config.limit_val_batches or 1.0,
        log_every_n_steps=15
    )
    if transfer:
        trainer.fit(system)
    else:
        trainer.fit(system, train_dataloader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='path to config file')
    parser.add_argument('--gpu-device', type=str, default=None)
    parser.add_argument('--profiler', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--wandb_proj_name', type=str, default="image_spn")
    args = parser.parse_args()

    # Ensure it's a string, even if from an older config
    gpu_device = str(args.gpu_device) if args.gpu_device else None
    run(args, gpu_device=gpu_device)

