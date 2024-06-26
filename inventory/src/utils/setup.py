import os
import sys
import torch
import logging
import getpass
import numpy as np
from pprint import pprint
from dotmap import DotMap
from logging import Formatter
from time import strftime, localtime, time
from logging.handlers import RotatingFileHandler
import datetime
from .utils import load_json, save_json

# DEFAULT_EXP_BASE = os.path.join(os.path.dirname(__file__), '../../experiments')


def makedirs(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)





def process_config(config_path, override_dotmap=None, exp_name_suffix=None):
    #config_json = load_json(config_path)
    print(config_path)
    return _process_config(config_path, override_dotmap=override_dotmap,
                          exp_name_suffix=exp_name_suffix)


def _process_config(config_json, override_dotmap=None, exp_name_suffix=None):
    """
    Processes config file:
        1) Converts it to a DotMap
        2) Creates experiments path and required subdirs
        3) Set up logging
    """
    config = DotMap(config_json)
    if override_dotmap is not None:
        config.update(override_dotmap)

    if exp_name_suffix is not None:
        config.exp_name = f'{config.exp_name}_{exp_name_suffix}'

    print("Loaded configuration: ")
    pprint(config)

    print()
    print(" *************************************** ")
    print("      Running experiment {}".format(config.exp_name))
    print(" *************************************** ")
    print()

    exp_base = config.exp_base # or DEFAULT_EXP_BASE
        
    # Uncomment me if you wish to not overwrite
    # timestamp = strftime('%Y-%m-%d--%H_%M_%S', localtime())
    exp_dir = os.path.join(exp_base, "experiments", config.exp_name)
    config.exp_dir = exp_dir

    # create some important directories to be used for the experiment.
    config.checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    config.out_dir = os.path.join(exp_dir, "out/")
    config.log_dir = os.path.join(exp_dir, "logs/")
    config.metric_dir = os.path.join(exp_dir, "metrics/")

    # will not create if already existing
    makedirs([config.checkpoint_dir, config.out_dir, config.log_dir])

    # save config to experiment dir
    config_out = os.path.join(exp_dir, 'config.json')
    save_json(config.toDict(), config_out)

    # setup logging in the project
    setup_logging(config.log_dir)

    # logging.getLogger().info(
    #     "Configurations and directories successfully set up.")

    return config


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    # file_handler=RotatingFileHandler(
    #     os.path.join(
    #     log_dir,
    #     datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"), '.json'),
    #     maxBytes=10**6, backupCount=5
    # )

    # file_formatter=logging.Formatter(
    # "%(message)s")
    # file_handler.setFormatter(file_formatter)
    
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler(
        '{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        '{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))


    # main_logger.addHandler(file_handler)
    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)



def print_cuda_statistics():
    logger = logging.getLogger("CUDA Statistics")
    import sys
    from subprocess import call
    import torch
    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    try:
        call(["nvcc", "--version"])
    except:
        pass
    logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))


# def loss_log(cfg, loss):
    
#     log_file = os.path.join(
#         cfg.metric_dir,
#         datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"), '.json')
    
#     logging.basicConfig(filemode='a',
#                     datefmt='%H:%M:%S',
#                     level=logging.INFO)
#     logger = logging.getLogger(log_file)

#     file_handler=logging.FileHandler()
#     file_formatter=logging.Formatter(
#     "%(message)s")
#     file_handler.setFormatter(file_formatter)
#     logger.addHandler(file_handler)
    


