import argparse
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional
import os
import flwr as fl
import numpy as np
from math import exp, log
import torch
import torch.nn as nn
from timeit import default_timer as timer
import re
import ray
import time
import shutil
import random

from dotmap import DotMap
from fedssl import image_ssl
import json
import time
from flwr.common import parameter
from functools import reduce
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

DIR = 'CIFAR10'


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
             
        weights_results = [
        (parameters_to_weights(fit_res.parameters), fit_res.metrics['loss'], fit_res.num_examples)
        for client, fit_res in results
        ]
        
        
        # Implementation of DACS
        GLB_RND = [fit_res.metrics['strt_rnd'] for client, fit_res in results][0]
        GLB_RND_INFO = [fit_res.metrics['glb_dir'] for client, fit_res in results][0]

        clients_wt_w_keys_ = [(parameters_to_weights(fit_res.parameters), 
                                  fit_res.metrics['state_dict_key'])
                                  for client, fit_res in results]

        
        
        if int(GLB_RND) > 0:
            tar_path = os.path.join(GLB_RND_INFO, 'round-{}-weights.npy.npz'.format(GLB_RND))
            glb_pr_parameters = load_npz_file(tar_path)

            # compute the L2 weights divergence between all current clients and previous model
            div_L2 = weights_div(glb_pr_parameters, clients_wt_w_keys_)

            assert len(div_L2) == len(clients_wt_w_keys_)

            # compute the mean of divergence
            out_STD = np.mean(div_L2)

            # file to write 
            path_STD = os.path.join(GLB_RND_INFO, 'DACS_mean.txt')

            with open(path_STD, 'a') as f:
                f.write('{},{}\n'.format(str(GLB_RND), str(out_STD)))


        if int(GLB_RND) > 2: # after warm up
           
            weights_results_fedavg = [(para, num) for para, loss, num in weights_results]
            start = timer()
            weights_loss = my_aggregate(weights_results_fedavg,glb_pr_parameters) # fedavg          
            weights = weights_to_parameters(weights_loss)

        else:     

            weights_results_fedavg = [(para, num) for para, loss, num in weights_results] 
            start = timer()
            weights_loss = aggregate(weights_results_fedavg) # fedavg
            end = timer()
            print("The time for this strategy is:", (end - start))
            weights = weights_to_parameters(weights_loss)

        avg_loss_all_C = [-1*log(fit_res.metrics['loss']) for client, fit_res in results]
        avg_loss_all_c = np.sum(avg_loss_all_C)/len(avg_loss_all_C)
        path_avg_loss = os.path.join(GLB_RND_INFO, 'DACS_random_global_loss.txt')
                   
        if rnd == 1:
            rnd = int(GLB_RND) + rnd
        else:
            rnd = int(GLB_RND) + 1

        # saving the average loss
        with open(path_avg_loss, 'a') as f:
                f.write('{},{}\n'.format(str(rnd), str(avg_loss_all_c)))
        
        if weights is not None:
            # save weights
            print(f"round-{rnd}-weights...",)
            glb_dir = GLB_RND_INFO
            if not os.path.exists(os.path.abspath(glb_dir)):
                os.makedirs(os.path.abspath(glb_dir))
            np.savez(os.path.join(glb_dir,f"round-{rnd}-weights.npy"), weights)
        return weights, {}


def my_aggregate(results: List[Tuple[Weights, float]], w_g):
    d = np.arange(len(w_g)) # numpy array equal to the length of layers
    c = [weights for weights, _ in results] # pass: it will give the weights of the clients as a list
    # print(c[0][0].shape)
    total_clients = len(results)
    for wi in range(len(c)):
        for x, y, id in zip(c[wi], w_g, d): # for each layer
            if len(x.shape) and len(y.shape) > 0: # remove the empty values
                if np.linalg.norm(x) > 0: # if it is zero
                    v = (x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y))
                    if v > 1: 
                        v = 1.0 # the value cannot be greater than 1
                        c[wi][id] = x*v 
                    else:
                        # multiply the layer by the cost value
                        c[wi][id] =  x*v 
            else:
                c[wi][id] = c[wi][id] # retain the actual value; e.g tracking mean, tracking var
    

    # after computing the divergence apply the averaging
    weights_prime: Weights = [
        reduce(np.add, layer_updates) / total_clients
        for layer_updates in zip(*c)
    ]
    return weights_prime


def aggregate(results: List[Tuple[Weights, float]]) -> Weights:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    weights_prime: Weights = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def load_npz_file(parameters):
    print('load pretrain parameters')
    params = np.load(parameters, allow_pickle=True)
    params = parameters_to_weights(params['arr_0'].item()) #return a list of weights
    return params


def weights_div(glb_params, client_model):
    weights_diff_L2 = []
    for i in range(len(client_model)):

        c_state_dict = client_model[i][-1]  #state_dict keys
        c_state_dict_valid = Extract_Layers(c_state_dict) #only extract weights and biases
        
        c_parameters = client_model[i][0]  # state_dict parameters

        client_i = [v for k, v in zip(c_state_dict, c_parameters) 
                    if k in c_state_dict_valid]

        glb_model = [v for k, v in zip(c_state_dict, glb_params)
                    if k in c_state_dict_valid]
        # print(len(client_i))
        # print(len(glb_model))
        weight_diff = [np.square(A.flatten() - B.flatten()) 
                    for A, B in zip(glb_model, client_i)]

        weights_diff = np.sum(np.concatenate(weight_diff)) / len(weight_diff)

        weights_diff_L2.append(weights_diff)
    
    return weights_diff_L2

def Extract_Layers(layer_names):
    # input state_dict_keys
    layer_names_valid = []
    for layer_name in layer_names:
        if not (layer_name.endswith('num_batches_tracked') or layer_name.endswith('running_mean') or layer_name.endswith('running_var') or layer_name.startswith('head')):
            layer_names_valid.append(layer_name)
    
    return layer_names_valid


# order classes by number of samples
def takeSecond(elem):
    return elem[1]


# Flower Client
class SslClient(fl.client.NumPyClient):
    """Flower client implementing video SSL w/ PyTorch."""

    def __init__(self, model, train_data_loader, train_dataset, 
                test_data_loader, cfg, args, image_ssl):
        # model, train_data_loader,train_dataset, test_data_loader, config, args,  image_ssl
        self.model = model
        self.train_data_loader = train_data_loader
        self.train_dataset = train_dataset
        self.test_data_loader = test_data_loader
        self.cfg = cfg
        self.args = args
        self.image_ssl = image_ssl
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
       

    def get_parameters(self) -> List[np.ndarray]:
        # Return local model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
       _set_parameters(self.model, parameters)

        
    
    def get_properties(self, ins):
        return self.properties

    def fit(self, parameters, config):
        # before replacing the new parameters find the old parameters if any 
        chk_name_list = [fn for fn in os.listdir(self.args.checkpoint_dir) if fn.endswith('.ckpt')]
        
        # print(chk_name_list)
        if chk_name_list:
            checkpoint = os.path.join(self.args.checkpoint_dir, chk_name_list[-1])
            pr_model = torch.load(checkpoint)
            state_dict_pr = pr_model['state_dict']
            if list(state_dict_pr.keys())[0].startswith('model.'):
                state_dict_new = {k[6:]: v for k, v in state_dict_pr.items()}
            # load the model with previous state_dictionary
            # self.model.load_state_dict(state_dict_new, strict=True)
            print("The local model has been updated with local weights")
        # # Update local model w/ global parameters
        self.set_parameters(parameters)
        # Get hyperparameters from config
        # Train model on client-local data
        self.image_ssl.train_model_cl(self.model, 
                                    self.train_data_loader,
                                    self.train_dataset, 
                                    self.test_data_loader, 
                                    self.cfg)
        
        
        # Return updated model parameters to the server
        num_examples = len(self.train_dataset)  # TODO len(self.trainloader)

        # fetch loss from log file
        work_dir = self.args.work_dir
        log_f_list = []
        for f in os.listdir(work_dir):
            if f.endswith('log.json'):
                num = int(''.join(f.split('log.')[0].split('_')))
                log_f_list.append((f, num))

        # take the last log file
        log_f_list.sort(key=takeSecond)
        log_f_name = work_dir + '/' + log_f_list[-1][0]
        loss_list = []
        with open(log_f_name, 'r') as f:
            for line in f.readlines():
                line_dict = eval(line.strip())
                loss = float(line_dict['loss'])
                loss_list.append(loss)

        avg_loss = sum(loss_list) / len(loss_list)
       
        exp_loss = exp(- avg_loss)
        metrics = {'loss': exp_loss}

        # get the model keys 
        metrics['state_dict_key'] = [k for k in self.model.state_dict().keys()]
        # metrics['global_round'] = global_round
        metrics['glb_dir'] = self.args.glb_dir
        metrics['strt_rnd'] = self.args.strt_rnd

        return self.get_parameters(), num_examples, metrics

    def evaluate(self, parameters, config):
        
        result = 0
        # print(float(0))
        # print(int(len(self.test_dataset)))
        # print(float(result))
        return float(0), int(60000), {"accuracy": float(result)}

def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def _temp_get_parameters(model):
    # Return local model parameters as a list of NumPy ndarrays
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
    }
    return config



def mkdir_or_exist(dir_name, mode=0o777):
    # copied from https://github.com/open-mmlab/mmcv/blob/7e6f4624954d62e6900c3927c7608785cb4a593f/mmcv/utils/path.py#L26
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedSSL_loss')
    parser.add_argument('--rounds', 
                        default=200, 
                        type=int, 
                        help='pretrain epoch to be loaded')
    parser.add_argument('--pool_size', 
                        default=100, 
                        type=int, 
                        help='Pool Size')
    parser.add_argument('--resume', 
                        dest='resume', 
                        action='store_true')
    parser.add_argument('--resume_checkpoint', 
                        default='None', 
                        type=str, 
                        help='resume checkpoint path')
    parser.add_argument('--config_path',
                        default=None,
                        type=str,
                        help='pretrain path to the configuration file')
    parser.add_argument('--exp_name',
                        default=None,
                        type=str,
                        help='pretrain path to the configuration file')
    parser.add_argument('--fraction_fit',
                        default=0.1,
                        type=float,
                        help='pretrain path to the configuration file')
    
    parser.add_argument('--local_epochs',
                        default=10,
                        type=int,
                        help='number of local epochs')
    
    args = parser.parse_args()

    if args.config_path:
        with open(args.config_path) as f:
            x = json.load(f)
            config = DotMap(x)
    else:
        raise ValueError('Cannot find the configuration file')
    args = parser.parse_args()
    
    ######################################################################################################
    
    if args.exp_name:
        config.Experiment_name = args.exp_name
    
    GLB_DIR = os.path.join(config.default_path, config.Experiment_name) 
    args.glb_dir = GLB_DIR
    
    # args.glb_dir = GLB_DIR + 
    #     'full_iid_fedavg_k10_DACS_random_r_0.3_w_moment_selection/glb_epochs/' + DIR
    mkdir_or_exist(args.glb_dir)
    glb_chk_list = [fn for fn in os.listdir(args.glb_dir) if fn.endswith('.npz')]
    RND = [int(x.strip().split('-')[1]) for x in glb_chk_list]
    RND.sort() 
    if glb_chk_list:
        args.resume_checkpoint = os.path.join(args.glb_dir, 'round-'+ str(RND[-1]) + '-weights.npy.npz')
        # number of total rounds to be trained
        args.strt_rnd = RND[-1]
        rounds = args.rounds - args.strt_rnd  # e.g., 500 - 40 = 460
        
        initial_parameters = args.resume_checkpoint 
        # following changes are made here
        print('load pretrain parameters')
        params = np.load(initial_parameters, allow_pickle=True)
        params = params['arr_0'].item()
        # params = parameter.parameters_to_weights(params)
        initial_parameters = params
    else:
        rounds = args.rounds
        args.strt_rnd = 0
        initial_parameters = None

    # print(args.resume_checkpoint)
    # number of dataset partions (= number of total clients)
    pool_size = args.pool_size 
    client_resources = {"num_cpus": 2,"num_gpus": 1}  # each client will get allocated 1 CPUs
    timestr = time.strftime("%Y%m%d_%H%M%S")
    
   
   
    
    ################################################################################################## 

    # configure the strategy
    strategy = SaveModelStrategy(
        fraction_fit=args.fraction_fit,
        fraction_eval=0.001,
        min_fit_clients=int(args.fraction_fit*pool_size),
        min_eval_clients=1,
        min_available_clients=pool_size,
        initial_parameters=initial_parameters,
    )
     
    def main(cid:str):
    # Parse command line argument `cid` (client ID)
        cid_plus_one = str(int(cid)+1)
               
        # configuration files
        config.data_params.img_list = os.path.join(config.data_params.img_list, 
                                        'client_dist'+cid_plus_one +'.json')    

        Experiment_name_path = os.path.join(config.default_path, 
                                          config.Experiment_name)   

        mkdir_or_exist(Experiment_name_path)

        config.exp_base = os.path.join(Experiment_name_path,"client_" + cid_plus_one)
        exp_dir = os.path.join(config.exp_base, "experiments")
        
        config.exp_dir = exp_dir
        config.metric_dir = os.path.join(exp_dir, "metrics")
        print(config.metric_dir)

        args.work_dir = config.metric_dir
        args.checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        mkdir_or_exist(args.checkpoint_dir)

        # path to save the global rounds
        # args.glb_dir = GLB_DIR 

        glb_chk_list = [fn for fn in os.listdir(args.glb_dir) if fn.endswith('.npz')]
        RND = [int(x.strip().split('-')[1]) for x in glb_chk_list]
        RND.sort() 
        if glb_chk_list:
            # number of total rounds to be trained
            args.strt_rnd = RND[-1]
        else:
            args.strt_rnd = 0

        if args.local_epochs:
            config.num_epochs = args.local_epochs
        # load the model
        model = image_ssl.load_model(config)
       
        # load the training data
        train_data_loader, train_dataset = image_ssl.load_data(config)

        # load the test data
        test_data_loader=None
   
        # Initialize and start client
        return SslClient(model, train_data_loader,train_dataset, test_data_loader, config, args,  image_ssl)
    # (optional) specify ray config
    ray_config = {"include_dashboard": False}

     # start simulation
    fl.simulation.start_simulation(
        client_fn=main,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=rounds,
        strategy=strategy,
        ray_init_args=ray_config,
    )



    


   
