import os
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
trial = 
S = 10
Local_Epochs = [10]
Rounds = [200]
clients = 10
ver = ["LDAWA"]
finetune_type = 'linear'
ssl_type = ["simclr"]
num_gpu = 1
distributed_backend = 'None'
root_path = os.path.join("/", (*os.getcwd().strip().split('/')[:-2]))
for r in Rounds:
    for le in Local_Epochs:
        for ssl in ssl_type:
            for v in ver:
                process_obj = subprocess.run(["python",
                f"{root_path}/fedssl/image_ssl_finetune.py",
                f"--config_path", f"{root_path}/inventory/config/fed_Linear/resnet_18/w_momentum/NIID/{ssl}/cross-Silo/cifar10/10C_10S_10LE_RAND_DAWA_10_clients-Dir0.1_improved_resnet34.json",
                f"--exp_type", f"{v}_{clients}C_{S}_{le}LE_{trial}",
                f"--rounds", f"{r}",
                f"--num_gpu", f"{num_gpu}",
                f"--backend", f"{distributed_backend}",
                f"--finetune_type", f"{finetune_type}",
                ]) 



