import os
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"

S = 10 # no of clients per round
trial=1
fraction_fit = 1
local_epochs = [1]
version = ['LDAWA']
data_type = 'NIID'
ssl_type = ['simclr']
Rounds = 200
p_size = 10

root_path = os.path.join("/", (*os.getcwd().strip().split('/')[:-2]))
for ssl in ssl_type:
    for local_e in local_epochs:
        for ver in version:
            process_obj = subprocess.run(["python",
            f"main_CIFAR10_{ver}.py",
            f"--config_path", f"{root_path}/inventory/config/fed_pretrain/my_idea/{data_type}/{ssl}/cross_silo_improved/CIFAR10_{p_size}C_{p_size*fraction_fit}S_10LE-Dir0.1_resnet34.json",
            f"--exp_name", f"{ver}_{p_size}C_{S}_{local_e}LE_{trial+1}",
            f"--pool_size", f"{p_size}",
            f"--rounds", f"{Rounds}",
            f"--fraction_fit", f"{fraction_fit}",
            f"--local_epochs", f"{local_e}"
            ])

