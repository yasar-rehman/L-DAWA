# L-DAWA
This repository contains the source code for [L-DAWA: Layer-wise Divergence Aware Weight Aggregation in Federated Self-Supervised Visual Representation Learning](https://arxiv.org/abs/2307.07393) that has been accepted in [ICCV-2023](https://iccv2023.thecvf.com/). </br>  </br>

![](/L-DAWA_.jpeg)

[Paper](https://arxiv.org/abs/2307.07393), [Supplementary materials](https://yasar-rehman.github.io/files/ICCV2023_image_SSL_FL__supplementary_.pdf) 

# Authors
- [Yasar Abbas Ur Rehman](https://yasar-rehman.github.io/yasar/) ,  [Yan Gao](https://www.cst.cam.ac.uk/people/yg381), [Pedro Porto Buarque de Gusmão](https://portobgusmao.com/), [Mina Alibegi](https://www.linkedin.com/in/mina-alibeigi-2b47739a/?originalSubdomain=se), [Jiajun Shen](https://www.linkedin.com/in/jiajunshen/), and [Nicholas D. Lane](http://niclane.org/) <br>

# Requirements:
- [x] [Anaconda environment with python 3.8](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) 
- [x] [Pytorch Lightning](https://lightning.ai/) : 1.4.19
- [x] [Flower](https://flower.dev/): 0.17.0 <br>

* For a complete list of the required packages please see the [requirement.txt](https://github.com/yasar-rehman/L-DAWA/blob/main/requirements.txt) file. One can easily install, all the requirements by running ````pip install -r requirement.txt````.
# Tutorials and Helping materials
* Federated Learning with PyTorch Lightning and Flower -- > https://github.com/adap/flower/tree/main/examples/quickstart-pytorch-lightning
* Building a custom strategy from scratch with Flower ---> https://flower.dev/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html
# How to run the code
* Go to the folder ```FL_Pretraining``` and run the ```pretrain_FL.py``` script.
* To execute the finetuning run the ```finetune_script.py``` script.    

# Datasets
* CIFAR-10, CIFAR-100, Tiny-ImageNet

# Way forward
## Supervised FL
* Although our main motivation for designing L-DAWA was to tackle the client bias in _Cross-Silo_ FL scenarios, in which each client runs a Self Supervised Learning (SSL) algorithm. L-DAWA can equally work well in _Cross-Silo_ FL scenarios with Supervised Learning (SL) algorithms as shown below (See. [Supplementary materials](https://yasar-rehman.github.io/files/ICCV2023_image_SSL_FL__supplementary_.pdf) for the details.)


  |Aggregation Type | E1 | E5 | E10|
  |-----------------|----|----|----| 
  FedAvg  | 77.91 | 83.76 | 81.31 |
  FedYogi | 77.49 | 72.50 | 74.85 |
  FedProx | 80.55 | 74.87 | 72.24 |
  L-DAWA  | 81.96 | 84.68 |  82.35|
   



## Performance with Large Models
The results below are obtained by pertaining SimCLR in _Cross-Silo_ FL settings first followed by linear-finetuning. 

### Results on CIFAR-100
  |Method | Architecture | E1 | E5 |E10|
  |-------|--------------|----|----|----|
  FedAvg | ResNet34 | 54.76  | 69.81 | 74.21|
  FedU | ResNet34 | 52.85 | 67.84 | 72.21|
  L-DAWA | ResNet34  | 61.92 | 73.33  | 77.30|
  FedAvg  |ResNet50 | 63.62  | 75.39 | 79.41 |
  FedU    |ResNet50 | 57.61 | 71.44 |  76.85 |
  L-DAWA  |ResNet50  | 63.90 | 75.58 | 79.11| 

### Results on Tiny-ImageNet
  
  |Method | Architecture | E1 |
  |-------|--------------|----|
  FedAvg | ResNet34|11.93|
  FedU   | ResNet34|11.75|
  L-DAWA | ResNet34 |18.64|
  FedAvg | ResNet50|13.51|
  FedU   | ResNet50|13.22|
  L-DAWA| ResNet50 |19.04|
         

# News
* Added the source code for L-DAWA
* Added the inventory folder
* Added the FL_pretraining script
* The build-up of this repository is in progress

# Issues: 
If you encounter any issues, feel free to open an issue in the GitHub. 

# Citations
````
@inproceedings{rehman2023dawa,
  title={L-DAWA: Layer-wise Divergence Aware Weight Aggregation in Federated Self-Supervised Visual Representation Learning},
  author={Rehman, Yasar Abbas Ur and Gao, Yan and de Gusmao, Pedro Porto Buarque and Alibeigi, Mina and Shen, Jiajun and Lane, Nicholas D},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16464--16473},
  year={2023}
}
````
