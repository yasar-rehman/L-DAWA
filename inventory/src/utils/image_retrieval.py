import numpy as np
import os
import torch
import json
from typing import Any, Dict, List, Tuple
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from tqdm import tqdm
# from mmcv.parallel import MMDistributedDataParallel, MMDataParallel, collate
import pickle
import pathlib
import matplotlib.pyplot as plt
import copy
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from flwr.common import parameter
import torch.nn as nn
import torch.nn.functional as F
RES = 25
# Controls the margin from the optim starting point to the edge of the graph.
# The value is a multiplier on the distance between the optim start and end
MARGIN = 0.3
strt_point = -0.25
end_point = 0.25
# import ray
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
class IMAGE_RETRIEVAL():
    def __init__(
        self,
        model,train_loader, train_dataset, val_dataloader, val_dataset, cfg):

        # self.embs_out = 1024
        self.model = model
        self.train_dataloader = train_loader
        self.val_dataloader = val_dataloader
        self.config = cfg
        # optim_path = os.path.join(self.pretrain_model.exp_dir, self.pretrain_model.checkpoint_name)
        self.load_pretrained_model()
        self.model.eval()
        
        self.model = self.model.to(device)

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


        

    def extract_features(self, tqdm_disable=False):
        features = []
        classes = []

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        for data in tqdm(self.train_dataloader):
            _, inp, _, _, label  = data
            inp = inp.to(device)

            if self.config.model_params.resnet_small:
                if self.config.model_params.use_prepool:
                    # print("########## Using the pre-pooling layers #############")
                    # print(self.encoder(img, layer=5).shape)
                    embs = self.model(inp, layer=5)
                else:
                    embs = self.model(inp, layer=6)
            elif self.config.model_params.resnet_50:
                if self.config.model_params.use_prepool:
                    # print("########## Using the pre-pooling layers #############")
                    # print(self.encoder(img, layer=5).shape)
                    embs = self.model(inp, layer=5)
                else:
                    embs = self.model(inp, layer=6)
            
            elif self.config.model_params.resnet_34:
                if self.config.model_params.use_prepool:
                    # print("########## Using the pre-pooling layers #############")
                    # print(self.encoder(img, layer=5).shape)
                    embs = self.model(inp, layer=5)
                else:
                    embs = self.model(inp, layer=6)

            else:
                self.model = nn.Sequential(*list(self.model.children())[:-1])  # keep pooling layer
                embs = self.model(inp)
            
            embs = nn.functional.adaptive_max_pool2d(embs, 1)
            
            embs = embs.view(self.config.optim_params.batch_size, -1)
            # embs = torch.nn.functional.normalize(embs, dim=1)
            
            # print(embs.shape)
            # embs = self.model(inp)
            features.append(embs.detach().cpu().numpy())
            classes.append(label.detach().cpu().numpy())
        
        # print(np.array(features).shape)
        
        features = np.array(features).reshape(-1, embs.shape[1])
        classes = np.array(classes).reshape(-1,)
        np.save(os.path.join(self.config.exp_dir, 'train_feature.npy'), features)
        np.save(os.path.join(self.config.exp_dir, 'train_class.npy'), classes)
        
        # compute the features for the testset 
        features = []
        classes = []
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        for data in tqdm(self.val_dataloader):
            _, img, _, _, label  = data
            inp = inp.to(device)
            emb = self.model(inp)
            
            if self.config.model_params.resnet_small:
                if self.config.model_params.use_prepool:
                    # print("########## Using the pre-pooling layers #############")
                    # print(self.encoder(img, layer=5).shape)
                    embs = self.model(inp, layer=5)
                else:
                    embs = self.model(inp, layer=6)
            
            elif self.config.model_params.resnet_50:
                if self.config.model_params.use_prepool:
                    # print("########## Using the pre-pooling layers #############")
                    # print(self.encoder(img, layer=5).shape)
                    embs = self.model(inp, layer=5)
                else:
                    embs = self.model(inp, layer=6)
            
            elif self.config.model_params.resnet_34:
                if self.config.model_params.use_prepool:
                    # print("########## Using the pre-pooling layers #############")
                    # print(self.encoder(img, layer=5).shape)
                    embs = self.model(inp, layer=5)
                else:
                    embs = self.model(inp, layer=6)
            else:
                self.model = nn.Sequential(*list(self.model.children())[:-1])  # keep pooling layer
                embs = self.model(inp)
            
            embs = nn.functional.adaptive_max_pool2d(embs, 1)
            embs = embs.view(self.config.optim_params.batch_size, -1)
            # embs = torch.nn.functional.normalize(embs, dim=1)
            # print(embs.shape)
            # embs = self.model(inp)
            features.append(embs.detach().cpu().numpy())
            classes.append(label.detach().cpu().numpy())
         
        features = np.array(features).reshape(-1, embs.shape[1])
        classes = np.array(classes).reshape(-1,)
        np.save(os.path.join(self.config.exp_dir, 'val_feature.npy'), features)
        np.save(os.path.join(self.config.exp_dir, 'val_class.npy'), classes)

    

    def knn_monitor(self):
        
        # net.eval()
        k=200
        t=0.07
        # device = None

        memory_data_loader = self.train_dataloader
        test_data_loader = self.val_dataloader

        classes = 10
        total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
        train_labels = []
        for p in self.model.parameters():
            p.requires_grad = False

        with torch.no_grad():
            # generate feature bank
            for _, data, _, _, target in tqdm(memory_data_loader):
                if device is None:
                    data = data.cuda(non_blocking=True)
                else:
                    data = data.to(device, non_blocking=True)


                if self.config.model_params.resnet_small:
                    if self.config.model_params.use_prepool:
                        # print("########## Using the pre-pooling layers #############")
                        # print(self.encoder(img, layer=5).shape)
                        embs = self.model(data, layer=5)
                    else:
                        embs = self.model(data, layer=6)
                elif self.config.model_params.resnet_50:
                    if self.config.model_params.use_prepool:
                        # print("########## Using the pre-pooling layers #############")
                        # print(self.encoder(img, layer=5).shape)
                        embs = self.model(data, layer=5)
                    else:
                        embs = self.model(data, layer=6)
                
                elif self.config.model_params.resnet_34:
                    if self.config.model_params.use_prepool:
                        # print("########## Using the pre-pooling layers #############")
                        # print(self.encoder(img, layer=5).shape)
                        embs = self.model(data, layer=5)
                    else:
                        embs = self.model(data, layer=6)
                else:
                    self.model = nn.Sequential(*list(self.model.children())[:-1])  # keep pooling layer
                    embs = self.model(data)
        
                embs = embs.view(self.config.optim_params.batch_size, -1)
                feature = embs
                feature = F.normalize(feature, dim=-1)
                feature_bank.append(feature)
                train_labels.append(target.detach().numpy())
            # [D, N]
            
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # feature_bank = F.normalize(feature_bank, dim=1)
            # [N]
            # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
            feature_labels = torch.tensor(train_labels).reshape(-1,).to(device)
            # loop test data to predict the label by weighted knn search
            
            test_bar = tqdm(test_data_loader, desc='kNN')
            for _,data, _, _, target in test_bar:
                if device is None:
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                else:
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                if self.config.model_params.resnet_small:
                    if self.config.model_params.use_prepool:
                        # print("########## Using the pre-pooling layers #############")
                        # print(self.encoder(img, layer=5).shape)
                        embs = self.model(data, layer=5)
                    else:
                        embs = self.model(data, layer=6)
                elif self.config.model_params.resnet_50:
                    if self.config.model_params.use_prepool:
                        # print("########## Using the pre-pooling layers #############")
                        # print(self.encoder(img, layer=5).shape)
                        embs = self.model(data, layer=5)
                    else:
                        embs = self.model(data, layer=6)
                
                elif self.config.model_params.resnet_34:
                    if self.config.model_params.use_prepool:
                        # print("########## Using the pre-pooling layers #############")
                        # print(self.encoder(img, layer=5).shape)
                        embs = self.model(data, layer=5)
                    else:
                        embs = self.model(data, layer=6)
                else:
                    self.model = nn.Sequential(*list(self.model.children())[:-1])  # keep pooling layer
                    embs = self.model(data)
                
                embs = embs.view(self.config.optim_params.batch_size, -1)
                feature = embs
                feature = F.normalize(feature, dim=-1)

                pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

                total_num += data.size(0)
                total_top1 += (pred_labels[:, 0] == target).float().sum().item()
                test_bar.set_postfix({'Accuracy': total_top1 / total_num * 100})
            print("Accuracy: {}".format(total_top1 / total_num * 100))
        return total_top1 / total_num * 100

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
            print(len(self.model.state_dict().items()), len(model_weights))

            params_dict = zip(self.model.state_dict().keys(), model_weights)
            state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        else:
            # params_dict = zip(self.model.state_dict().keys(), model_weights)
            # state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
            # self.model.load_state_dict(state_dict, strict=True)
            model_weights = torch.load(pretrain_weight_path)
            weights = [val.cpu().numpy() for _, val in model_weights['state_dict'].items()]
            params_dict = zip(self.model.state_dict().keys(), weights)
            state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    
    def topk_retrieval(self):
        """Extract features from test split and search on train split features."""
        print('Load local .npy files.')
        X_train = np.load(os.path.join(self.config.exp_dir, 'train_feature.npy'))
        y_train = np.load(os.path.join(self.config.exp_dir, 'train_class.npy'))
        # X_train = np.mean(X_train,1)
        y_train = y_train
        X_train = X_train.reshape((-1, X_train.shape[-1]))
        y_train = y_train.reshape(-1)

        X_test = np.load(os.path.join(self.config.exp_dir, 'val_feature.npy'))
        y_test = np.load(os.path.join(self.config.exp_dir, 'val_class.npy'))
        # X_test = np.mean(X_test,1)
        y_test = y_test
        X_test = X_test.reshape((-1, X_test.shape[-1]))
        y_test = y_test.reshape(-1)


        X_train = torch.from_numpy(X_train)
        # y_train = torch.from_numpy(y_train)
        X_test = torch.from_numpy(X_test)
        # y_test = torch.from_numpy(y_test)

        inner_prod = torch.matmul(X_test, X_train.transpose(0, 1))
        train_norm = X_train.norm(p=None, dim=1)
        cos_dist = inner_prod / train_norm.view(1, -1)  # [N_test, N_train]
        cos_dist = cos_dist.cpu().numpy()
        print(cos_dist.shape)
        indices = np.argsort(cos_dist, axis=1)[:, ::-1]
        


        train_labels  = y_train
        test_labels = y_test
        assign_labels = train_labels[indices]

        results = dict(
            train_labels=train_labels,
            test_labels=test_labels,
            assign_labels=assign_labels,
            indices = indices
        )
        results_12 = {}

        for nk in [1, 5, 10, 20, 50, 200]:
            correct = test_labels.reshape(-1, 1) == assign_labels[:, 0:nk]
            correct = np.any(correct, axis=1)
            num_correct = correct.astype(np.float32).sum()
            accuracy = num_correct / test_labels.shape[0]
            print(accuracy)
            results['top{}_acc'.format(nk)] = accuracy
            results_12['top{}_acc'.format(nk)] = accuracy

        with open(os.path.join(self.config.exp_dir,  'topk_correct.json'), 'w') as fp:
            json.dump(results_12, fp)

        return results



        # ks = [1, 5, 10, 20, 50]
        # topk_correct = {k:0 for k in ks}

        # # distances = cosine_distances(X_test, X_train)
        # # indices = np.argsort(distances)

        # for k in ks:
        #     # print(k)
        #     top_k_indices = indices[:, :k]
        #     print(top_k_indices.shape, y_test.shape)
        #     for ind, test_label in zip(top_k_indices, y_test):
        #         labels = y_train[ind]
        #         if test_label in labels:
        #             print(test_label, labels)
        #             topk_correct[k] += 1

        # for k in ks:
        #     correct = topk_correct[k]
        #     total = len(X_test)
        #     print('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(k, correct, total, correct/total))

        # with open(os.path.join(self.config.exp_dir, 'topk_correct.json'), 'w') as fp:
        #     json.dump(topk_correct, fp)

           
   



    # def load_pretrained_model(self):
    #         base_dir = self.config.pretrain_model.exp_dir
    #         checkpoint_name = self.config.pretrain_model.checkpoint_name
    #         pretrain_weight_path = os.path.join(base_dir, checkpoint_name)
    #         model_weights = torch.load(pretrain_weight_path)
    #         # print(model_weights['state_dict'].keys())
            
            
    #         weights = [val.cpu().numpy() for _, val in model_weights['state_dict'].items()]
    #         params_dict = zip(self.model.state_dict().keys(), weights)
    #         state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    #         self.model.load_state_dict(state_dict, strict=True)



# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # print(feature_bank.norm(p=None, dim=0))
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels



    






    
    
    
    






def _load_checkpoint(_model, _chk_path):
    state_dict={}
    
    params = np.load(_chk_path, allow_pickle=True)
    if _chk_path.endswith('.array_copy.npz'):
        params = params['arr_0']
        params_list = []
        for item in params:
            if type(item) != np.ndarray:
                item = np.array(item)
            params_list.append(item)
        params = params_list     
    else:
        params = params['arr_0'].item()
        params = parameter.parameters_to_weights(params)
    # _load_checkpoint(_model, params)
    print(len(params), len(_model.state_dict().keys()))
    params_dict = zip(_model.state_dict().keys(), params)
    
    state_dict['state_dict'] = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
   
    print(f"length of the ordered dict_keys is: {len(state_dict['state_dict'].keys())}")
    # _model.load_state_dict(state_dict, strict=True)

    return state_dict




def get_model(_model, tr_model) -> List[np.ndarray]:
    # Return local model parameters as a list of NumPy ndarrays
    # print(_model)
    
    if _model is not None:
        if _model.endswith('.npz'):
            optim = _load_checkpoint(tr_model, _model)
            print("The model is being loaded...")
        else:
            print("The model is being loaded...")
            optim = torch.load(_model)
    
    if isinstance(optim['state_dict'], dict):
        return optim['state_dict']
    else:
        raise ValueError("Cannot find state_dict")
    
    return optim

def cosine_distances(X, Y=None):
    """Compute cosine distance between samples in X and Y.
    Cosine distance is defined as 1.0 minus the cosine similarity.
    Read more in the :ref:`User Guide <metrics>`.
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        Matrix `X`.
    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), \
            default=None
        Matrix `Y`.
    Returns
    -------
    distance matrix : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the cosine distance between samples in X and Y.
    See Also
    --------
    cosine_similarity : Compute cosine similarity between samples in X and Y.
    scipy.spatial.distance.cosine : Dense matrices only.
    """
    # 1.0 - cosine_similarity(X, Y) without copy
    S = cosine_similarity(X, Y)
    S *= -1
    S += 1
    np.clip(S, 0, 2, out=S)
    if X is Y or Y is None:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        S[np.diag_indices_from(S)] = 0.0
    return S

def cosine_similarity(X, Y=None, dense_output=True):
    """Compute cosine similarity between samples in X and Y.
    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:
        K(X, Y) = <X, Y> / (||X||*||Y||)
    On L2-normalized data, this function is equivalent to linear_kernel.
    Read more in the :ref:`User Guide <cosine_similarity>`.
    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples_X, n_features)
        Input data.
    Y : {ndarray, sparse matrix} of shape (n_samples_Y, n_features), \
            default=None
        Input data. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``.
    dense_output : bool, default=True
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input arrays are sparse.
        .. versionadded:: 0.17
           parameter ``dense_output`` for dense output.
    Returns
    -------
    kernel matrix : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the cosine similarity between samples in X and Y.
    """
    # to avoid recursive import

    X, Y = check_pairwise_arrays(X, Y)

    X_normalized = normalize(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, copy=True)

    K = safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=dense_output)

    return K


def check_pairwise_arrays(
    X,
    Y,
    *,
    precomputed=False,
    dtype=None,
    accept_sparse="csr",
    force_all_finite=True,
    copy=False,
):
    """Set X and Y appropriately and checks inputs.
    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.
    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats (or dtype if provided). Finally, the function
    checks that the size of the second dimension of the two arrays is equal, or
    the equivalent check for a precomputed distance matrix.
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)
    precomputed : bool, default=False
        True if X is to be treated as precomputed distances to the samples in
        Y.
    dtype : str, type, list of type, default=None
        Data type required for X and Y. If None, the dtype will be an
        appropriate float type selected by _return_float_dtype.
        .. versionadded:: 0.18
    accept_sparse : str, bool or list/tuple of str, default='csr'
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.
    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:
        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.
        .. versionadded:: 0.22
           ``force_all_finite`` accepts the string ``'allow-nan'``.
        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`.
    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
        .. versionadded:: 0.22
    Returns
    -------
    safe_X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        An array equal to X, guaranteed to be a numpy array.
    safe_Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.
    """
    X, Y, dtype_float = _return_float_dtype(X, Y)

    estimator = "check_pairwise_arrays"
    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = check_array(
            X,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            estimator=estimator,
        )
    else:
        X = check_array(
            X,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            estimator=estimator,
        )
        Y = check_array(
            Y,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            estimator=estimator,
        )

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError(
                "Precomputed metric requires shape "
                "(n_queries, n_indexed). Got (%d, %d) "
                "for %d indexed." % (X.shape[0], X.shape[1], Y.shape[0])
            )
    elif X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Incompatible dimension for X and Y matrices: "
            "X.shape[1] == %d while Y.shape[1] == %d" % (X.shape[1], Y.shape[1])
        )

    return X, 
def normalize(X, norm="l2", *, axis=1, copy=True, return_norm=False):
    """Scale input vectors individually to unit norm (vector length).
    Read more in the :ref:`User Guide <preprocessing_normalization>`.
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).
    axis : {0, 1}, default=1
        Define axis used to normalize the data along. If 1, independently
        normalize each sample, otherwise (if 0) normalize each feature.
    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).
    return_norm : bool, default=False
        Whether to return the computed norms.
    Returns
    -------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Normalized input X.
    norms : ndarray of shape (n_samples, ) if axis=1 else (n_features, )
        An array of norms along given axis for X.
        When X is sparse, a NotImplementedError will be raised
        for norm 'l1' or 'l2'.
    See Also
    --------
    Normalizer : Performs normalization using the Transformer API
        (e.g. as part of a preprocessing :class:`~sklearn.pipeline.Pipeline`).
    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """
    if norm not in ("l1", "l2", "max"):
        raise ValueError("'%s' is not a supported norm" % norm)

    if axis == 0:
        sparse_format = "csc"
    elif axis == 1:
        sparse_format = "csr"
    else:
        raise ValueError("'%d' is not a supported axis" % axis)

    X = check_array(
        X,
        accept_sparse=sparse_format,
        copy=copy,
        estimator="the normalize function",
        dtype=FLOAT_DTYPES,
    )
    if axis == 0:
        X = X.T

    if sparse.issparse(X):
        if return_norm and norm in ("l1", "l2"):
            raise NotImplementedError(
                "return_norm=True is not implemented "
                "for sparse matrices with norm 'l1' "
                "or norm 'l2'"
            )
        if norm == "l1":
            inplace_csr_row_normalize_l1(X)
        elif norm == "l2":
            inplace_csr_row_normalize_l2(X)
        elif norm == "max":
            mins, maxes = min_max_axis(X, 1)
            norms = np.maximum(abs(mins), maxes)
            norms_elementwise = norms.repeat(np.diff(X.indptr))
            mask = norms_elementwise != 0
            X.data[mask] /= norms_elementwise[mask]
    else:
        if norm == "l1":
            norms = np.abs(X).sum(axis=1)
        elif norm == "l2":
            norms = row_norms(X)
        elif norm == "max":
            norms = np.max(abs(X), axis=1)
        norms = _handle_zeros_in_scale(norms, copy=False)
        X /= norms[:, np.newaxis]

    if axis == 0:
        X = X.T

    if return_norm:
        return X, norms
    else:
        return X

def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    """
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = float

    return X, Y, 