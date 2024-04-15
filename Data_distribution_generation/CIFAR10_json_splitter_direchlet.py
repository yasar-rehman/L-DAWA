from http import client
import numpy as np
import argparse
import json
import copy
import os



def load_json(data_path):
    with open(data_path, 'r') as f:
        json_object = json.load(f)
        json_object_copy = copy.deepcopy(json_object)
        
    name_list = []
    label_list = []
    for i in json_object_copy:
        # print(i['name'])
        name_list.append(i['name']) 
        label_list.append(i['label']) 
    name_list = np.asarray(name_list)
    label_list = np.asarray(label_list)
    return name_list, label_list
  
def partition_data(dataset, 
                   data_path,
                   out_path, 
                   partition_type, 
                   clients, 
                   dir_alpha=0.1,
                   ):
    if dataset == 'Cifar10':
        # data loading function
        datalist_name, datalist_labels = load_json(data_path)
        datasize = len(datalist_labels)
        
        # non-i.i.d partition based on dirichlet distribution
        nofclients = clients
        nof_cls = 10
       
        if partition_type == 'N_IID':
            alpha = dir_alpha
        elif partition_type == 'IID':
            # https://github.com/akhilmathurs/orchestra/blob/68191ff63b7244262b28a5d8376225963067b324/sampler.py#L104
            alpha = 1e5
        else:
            raise ValueError("cannot interpret the partition type. The partition can be IID or N_IID")   
        
        
        idx_k = {c_label: np.where(datalist_labels ==c_label)[0] 
                                   for c_label in range(nof_cls)}
        np.random.seed(15)
        clients_dist = np.random.dirichlet(np.repeat(alpha, nofclients), size=nof_cls).transpose()
        clients_dist /= clients_dist.sum()
        
        samples_per_class_train = (np.floor(clients_dist * datasize))
        
        
        start_ids_train = np.zeros((nofclients+1, nof_cls), dtype=np.int32)
        
        for i in range(0, nofclients):
            start_ids_train[i+1] = start_ids_train[i] + samples_per_class_train[i]
        print(start_ids_train)
        # https://github.com/akhilmathurs/orchestra/blob/68191ff63b7244262b28a5d8376225963067b324/sampler.py
        print("\nSanity checks:")
        print("\tSum of dist. of classes over clients: {}".format(clients_dist.sum(axis=0)))
        print("\tSum of dist. of clients over classes: {}".format(clients_dist.sum(axis=1)))
        print("\tTotal trainset size: {}".format(samples_per_class_train.sum()))
        
        
        client_ids = {client_num: {} for client_num in range(nofclients)}
        for client_num in range(nofclients):
            l = np.array([], dtype=np.int32)
            for class_num in range(nof_cls):
                start, end = start_ids_train[client_num, class_num], start_ids_train[client_num+1, class_num]
                l = np.concatenate((l, idx_k[class_num][start:end].tolist())).astype(np.int32)
            client_ids[client_num] = l
        
       
        # convert the client_id to json list
        # print(client_ids)
        sum_list = []
        for i in range(len(client_ids)):
            datalist = []
            clients_idx = client_ids[i]
            client_name = datalist_name[clients_idx]
            client_labels = datalist_labels[clients_idx]
            assert len(client_name) == len(client_labels)
            # print(f'client{i} gets {len(client_name)} samples')
            sum_list.append(len(client_name))
            for x, y in zip(client_name, client_labels):
                data_dict = {"name": x,
                            "label": int(y)}
                datalist.append(data_dict)
            
            with open(os.path.join(out_path,"client_dist" + str(i+1) + ".json"), "w") as f:
                json.dump(datalist, f, indent=2)
                f.close()
        print(f'The {np.sum(sum_list)} number of samples are distributed among {nofclients} clients')
               
if __name__ == "__main__":

    dataset = 'Cifar10'
    data_path = '/CIFAR-10_data/train_list.json'
    out_path = '/home/data_ssd/CIFAR-10_data/annotations_fed_alpha_0.1_clients_10'
    partition_type ="N_IID"
    clients = 10
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    else:
        pass

    partition_data(dataset, 
                   data_path,
                   out_path, 
                   partition_type, 
                   clients, 
                   dir_alpha=0.1)
    