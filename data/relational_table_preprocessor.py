from data.noniid_partition import non_iid_partition_with_dirichlet_distribution
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset,ConcatDataset
import random

def relational_table_preprocess_dl(args,data,label_list,test_partition=0.2):

    label_list = torch.LongTensor(label_list)
    data = torch.FloatTensor(data)

    net_dataidx_map = non_iid_partition_with_dirichlet_distribution(label_list=label_list,
                                                                    client_num=args.client_num_in_total,
                                                                    classes=args.class_num,
                                                                    alpha=args.partition_alpha,
                                                                    task='task')
    # wish to set up proper configuration of data correctly as seen used in FedML

    train_data_local_dict = {}
    test_data_local_dict = {}
    train_data_local_num_dict = {}
    test_partition = test_partition
    train_data_global = []
    test_data_global = []
    train_data_num = 0
    test_data_num = 0

    for key,client_data in net_dataidx_map.items():
        N_client = len(client_data)
        N_train = int(N_client*(1-test_partition))

        train_data_local_num_dict[key] = N_train
        random.shuffle(client_data)


        train,train_label = data[client_data[:N_train],:],label_list[client_data[:N_train]]
        train_dataset = TensorDataset(train,train_label)
        train_data_local_dict[key] = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

        test, test_label = data[client_data[N_train:], :], label_list[client_data[N_train:]]
        test_dataset = TensorDataset(test, test_label)
        test_data_local_dict[key] = DataLoader(test_dataset, batch_size=args.batch_size)

        train_data_global.append(train_dataset)
        test_data_global.append(test_dataset)

        train_data_num += N_train
        test_data_num += (N_client-N_train)

    train_data_global = DataLoader(ConcatDataset(train_data_global),batch_size=args.batch_size)
    test_data_global = DataLoader(ConcatDataset(test_data_global),batch_size=args.batch_size)

    return [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args.class_num]

def image_preprocess_dl(args,data,label_list,test_partition=0.2):

    label_list = torch.LongTensor(label_list)
    data = torch.FloatTensor(data)

    net_dataidx_map = non_iid_partition_with_dirichlet_distribution(label_list=label_list,
                                                                    client_num=args.client_num_in_total,
                                                                    classes=args.class_num,
                                                                    alpha=args.partition_alpha,
                                                                    task='task')
    # wish to set up proper configuration of data correctly as seen used in FedML

    train_data_local_dict = {}
    test_data_local_dict = {}
    train_data_local_num_dict = {}
    test_partition = test_partition
    train_data_global = []
    test_data_global = []
    train_data_num = 0
    test_data_num = 0

    for key,client_data in net_dataidx_map.items():
        N_client = len(client_data)
        N_train = int(N_client*(1-test_partition))

        train_data_local_num_dict[key] = N_train
        random.shuffle(client_data)


        train,train_label = data[client_data[:N_train]],label_list[client_data[:N_train]]
        train_dataset = TensorDataset(train,train_label)
        train_data_local_dict[key] = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

        test, test_label = data[client_data[N_train:]], label_list[client_data[N_train:]]
        test_dataset = TensorDataset(test, test_label)
        test_data_local_dict[key] = DataLoader(test_dataset, batch_size=args.batch_size)

        train_data_global.append(train_dataset)
        test_data_global.append(test_dataset)

        train_data_num += N_train
        test_data_num += (N_client-N_train)

    train_data_global = DataLoader(ConcatDataset(train_data_global),batch_size=args.batch_size)
    test_data_global = DataLoader(ConcatDataset(test_data_global),batch_size=args.batch_size)

    return [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args.class_num]