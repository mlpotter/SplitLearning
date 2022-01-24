from data.relational_table_preprocessor import image_preprocess_dl
import pandas as pd
import numpy as np
import torch
from collections import Counter
import os
from sklearn.datasets import fetch_openml

def load_mnist_flat(args):
    dataset = fetch_openml("mnist_784")

    data = dataset.data
    label_list = dataset.target.astype(int).tolist()

    args.class_num = len(np.unique(label_list))

    [_, _, _, _,_, train_data_local_dict, test_data_local_dict, args.class_num] = relational_table_preprocess_dl(args,
                                                                                                                 data,
                                                                                                                 label_list,
                                                                                                                 test_partition=0.2)
    for key in train_data_local_dict.keys():
        torch.save(train_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_train.pt"))
        print(dict(sorted(dict(Counter(train_data_local_dict[key].dataset[:][1].numpy().tolist())).items())))
        torch.save(test_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_test.pt"))

def load_mnist_image(args):
    dataset = fetch_openml("mnist_784")

    data = dataset.data.reshape(-1, 1, 28, 28)
    label_list = dataset.target.astype(int).tolist()

    args.class_num = len(np.unique(label_list))

    [_, _, _, _,_, train_data_local_dict, test_data_local_dict, args.class_num] = image_preprocess_dl(args,
                                                                                                                 data,
                                                                                                                 label_list,
                                                                                                                 test_partition=0.2)
    for key in train_data_local_dict.keys():
        torch.save(train_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_train.pt"))
        print(dict(sorted(dict(Counter(train_data_local_dict[key].dataset[:][1].numpy().tolist())).items())))
        torch.save(test_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_test.pt"))

if __name__ == '__main__':
    print("Nothing")