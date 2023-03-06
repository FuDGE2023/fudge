import sys
import numpy as np
import torch
import networkx as nx
from time import time
import torch_geometric
import os
from tqdm import tqdm
import numpy as np
from utils import loadList
from torch_geometric.utils.convert import to_networkx
import argparse


def graph_statistics(G, torch_g=False):
    if not torch_g:
        G = to_networkx(G, to_undirected=False)

    print('Max diameter in G: ')
    diameter = max([max(j.values()) for (i, j) in nx.shortest_path_length(G)])
    print(diameter)
    print('-----' * 20)

    print('Average coef. clustering: ', nx.average_clustering(G))

parser = argparse.ArgumentParser('Interface for TGL framework')
parser.add_argument('--data', type=str)
parser.add_argument('--extension', type=str)
parser.add_argument('--perc_train', type=float)
parser.add_argument('--factor', type=str)
parser.add_argument('--validation_set', type=str)


args = parser.parse_args()

dataset = args.data
dataset_extension = args.extension
perc_train = args.perc_train
factor = args.factor
end_filename = "val" if args.validation_set == "True" else "test"

print(sys.argv)

snap=loadList('../../data/{}'.format(dataset+dataset_extension))

seed = 12345

amount = len(snap)

perc_val_test = (1 - perc_train) / 2.0

train_size = int(amount * perc_train)
val_size = int(amount * perc_val_test)

train_set = snap[0:train_size]
val_set = snap[train_size:train_size + val_size]
test_set = snap[train_size + val_size: ]

# print(test_set[0])
if type(snap[0]) == list:
    snap = np.array(snap)[:, 4]
    train_set = snap[0:train_size]
    val_set = snap[train_size:train_size + val_size]
    test_set = snap[train_size + val_size:]

G_real = test_set[-1]

G_pred = torch.load("./generated_data/{}/generated_{}_graphs/TGL/{}_{}/graph_{}.pt"
                    .format(perc_train, dataset, factor, end_filename, len(test_set)-1))

print("G real")
graph_statistics(G_real)
print("G pred")
graph_statistics(G_pred)