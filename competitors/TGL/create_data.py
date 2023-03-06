import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import torch

def preprocess(data_name, extension='csv', perc_train=0.7):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  if extension == '.npy':
    data = np.load(data_name, allow_pickle=True)

    perc_val_test = (1 - perc_train) / 2.0

    first_graph = data[0, 1]

    first_graph_edges = np.array(first_graph.edges())

    u_list = first_graph_edges[:, 0]
    i_list = first_graph_edges[:, 1]

    ts_list = np.zeros(len(first_graph_edges))

    u_list = np.concatenate((u_list, data[1:, 2]), axis=0)
    i_list = np.concatenate((i_list, data[1:, 3]), axis=0)
    ts_list = np.concatenate((ts_list, data[1:, 0]), axis=0)

    amount = len(data)# - len(first_graph_edges)

    train_size = int(amount * perc_train)# + len(first_graph_edges)
    val_size = int(amount * perc_val_test)
    test_size = amount - train_size - val_size

    label_train = np.zeros(len(first_graph_edges) + train_size - 1, dtype=int)
    label_val = np.ones(val_size, dtype=int)
    label_test = np.full(test_size, 2, dtype=int)

    label_list = np.concatenate((label_train, label_val), axis=0)
    label_list = np.concatenate((label_list, label_test), axis=0)

    feat_l = np.ones((len(data)-1 + len(first_graph_edges), 1))
    idx_list = np.arange(len(data)-1 + len(first_graph_edges))


  elif extension == '.pt':
    data = np.array(torch.load(data_name))

    ts = 0
    u_list = []
    i_list = []
    ts_list = []
    label_list = []
    feat_l = []
    idx_list = []

    perc_val_test = (1 - perc_train) / 2.0

    amount = len(data)

    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test) + train_size

    count = 0
    n_operations_train = 0
    n_operations_val = 0
    n_operations_test = 0
    for d in data:
      starting_index = len(u_list)
      # print(starting_index, d)
      # print(d[5,1])
      n_operations = d[5, 1] if len(d) == 6 else d[6, 1]

      if count < train_size:
        n_operations_train += n_operations
      elif train_size <= count < val_size:
        n_operations_val += n_operations
      else:
        n_operations_test += n_operations

      count += 1

      for i in range(0, n_operations):
        u_list.append(d[1, 1][0, starting_index+i].item())
        i_list.append(d[1, 1][1, starting_index+i].item())
        ts_list.append(ts)
        if len(d) == 7:
          feat_l.append([float(d[2,1][starting_index+i].item())])
        else:
          feat_l.append([1.0])

      ts += 1

    idx_list = np.arange(len(u_list))
    label_list = np.concatenate((np.zeros(n_operations_train, dtype=int),
                                np.ones(n_operations_val, dtype=int),
                                np.full(n_operations_test, 2, dtype=int)))


  return pd.DataFrame({'': idx_list,
                       'src': u_list,
                       'dst': i_list,
                       'time': ts_list,
                       'ext_roll': label_list}), np.array(feat_l)

def run(data_name, extension='.npy', perc_train=0.7):
  Path("DATA/{}_{}".format(data_name, perc_train)).mkdir(parents=True, exist_ok=True)

  PATH = '../../data/{}'.format(data_name + args.extension)

  OUT_EDGES = './DATA/{}_{}/edges.csv'.format(data_name, perc_train)
  OUT_EDGES_FEAT = './DATA/{}_{}/edge_features.pt'.format(data_name, perc_train)

  print(PATH)

  df, feat = preprocess(PATH, extension=extension, perc_train=perc_train)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = torch.FloatTensor(np.vstack([empty, feat]))

  df.to_csv(OUT_EDGES, index=False)
  torch.save(feat, OUT_EDGES_FEAT)
  print("Dataset created")

parser = argparse.ArgumentParser('Interface for TGL data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--extension', type=str)
parser.add_argument('--perc_train', type=float, default=0.7)

args = parser.parse_args()

run(args.data, extension=args.extension, perc_train=args.perc_train)