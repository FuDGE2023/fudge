import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import torch

def preprocess(data_name, extension='csv'):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  if extension == '.npy':
    data = np.load(data_name, allow_pickle=True)
    first_graph = data[0, 1]

    first_graph_edges = np.array(first_graph.edges())

    u_list = first_graph_edges[:, 0]
    i_list = first_graph_edges[:, 1]

    ts_list = np.zeros(len(first_graph_edges))

    u_list = np.concatenate((u_list, data[1:, 2]), axis=0)
    i_list = np.concatenate((i_list, data[1:, 3]), axis=0)
    ts_list = np.concatenate((ts_list, data[1:, 0]), axis=0)

    # print(data[1:, 7])
    # print(len(data[0]))
    if len(data[0]) == 9:
      label_list = np.zeros(len(first_graph_edges))
      label_temp = np.array([0 if x == 'Add' else 1 for x in data[1:, 7]])
      label_list = np.concatenate((label_list, label_temp), axis=0)
    else:
      label_list = np.zeros((len(data)-1 + len(first_graph_edges)))

    feat_l = np.ones((len(data)-1 + len(first_graph_edges), 1))
    idx_list = np.arange(len(data)-1 + len(first_graph_edges))
    # print(len(idx_list))

  elif extension == '.pt':
    data = np.array(torch.load(data_name))

    ts = 0
    # idx = 0
    u_list = []
    i_list = []
    ts_list = []
    label_list = []
    feat_l = []
    idx_list = []

    for d in data:
      starting_index = len(u_list)
      # print(starting_index, d)
      # print(d[5,1])
      n_operations = d[5, 1] if len(d) == 6 else d[6, 1]
      for i in range(0, n_operations):
        u_list.append(d[1, 1][0, starting_index+i].item())
        i_list.append(d[1, 1][1, starting_index+i].item())
        ts_list.append(ts)
        label_list.append(0)
        if len(d) == 7:
          feat_l.append([float(d[2,1][starting_index+i].item())])
        else:
          feat_l.append([1.0])
        # idx_list.append(ts)

      ts += 1
    idx_list = np.arange(len(u_list))

  else:
    with open(data_name) as f:
      s = next(f)
      for idx, line in enumerate(f):
        e = line.strip().split(',')
        u = int(e[0])
        i = int(e[1])

        ts = float(e[2])
        label = float(e[3])  # int(e[3])

        feat = np.array([float(x) for x in e[4:]])

        u_list.append(u)
        i_list.append(i)
        ts_list.append(ts)
        label_list.append(label)
        idx_list.append(idx)

        feat_l.append(feat)

  print(len(u_list))
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True, extension='.npy'):
  Path("data/").mkdir(parents=True, exist_ok=True)

  PATH = '../../data/{}'.format(data_name+extension)

  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

  print(PATH)

  df, feat = preprocess(PATH, extension=extension)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, 172))

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
parser.add_argument('--extension', type=str)

args = parser.parse_args()

run(args.data, bipartite=args.bipartite, extension=args.extension)

