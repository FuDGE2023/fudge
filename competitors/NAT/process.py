import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import sys
import torch
from pathlib import Path


def get_one_hot(valid_len, tot_len):
    return np.concatenate((np.eye(valid_len), np.zeros((valid_len, tot_len-valid_len))), axis=-1)


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
        idx_list.append(ts)

      ts += 1

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


def reindex(df, jodie_data):
    if jodie_data:
        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df = df.copy()
        new_df.i = new_i

        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df = df.copy()        
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1    
    return new_df

def to_csv(data_name):
    a = open('./processed/{}.txt'.format(data_name), "r")
    f = open('./processed/{}.csv'.format(data_name), "w")

    rehash = {}

    counter = 0
    counter = 0
    min_ts = 0
    max_ts = 0
    f.write("user_id,item_id,timestamp,state_label,comma_separated_list_of_features\n")

    u_s = []
    i_s = []
    t_s = []
    for x in a:
        edge = x.strip().split(' ')
        u = edge[0]
        i = edge[1]
        t = float(edge[2])
        if min_ts == 0:
            min_ts = t
            max_ts = t
        if t < min_ts:
            min_ts = t
        if t > max_ts:
            max_ts = t
        u_s.append(u)
        i_s.append(i)
        t_s.append(t)
        #

    order = np.argsort(t_s)

    for o in order:
        u = u_s[o]
        i = i_s[o]
        t = t_s[o]
        # # if t < max_ts - 1 * 365*24*60*60:
        #   continue
        # t -= min_ts
        if u not in rehash:
            rehash[u] = counter
            counter += 1
        u_new = rehash[u]
        if i not in rehash:
            rehash[i] = counter
            counter += 1
        i_new = rehash[i]
        f.write(','.join([str(u_new), str(i_new), str(t), "0", "0"]) + '\n') #  + ", 0" * 171
        # f.write(','.join([str(u), str(i), str(t), "0", "0"]) + '\n')

    # for x in a:
    #   edge = x.strip().split(' ')
    #   f.write(','.join(edge) + ',0,0\n')
    f.close()

def run(args):
    data_name = args.data
    # to_csv(data_name, args.extension)
    Path("processed/").mkdir(parents=True, exist_ok=True)

    node_edge_feat_dim = args.node_edge_feat_dim
    # PATH = './processed/{}.csv'.format(data_name)
    PATH = '../../data/{}'.format(data_name + args.extension)
    OUT_DF = './processed/ml_{}.csv'.format(data_name)
    OUT_FEAT = './processed/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './processed/ml_{}_node.npy'.format(data_name)
    
    # jodie_data = data_name in ['wikipedia', 'reddit', 'mooc']
    print('preprocess {} dataset ...')
    df, feat = preprocess(PATH, args.extension)
    new_df = reindex(df, jodie_data=False)

    if not args.one_hot_node:
        empty = np.zeros(feat.shape[1])[np.newaxis, :]
        feat = np.vstack([empty, feat])
        max_idx = max(new_df.u.max(), new_df.i.max())
        rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
        # if 'socialevolve' in data_name:
        #     feat = np.zeros((feat.shape[0], node_edge_feat_dim))
        #     rand_feat = np.zeros((rand_feat.shape[0], node_edge_feat_dim))
        print('node feature shape:', rand_feat.shape)
        print('edge feature shape:', feat.shape)
    else:
        # (obsolete branch) TODO: still problematic, add one-hot encoding if possible
        empty = np.zeros(feat.shape[1])[np.newaxis, :]
        feat = np.vstack([empty, feat])
        feat = np.concatenate()
        max_idx = max(new_df.u.max(), new_df.i.max())        
        rand_feat = get_one_hot(max_idx+1, feat.shape[1])
        
        print('one-hot node feature:', rand_feat.shape)        
    print(feat.shape)
    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)


parser = argparse.ArgumentParser('Interface for propressing csv source data for TGAT framework')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--extension', type=str)
parser.add_argument('--node_edge_feat_dim', default=172, help='number of dimensions for 0-padded node and edge features')
parser.add_argument('--one-hot-node', type=bool, default=False,
                   help='using one hot embedding for node (which means inductive learning is impossible though)')
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
    
run(args)