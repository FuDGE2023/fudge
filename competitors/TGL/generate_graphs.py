import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')

parser.add_argument('--perc_train', type=float)
parser.add_argument('--need_regressor', type=str)
parser.add_argument('--validation_set', type=str)

args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch_geometric
import time
import random
import dgl
import numpy as np
from modules import *
from sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
import networkx as nx
import sys
from regressor import DiscreteRegressor

seed = 12345

device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

torch.manual_seed(seed)
np.random.seed(seed)

module = args.config.split("config/")[1].split(".")[0]
validation_set = True if args.validation_set == "True" else False
end_filename = "val" if validation_set else "test"

def sample(all_nodes: set, existing_edges: dict, positive_edges: dict, n_negative_edges: int, tries=5):
    negative_edges = dict()

    while len(negative_edges) < n_negative_edges:
        if tries <= 0:
            break

        for src in np.random.choice(list(all_nodes), size=n_negative_edges):
            negatives = set(all_nodes)

            if src in existing_edges.keys():
                negatives = negatives - set(existing_edges[src])

            if src in negative_edges.keys():
                negatives = negatives - set(negative_edges[src])

            if src in positive_edges.keys():
                negatives = negatives - positive_edges[src]

            if len(negatives) > 0:
                negative_edges[src] = np.random.choice(list(negatives), size=1)

            if len(negative_edges) >= n_negative_edges:
                tries = 0
                break

        tries -= 1

    return negative_edges

node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)

g, df = load_graph(args.data)

sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
combine_first = False
if 'combine_neighs' in train_param and train_param['combine_neighs']:
    combine_first = True

model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).to(device)
mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None

if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if node_feats is not None:
        node_feats = node_feats.to(device)
    if edge_feats is not None:
        edge_feats = edge_feats.to(device)
    if mailbox is not None:
        mailbox.move_to_gpu()

sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))

if not os.path.isdir('models'):
    os.mkdir('models')

path_saver = 'models/{}_{}.pkl'.format(args.data, module)

model.load_state_dict(torch.load(path_saver))

model.eval()
if sampler is not None:
    sampler.reset()
if mailbox is not None:
    mailbox.reset()
    model.memory_updater.last_updated_nid = None


model.eval()

if validation_set:
    starting_df = df[:train_edge_end]
else:
    starting_df = df[:val_edge_end]

neg_samples = args.eval_neg_samples

graph = nx.DiGraph()

for id, src, dst, ts in zip(starting_df['Unnamed: 0'], starting_df.src,
                                              starting_df.dst, starting_df.time):
    graph.add_edge(src, dst, **{'id': id, 'timestamp': ts})


timestamps = starting_df.time.values
idxs = starting_df['Unnamed: 0'].values

last_ts = starting_df.iloc[-1].time
last_id = starting_df.iloc[-1]['Unnamed: 0']

dataset_extension = '.pt' if args.data.startswith('snapshots') else '.npy'
snap = loadList('../../data/{}'.format(args.data.split("_"+str(args.perc_train))[0] + dataset_extension))
amount = len(snap)

perc_train = args.perc_train
perc_val_test = (1 - perc_train) / 2.0

if type(snap[0]) == list:
    snap = np.array(snap)[:, 4]

train_size = int(amount * perc_train)
val_size = int(amount * perc_val_test)

train_set = snap[0:train_size]
val_set = snap[train_size:train_size + val_size]
test_set = snap[train_size + val_size:]
num_generations = len(val_set) if validation_set else len(test_set)

need_regressor = True if args.need_regressor == "True" else False

regressor = DiscreteRegressor(train_set, need_regressor=need_regressor, seed=seed)
n_operation_list = regressor.sample(num_generations)


all_nodes = set(list(df['src'].values) + list(df['dst'].values))

if not os.path.exists("./generated_data/{}/generated_{}_graphs/TGL/{}_{}/".format(
            args.perc_train, args.data.split("_"+str(args.perc_train))[0], module, end_filename)):
    os.makedirs("./generated_data/{}/generated_{}_graphs/TGL/{}_{}/".format(
            args.perc_train, args.data.split("_"+str(args.perc_train))[0], module, end_filename))

with torch.no_grad():
    for k in tqdm(range(num_generations)):

        existing_edges = dict()
        for s, d in graph.edges():
            if s not in existing_edges.keys():
                existing_edges[s] = {d}
            else:
                existing_edges[s].add(d)

        if validation_set:
            next_edges = np.array(val_set[k].edge_index)
        else:
            next_edges = np.array(test_set[k].edge_index)

        next_edges = set(zip(next_edges[0, :], next_edges[1, :]))

        if validation_set:
            if k > 0:
                now_edges = np.array(val_set[k - 1].edge_index)
                now_edges = set(zip(now_edges[0, :], now_edges[1, :]))
            else:
                now_edges = np.array(train_set[-1].edge_index)
                now_edges = set(zip(now_edges[0, :], now_edges[1, :]))
        else:
            if k > 0:
                now_edges = np.array(test_set[k - 1].edge_index)
                now_edges = set(zip(now_edges[0, :], now_edges[1, :]))
            else:
                now_edges = np.array(val_set[-1].edge_index)
                now_edges = set(zip(now_edges[0, :], now_edges[1, :]))

        real_edges_list = next_edges - now_edges
        real_edges = dict()
        for s, d in real_edges_list:
            if s not in real_edges.keys():
                real_edges[s] = {d}
            else:
                real_edges[s].add(d)
        # print(len(real_edges))

        if validation_set:
            sample_to_generate = 500
        else:
            sample_to_generate = 1000

        negative_edges = sample(all_nodes, existing_edges, real_edges, sample_to_generate, 2)

        # print(len(negative_edges))

        real_edges_list = np.array(list(real_edges_list)).astype(int)
        real_sources = list(real_edges_list[:, 0]) if len(real_edges_list) > 0 else []
        real_destinations = list(real_edges_list[:, 1]) if len(real_edges_list) > 0 else []

        sources = list(negative_edges.keys()) + real_sources
        negative_destinations = np.array(list(negative_edges.values())).reshape(-1)
        negative_samples = np.concatenate((negative_destinations, real_destinations)).astype(int)

        # print(len(sources))
        # print(len(real_sources))


        root_nodes = np.concatenate(
            [sources, np.ones(len(negative_samples), dtype=int), negative_samples]).astype(np.int32)

        ts = np.tile(np.full(shape=len(negative_samples), fill_value=last_ts + 1), 3).astype(np.float32)
        # print(len(ts))

        # print(sources)
        # print(negative_samples)

        if sampler is not None:
            sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
            # exit(0)
        if gnn_param['arch'] != 'identity':
            mfgs = to_dgl_blocks(ret, sample_param['history'])
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts)
        mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])

        # neg samples = 1 because it divides the array as len(array) // neg_samples + 2 -> so it should be len(array) // 3
        pred_pos, pred_neg = model(mfgs, neg_samples=1)

        m = torch.nn.Softmax(dim=0)
        pred_neg = m(pred_neg).cpu().numpy().reshape(-1)

        n_operations = n_operation_list[k]

        idx_sampled = np.random.choice(np.arange(len(pred_neg)), p=pred_neg, size=n_operations, replace=False)
        last_ts += 1

        idx_sampled_destinations = idx_sampled + len(sources) + len(negative_destinations)

        new_sources = []
        new_destinations = []
        for idx in idx_sampled:
            new_source = sources[idx]
            new_destination = negative_samples[idx]

            if not graph.has_edge(new_source, new_destination):
                last_id += 1

                graph.add_edge(new_source, new_destination, **{'id': last_id, 'timestamp': last_ts})

                new_sources.append(new_source)
                new_destinations.append(new_destination)
                timestamps = np.append(timestamps, last_ts)
                idxs = np.append(idxs, last_id)

        pyg = torch_geometric.utils.convert.from_networkx(graph)

        pyg.x = torch.ones((graph.number_of_nodes(), 1))
        edge_index_t = torch.cat((pyg.edge_index[1].reshape(1, pyg.edge_index[1].shape[0]),
                                  pyg.edge_index[0].reshape(1, pyg.edge_index[0].shape[0])),
                                 dim=0)
        pyg.edge_index_t = edge_index_t
        pyg.n_operations = n_operations

        # print("Generation", k, "in", time.time() - t)
        torch.save(pyg, "./generated_data/{}/generated_{}_graphs/TGL/{}_{}/graph_{}.pt".format(
            args.perc_train, args.data.split("_"+str(args.perc_train))[0], module, end_filename, k))


        if mailbox is not None:
            # eid = np.ones(len(sources))#starting_df['Unnamed: 0'].values
            # mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            mem_edge_feats = torch.ones((len(idx_sampled), 1))
            block = None
            if memory_param['deliver_to'] == 'neighbors':
                block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                print("block shape", block.edges()[0].shape)
            mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory,
                                   root_nodes, ts, mem_edge_feats, block, neg_samples=1, generation=True,
                                   sources=np.array(new_sources), destinations=np.array(new_destinations), idx_sources=idx_sampled,
                                   idx_destinations=idx_sampled_destinations)
            mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory,
                                  root_nodes, model.memory_updater.last_updated_ts, neg_samples=1, generation=True, idx_nodes=idx_sampled)



