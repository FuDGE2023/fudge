import argparse
import os
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
from itertools import chain
from scipy.special import softmax

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
parser.add_argument('--sample_size', type=int)
parser.add_argument('--seed', type=int)

args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
seed = args.seed
sample_size = args.sample_size

device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.use_deterministic_algorithms = True
torch.backends.cudnn.benchmark = False

module = args.config.split("config/")[1].split(".")[0]
validation_set = True if args.validation_set == "True" else False
end_filename = "val" if validation_set else "test"

def sample(all_nodes: set, existing_edges: dict, positive_edges: dict, n_negative_edges: int, tries=5, all=False, seed=123456):
    negative_edges = dict()
    np.random.seed(seed)

    if all:
        for src in range(len(all_nodes)):
            negatives = set(all_nodes)

            if src in existing_edges.keys():
                negatives = negatives - set(existing_edges[src])

            if src in negative_edges.keys():
                negatives = negatives - set(negative_edges[src])

            #if src in positive_edges.keys():
            #    negatives = negatives - positive_edges[src]

            if len(negatives) > 0:
                if src not in negative_edges.keys():
                    negative_edges[src] = []

                negative_edges[src].extend(list(negatives))
    else:

        negative_edges_cont = 0

        while negative_edges_cont < n_negative_edges:
            if tries <= 0:
                break

            for src in np.random.choice(list(all_nodes), size=n_negative_edges):
                negatives = set(all_nodes)

                if src in existing_edges.keys():
                    negatives = negatives - set(existing_edges[src])

                if src in negative_edges.keys():
                    negatives = negatives - set(negative_edges[src])

                #if src in positive_edges.keys():
                #    negatives = negatives - positive_edges[src]

                if len(negatives) > 0:
                    if src not in negative_edges.keys():
                        negative_edges[src] = []

                    negative_edges[src].extend(np.random.choice(list(negatives), size=1))
                    negative_edges_cont += 1

                if negative_edges_cont >= n_negative_edges:
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

model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, device, combined=combine_first).to(device)
mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge, device) if memory_param['type'] != 'none' else None

if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if node_feats is not None:
        node_feats = node_feats.to(device)
    if edge_feats is not None:
        edge_feats = edge_feats.to(device)
    if mailbox is not None and device_string != "cpu":
        mailbox.move_to_gpu()

sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))
# neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

if not os.path.isdir('models'):
    os.mkdir('models')

path_saver = 'models/{}_{}.pkl'.format(args.data, module)

model.load_state_dict(torch.load(path_saver, map_location=device))

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

if not os.path.exists("./generated_data/{}/generated_{}_graphs/TGL/{}/{}/{}_{}/".format(
            args.perc_train, args.data.split("_"+str(args.perc_train))[0], seed, sample_size, module, end_filename)):
    os.makedirs("./generated_data/{}/generated_{}_graphs/TGL/{}/{}/{}_{}/".format(
            args.perc_train, args.data.split("_"+str(args.perc_train))[0], seed, sample_size, module, end_filename))

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

        # print(len(all_nodes))

        sample_to_generate = sample_size
        sample_to_generate_tmp = sample_to_generate

        real_edges_list = np.array(list(real_edges_list)).astype(int)

        if n_operation_list[k] > sample_to_generate:
            sample_to_generate_tmp = n_operation_list[k] * 2

        negative_edges = sample(all_nodes, existing_edges, real_edges, sample_to_generate_tmp, 5,
                                seed=seed)

        # print(len(negative_edges))

        sources = []
        for src in list(negative_edges.keys()):
            sources.extend([src] * len(negative_edges[src]))

        negative_samples = np.array(list(chain(*negative_edges.values())))

        root_nodes = np.concatenate(
            [sources, np.ones(len(negative_samples), dtype=int), negative_samples]).astype(np.int32)

        ts = np.tile(np.full(shape=len(negative_samples), fill_value=last_ts + 1), 3).astype(np.float32)
        # print(len(ts))

        # print(sources)

        # t_temp = time.time()
        if sampler is not None:
            sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
            # exit(0)
        cuda = False if device.type == 'cpu' else True
        if gnn_param['arch'] != 'identity':
            mfgs = to_dgl_blocks(ret, sample_param['history'], device, cuda=cuda)
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts, device, cuda=cuda)
        mfgs = prepare_input(mfgs, node_feats, edge_feats, device, combine_first=combine_first)
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])

        pred_pos, pred_neg = model(mfgs, neg_samples=1)

        p_flatten = pred_neg.flatten().cpu().numpy()
        p_flatten = softmax(p_flatten)

        n_operations = n_operation_list[k]

        idx_sampled = np.random.choice(np.arange(len(p_flatten)), p=p_flatten, size=n_operations, replace=False)

        last_ts += 1

        idx_sampled_destinations = idx_sampled + len(sources) + len(negative_samples)

        new_sources = []
        new_destinations = []

        for idx in idx_sampled:
            new_source = sources[idx]
            new_destination = negative_samples[idx]
            # print("NEW", new_source, new_destination)

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

        torch.save(pyg, "./generated_data/{}/generated_{}_graphs/TGL/{}/{}/{}_{}/graph_{}.pt".format(
            args.perc_train, args.data.split("_"+str(args.perc_train))[0], seed, sample_size, module, end_filename, k))

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
                                   sources=np.array(new_sources), destinations=np.array(new_destinations),
                                   idx_sources=idx_sampled,
                                   idx_destinations=idx_sampled_destinations)
            mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory,
                                  root_nodes, model.memory_updater.last_updated_ts, neg_samples=1, generation=True,
                                  idx_nodes=idx_sampled)

