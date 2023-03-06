import math
import logging
import time
import sys
import argparse
import os

import networkx as nx
import torch
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path

import torch_geometric

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, Data, loadList

from regressor import DiscreteRegressor

seed = 12345

torch.manual_seed(seed)
np.random.seed(seed)

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


### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised generation')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')

parser.add_argument('--need_regressor', type=str, default="True")
parser.add_argument('--perc_train', type=float)
parser.add_argument('--factor', type=str)

parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')

parser.add_argument('--validation_set', type=str)


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

factor = args.factor
perc_train = args.perc_train
dataset_name = args.data
validation_set = True if args.validation_set == "True" else False


BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = dataset_name
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

args.embedding_module = factor

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{dataset_name}_{factor}_{perc_train}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{dataset_name}-{epoch}.pth'

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data  = get_data(dataset_name,
                              different_new_nodes_between_val_and_test=args.different_new_nodes,
                               randomize_features=args.randomize_features, perc_train=perc_train)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device_string = 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

best_model_path = "saved_models/{}_{}_{}.pth".format(dataset_name, factor, perc_train)

# Initialize Model
tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
        edge_features=edge_features, device=device,
        n_layers=NUM_LAYER,
        n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
        message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
        memory_update_at_start=not args.memory_update_at_end,
        embedding_module_type=args.embedding_module,
        message_function=args.message_function,
        aggregator_type=args.aggregator,
        memory_updater_type=args.memory_updater,
        n_neighbors=NUM_NEIGHBORS,
        mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
        mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
        use_source_embedding_in_message=args.use_source_embedding_in_message,
        dyrep=args.dyrep)
tgn = tgn.to(device)

tgn.load_state_dict(torch.load(best_model_path))
tgn.eval()

# num_generations = test_data.n_interactions

end_filename = "val" if validation_set else "test"

if not os.path.exists("./generated_data/{}/generated_{}_graphs/TGN/{}_{}".format(perc_train, dataset_name, factor, end_filename)):
    os.makedirs("./generated_data/{}/generated_{}_graphs/TGN/{}_{}".format(perc_train, dataset_name, factor, end_filename))

with torch.no_grad():
    tgn = tgn.eval()

    graph = nx.DiGraph()

    for source, destination, label, id, ts in zip(train_data.sources, train_data.destinations,
                                                  train_data.labels, train_data.edge_idxs, train_data.timestamps):
        if label == 0.0:
            graph.add_edge(source, destination, **{'id': id, 'timestamp': ts})
        else:
            graph.remove_edge(source, destination)

    if not validation_set:
        for source, destination, label, id, ts in zip(val_data.sources, val_data.destinations,
                                                      val_data.labels, val_data.edge_idxs, val_data.timestamps):
            if label == 0.0:
                graph.add_edge(source, destination, **{'id': id, 'timestamp': ts})
            else:
                graph.remove_edge(source, destination)

    generated_graphs = []
    n_neighbors = 10 # default for TGN

    edges = np.array(list(graph.edges(data=True)))

    sources = edges[:, 0].astype(int)
    destinations = edges[:, 1].astype(int)
    timestamps = np.array([x['timestamp'] for x in edges[:, 2]])
    idxs = np.array([x['id'] for x in edges[:, 2]])

    last_ts = timestamps[-1]
    last_id = idxs[-1]

    dataset_extension = '.pt' if DATA.startswith('snapshots') else '.npy'
    snap = loadList('../../data/{}'.format(dataset_name + dataset_extension))
    amount = len(snap)

    perc_val_test = (1 - perc_train) / 2.0

    if type(snap[0]) == list:
        snap = np.array(snap)[:, 4]

    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test)

    train_set = snap[0:train_size]
    val_set = snap[train_size:train_size + val_size].copy()
    test_set = snap[train_size + val_size:].copy()

    if validation_set:
        num_generations = len(val_set)
    else:
        num_generations = len(test_set)

    need_regressor = True if args.need_regressor == "True" else False

    regressor = DiscreteRegressor(train_set, need_regressor=need_regressor, seed=seed)
    n_operation_list = regressor.sample(num_generations)

    # num_generations = 1
    all_nodes = full_data.unique_nodes

    for k in tqdm(range(num_generations)):
        t = time.time()

        data = Data(np.array(sources), np.array(destinations), timestamps, idxs, np.zeros(len(sources)))
        ngh_finder = get_neighbor_finder(data, uniform=False)
        num_known_nodes = len(ngh_finder.node_to_neighbors)
        tgn.set_neighbor_finder(ngh_finder)
        tgn.embedding_module.neighbor_finder = ngh_finder

        # size = len(sources)

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

        # reindexing and taking the unique
        next_edges = set(zip(next_edges[0, :] + 1, next_edges[1, :] + 1))

        if validation_set:
            if k > 0:
                now_edges = np.array(val_set[k - 1].edge_index)
                now_edges = set(zip(now_edges[0, :] + 1, now_edges[1, :] + 1))
            else:
                now_edges = np.array(train_set[-1].edge_index)
                now_edges = set(zip(now_edges[0, :] + 1, now_edges[1, :] + 1))
        else:
            if k > 0:
                now_edges = np.array(test_set[k-1].edge_index)
                now_edges = set(zip(now_edges[0, :] + 1, now_edges[1, :] + 1))
            else:
                now_edges = np.array(val_set[-1].edge_index)
                now_edges = set(zip(now_edges[0, :] + 1, now_edges[1, :] + 1))


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
        t = time.time()
        negative_edges = sample(all_nodes, existing_edges, real_edges, sample_to_generate, 2)

        # print(len(negative_edges))

        real_edges_list = np.array(list(real_edges_list)).astype(int)
        real_sources = list(real_edges_list[:, 0]) if len(real_edges_list) > 0 else []
        real_destinations = list(real_edges_list[:, 1]) if len(real_edges_list) > 0 else []

        sources = list(negative_edges.keys()) + real_sources
        negative_destinations = np.array(list(negative_edges.values())).reshape(-1)
        negative_samples = np.concatenate((negative_destinations, real_destinations)).astype(int)

        # We update nodes' memories after the sampling, in this case "idxs" are not used
        pos_prob, neg_prob = tgn.compute_edge_probabilities(sources,
                                                            np.ones(len(negative_samples), dtype=int),
                                                            negative_samples,
                                                            np.full(shape=len(negative_samples), fill_value=last_ts+1),
                                                            idxs[-len(negative_samples):], n_neighbors,
                                                            generation=True, num_known_nodes=num_known_nodes)

        neg_prob = neg_prob.cpu().numpy().reshape(-1)
        neg_prob /= neg_prob.sum()

        n_operations = n_operation_list[k]
        # print("n operations", n_operations)
        idx_sampled = np.random.choice(np.arange(len(neg_prob)), p=neg_prob, size=n_operations, replace=False)
        last_ts += 1

        new_sources = []
        new_destinations = []
        new_ts = []
        new_idxs = []
        for idx in idx_sampled:
            new_source = sources[idx]
            new_destination = negative_samples[idx]

            if not graph.has_edge(new_source, new_destination):
                last_id += 1

                graph.add_edge(new_source, new_destination, **{'id': last_id, 'timestamp': last_ts})

                sources = np.append(sources, new_source)
                destinations = np.append(destinations, new_destination)
                timestamps = np.append(timestamps, last_ts)
                idxs = np.append(idxs, last_id)


        tgn.update_memory_after_generation(sources[-n_operations:], destinations[-n_operations:],
                                           timestamps[-n_operations:], idxs[-n_operations:])

        pyg = torch_geometric.utils.convert.from_networkx(graph)
        pyg.x = torch.ones((graph.number_of_nodes(), 1))
        edge_index_t = torch.cat((pyg.edge_index[1].reshape(1, pyg.edge_index[1].shape[0]),
                                  pyg.edge_index[0].reshape(1, pyg.edge_index[0].shape[0])),
                                 dim=0)
        pyg.edge_index_t = edge_index_t
        pyg.n_operations = n_operations

        torch.save(pyg, "./generated_data/{}/generated_{}_graphs/TGN/{}_{}/graph_{}.pt".format(perc_train, dataset_name, factor, end_filename, k))

print("Graphs saved")



