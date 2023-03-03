import pandas as pd
from log import *
from parser import *
from eval import *
from utils import *
from train import *
from module import NAT
import resource
import torch.nn as nn
import statistics
import networkx as nx
import torch_geometric
import gc

sys.path.append("/home/mungari/Desktop/evograph/components/") # todo relative

from regressor import DiscreteRegressor

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


def loadList(filename):
    if "synthetic" in filename:
        # the filename should mention the extension 'npy'
        tempNumpyArray = np.load(filename, allow_pickle=True)
        snap = tempNumpyArray.tolist()
    else:
        snap = torch.load(filename)

    return snap


def get_split(dataset_name, train_percentage, g_df):
    if dataset_name.startswith('synthetic'):
        dataset_extension = '.npy'
    else:
        dataset_extension = '.pt'

    snap = loadList('/mnt/nas/liguori/evograph/data/{}'.format(dataset_name + dataset_extension))

    amount = len(snap)
    perc_train = train_percentage
    perc_val_test = (1 - perc_train) / 2.0

    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test)

    train_set = snap[0:train_size]
    val_set = snap[train_size:train_size + val_size]
    test_set = snap[train_size + val_size:]

    if dataset_name.startswith('synthetic'):
        train_size_regression = train_size + len(g_df[g_df.ts.values == 0]) - 1
        val_size_regression = train_size_regression + val_size
    else:
        train_size_regression = 0
        for d in train_set:
            n_operations = d.n_operation
            train_size_regression += n_operations

        val_size_regression = train_size_regression
        for d in val_set:
            n_operations = d.n_operation
            val_size_regression += n_operations

    return train_size_regression, val_size_regression


args, sys_argv = get_args()

datasets = args.data.split("-")
train_percentages = args.train_percentage.split("-")
ngh_dims = args.ngh_dim.split("-")
n_degrees = [x.split(",") for x in args.n_degree.split("-")]
validation_set = True if args.validation_set == "True" else False
end_filename = "val" if validation_set else "test"

print(datasets)
print(train_percentages)
print(ngh_dims)
print(n_degrees)

for train_percentage in train_percentages:
  train_percentage = float(train_percentage)
  for dataset in datasets:
    for ngh_dim in ngh_dims:
      ngh_dim = int(ngh_dim)
      for n_degree in n_degrees:

        gc.collect()
        n_degree = [int(x) for x in n_degree]

        factors = "{}_{}_{}".format(ngh_dim, n_degree[0], n_degree[1])

        BATCH_SIZE = args.bs
        NUM_NEIGHBORS = n_degree
        NUM_EPOCH = args.n_epoch
        ATTN_NUM_HEADS = args.attn_n_head
        DROP_OUT = args.drop_out
        DATA = dataset
        NUM_HOP = args.n_hop
        LEARNING_RATE = args.lr
        POS_DIM = args.pos_dim
        TOLERANCE = args.tolerance
        VERBOSITY = args.verbosity
        SEED = 12345  # args.seed
        TIME_DIM = args.time_dim
        REPLACE_PROB = args.replace_prob
        SELF_DIM = args.self_dim
        NGH_DIM = ngh_dim
        need_regressor = True if args.need_regressor == "True" else False
        # train_percentage = train_percentage#args.train_percentage
        assert (NUM_HOP < 3)  # only up to second hop is supported
        set_random_seed(SEED)
        logger, get_checkpoint_path, get_ngh_store_path, get_self_rep_path, get_prev_raw_path, best_model_path, \
        best_model_ngh_store_path = set_up_logger(args, sys_argv, dataset, n_degree, ngh_dim, train_percentage)

        # Load data and sanity check
        g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
        src_l = g_df.u.values.astype(int)
        tgt_l = g_df.i.values.astype(int)
        e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
        n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))

        e_idx_l = g_df.idx.values.astype(int)
        e_idx_l = np.zeros_like(e_idx_l)
        label_l = g_df.label.values
        ts_l = g_df.ts.values

        max_idx = max(src_l.max(), tgt_l.max())

        assert (np.unique(np.stack([src_l, tgt_l])).shape[
                    0] == max_idx)  # all nodes except node 0 should appear and be compactly indexed
        assert (n_feat.shape[0] == max_idx + 1)  # the nodes need to map one-to-one to the node feat matrix

        # split and pack the data by generating valid train/val/test mask according to the "mode"
        val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))


        total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
        num_total_unique_nodes = len(total_node_set)

        # split data according to the mask
        train_size_regression, val_size_regression = get_split(DATA, train_percentage, g_df)

        train_src_l, train_tgt_l, train_ts_l, train_e_idx_l, train_label_l = src_l[:train_size_regression], \
                                                                             tgt_l[:train_size_regression], \
                                                                             ts_l[:train_size_regression], \
                                                                             e_idx_l[:train_size_regression], \
                                                                             label_l[:train_size_regression]
        val_src_l, val_tgt_l, val_ts_l, val_e_idx_l, val_label_l = src_l[train_size_regression:val_size_regression], \
                                                                   tgt_l[train_size_regression:val_size_regression], \
                                                                   ts_l[train_size_regression:val_size_regression], \
                                                                   e_idx_l[train_size_regression:val_size_regression], \
                                                                   label_l[train_size_regression:val_size_regression]
        test_src_l, test_tgt_l, test_ts_l, test_e_idx_l, test_label_l = src_l[val_size_regression:], \
                                                                        tgt_l[val_size_regression:], \
                                                                        ts_l[val_size_regression:], \
                                                                        e_idx_l[val_size_regression:], \
                                                                        label_l[val_size_regression:]

        train_data = train_src_l, train_tgt_l, train_ts_l, train_e_idx_l, train_label_l
        val_data = val_src_l, val_tgt_l, val_ts_l, val_e_idx_l, val_label_l
        train_val_data = (train_data, val_data)

        print(len(train_src_l), len(val_src_l), len(test_src_l))

        # exit(0)


        # multiprocessing memory setting
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (200 * args.bs, rlimit[1]))

        # model initialization
        # device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
        device = torch.device('cpu')
        feat_dim = n_feat.shape[1]
        e_feat_dim = e_feat.shape[1]
        time_dim = TIME_DIM
        model_dim = feat_dim + e_feat_dim + time_dim
        hidden_dim = e_feat_dim + time_dim
        num_raw = 3
        memory_dim = NGH_DIM + num_raw
        num_neighbors = [1]
        for i in range(NUM_HOP):
            num_neighbors.extend([int(NUM_NEIGHBORS[i])])
        # num_neighbors.extend([int(n) for n in NUM_NEIGHBORS]) # the 0-hop neighborhood has only 1 node

        total_start = time.time()
        nat = NAT(n_feat, e_feat, memory_dim, max_idx + 1, time_dim=TIME_DIM, pos_dim=POS_DIM, n_head=ATTN_NUM_HEADS,
                  num_neighbors=num_neighbors, dropout=DROP_OUT,
                  linear_out=args.linear_out, get_checkpoint_path=get_checkpoint_path,
                  get_ngh_store_path=get_ngh_store_path, get_self_rep_path=get_self_rep_path,
                  get_prev_raw_path=get_prev_raw_path, verbosity=VERBOSITY,
                  n_hops=NUM_HOP, replace_prob=REPLACE_PROB, self_dim=SELF_DIM, ngh_dim=NGH_DIM, device=device)
        nat.to(device)
        nat.reset_store()
        nat.reset_self_rep()

        nat.load_state_dict(torch.load(best_model_path))

        nat.set_seed(SEED)

        with torch.no_grad():
            nat = nat.eval()

            if not os.path.exists(
                    "/mnt/nas/mungari/evograph/generated_data/{}/generated_{}_graphs/NAT/{}_{}".format(train_percentage,
                                                                                                    DATA, factors, end_filename)):
                os.makedirs("/mnt/nas/mungari/evograph/generated_data/{}/generated_{}_graphs/NAT/{}_{}".format(train_percentage,
                                                                                                            DATA,
                                                                                                            factors, end_filename))
            graph = nx.DiGraph()

            for source, destination, label, id, ts in zip(train_src_l, train_tgt_l,
                                                          train_label_l, train_e_idx_l, train_ts_l):
                # source -= 1
                # destination -= 1
                if label == 0.0:
                    graph.add_edge(source, destination, **{'id': id, 'timestamp': ts})
                else:
                    graph.remove_edge(source, destination)

            if not validation_set:
                for source, destination, label, id, ts in zip(val_src_l, val_tgt_l,
                                                              val_label_l, val_e_idx_l, val_ts_l):
                    # source -= 1
                    # destination -= 1
                    if label == 0.0:
                        graph.add_edge(source, destination, **{'id': id, 'timestamp': ts})
                    else:
                        graph.remove_edge(source, destination)

            n_neighbors = 10  # default for TGN

            edges = np.array(list(graph.edges(data=True)))

            sources = edges[:, 0].astype(int)
            destinations = edges[:, 1].astype(int)
            timestamps = np.array([x['timestamp'] for x in edges[:, 2]])
            idxs = np.array([x['id'] for x in edges[:, 2]])

            last_ts = timestamps[-1]
            last_id = idxs[-1]

            dataset_extension = '.pt' if DATA.startswith('snapshots') else '.npy'
            snap = loadList('/mnt/nas/liguori/evograph/data/{}'.format(DATA + dataset_extension))
            amount = len(snap)
            print(amount)

            perc_train = train_percentage
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

            regressor = DiscreteRegressor(train_set, need_regressor=need_regressor, seed=SEED)
            n_operation_list = regressor.sample(num_generations)

            # num_generations = 1
            all_nodes = total_node_set


            for k in tqdm(range(num_generations)):
                t = time.time()

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
                        now_edges = np.array(test_set[k - 1].edge_index)
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
                    sample_to_generate = 500  # len(real_edges)*50 if len(real_edges) > 0 else 50
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

                pos_prob, neg_prob = nat.contrast(np.array(sources), np.ones(len(negative_samples), dtype=int), negative_samples,
                                                  np.full(shape=len(negative_samples),
                                                          fill_value=last_ts + 1), idxs[-len(negative_samples):])
                # exit(0)

                # pos_prob, neg_prob = tgn.compute_edge_probabilities(sources,
                #                                                     np.ones(len(negative_samples), dtype=int),
                #                                                     negative_samples,
                #                                                     np.full(shape=len(negative_samples),
                #                                                             fill_value=last_ts + 1),
                #                                                     idxs[-len(negative_samples):], n_neighbors,
                #                                                     generation=True, num_known_nodes=num_known_nodes)

                neg_prob = neg_prob.cpu().numpy().reshape(-1)
                neg_prob /= neg_prob.sum()
                # print(neg_prob)

                n_operations = n_operation_list[k]

                idx_sampled = np.random.choice(np.arange(len(neg_prob)), p=neg_prob, size=n_operations, replace=False)
                last_ts += 1

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

                    else:
                        print("here")

                pyg = torch_geometric.utils.convert.from_networkx(graph)
                pyg.x = torch.ones((graph.number_of_nodes(), 1))
                edge_index_t = torch.cat((pyg.edge_index[1].reshape(1, pyg.edge_index[1].shape[0]),
                                          pyg.edge_index[0].reshape(1, pyg.edge_index[0].shape[0])),
                                         dim=0)
                pyg.edge_index_t = edge_index_t
                pyg.n_operations = n_operations

                # generated_graphs.append(pyg)
                # tgn.memory.restore_memory(memory_backup)
                # print(pyg)

                # print("Generation", k, "in", time.time() - t)
                torch.save(pyg, "/mnt/nas/mungari/evograph/generated_data/{}/generated_{}_graphs/NAT/{}_{}/graph_{}.pt".format(
                    train_percentage, DATA, factors, end_filename, k))

        del sources, train_set, val_set, test_set, destination, timestamps, idxs
        del src_l, tgt_l, e_idx_l, label_l, ts_l


