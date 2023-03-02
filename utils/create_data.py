import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import pandas as pd

from utils.utils import create_graph_star, create_graph_star_two_nodes
import numpy as np


def create_data_bitcoin_edge_weights(file_name, window_size, window_shift, is_data=True, path_generation=None,
                                     path_regression=None, perc_train=0.7):

    def create_mapping(df):
        timestamp = sorted(df['timestamp'].unique())
        node_mappings = {}
        node_idx = 0

        for ts in timestamp:
            sub_group = df[df['timestamp'] == ts]

            for g in sub_group.iterrows():
                source = g[1]['source']
                dst = g[1]['dest']

                if source not in node_mappings.keys():
                    node_mappings[source] = node_idx
                    node_idx += 1

                if dst not in node_mappings.keys():
                    node_mappings[dst] = node_idx
                    node_idx += 1
        return node_mappings

    def create_snap(df, node_mappings):
        timestamp = sorted(df['timestamp'].unique())
        edge_index, edge_index_t = [[], []], [[], []]
        snapshot = {}
        count = 0
        x = None
        edge_weights = None
        nodes_list = []

        count_occur = 0

        for ts in tqdm(timestamp, total=len(timestamp)):
            snap = []

            sub_group = df[df['timestamp'] == ts]

            n_operation = len(sub_group)

            for _, row in sub_group.iterrows():
                source = row['source']
                dst = row['dest']
                rating = row['rating']

                source = node_mappings[source]
                dst = node_mappings[dst]

                if source not in nodes_list and dst not in nodes_list:
                    count_occur += 1

                # Create edge index
                edge_index[0].extend([source])
                edge_index[1].extend([dst])

                # Create edge index T
                edge_index_t[0].extend([dst])
                edge_index_t[1].extend([source])

                # Create features
                n_nodes = max(np.concatenate((edge_index[0], edge_index[1]))) + 1

                if count == 0:
                    edge_weights = torch.tensor([rating])
                    x = torch.ones(n_nodes, 1)

                else:
                    edge_weights = torch.cat((edge_weights, torch.tensor([rating])))
                    x = torch.cat((x, torch.ones(n_nodes - x.shape[0], 1)))

                count += 1

                edge_weights_tmp = edge_weights.resize(edge_weights.shape[0], 1)
                data = Data(x=x, edge_weights=edge_weights_tmp, edge_index=torch.tensor(edge_index),
                            edge_index_t=torch.tensor(edge_index_t),
                            source=source, dst=dst, n_operation=n_operation)

                snap.append(data)

                nodes_list.append(source)
                nodes_list.append(dst)

            snapshot[ts] = snap

        print(f'# occur: {count_occur}, {(count_occur / len(np.unique(nodes_list))) * 100:.2f} %')

        return snapshot

    def create_data(snapshot):
        snap_generation = []
        snap_regression = []

        for key in snapshot.keys():
            snap = snapshot[key]

            snap_generation.extend(snap)
            snap_regression.append(snap[-1])

        return snap_generation, snap_regression

    def create_sequences(data, window_size, window_shift, starting_point=0):
        x_sequences = []
        y_sequences = []

        idxs = range(starting_point, len(data) - window_size, window_shift)

        for idx in tqdm(idxs):
            x_sequences.append(data[idx:idx + window_size])
            y_sequences.append(data[idx + window_size])

        return x_sequences, y_sequences

    def create_loader(x_seq, y_seq, window_size):
        snapshot = []

        for idx, history in tqdm(enumerate(x_seq)):
            next_graph = y_seq[idx]
            last_graph = history[-1]
            last_graph_star = create_graph_star(last_graph.clone())
            last_graph_star_two = create_graph_star_two_nodes(last_graph.clone())

            loader = DataLoader(history, batch_size=window_size)
            snapshot.append([loader, next_graph, last_graph, last_graph_star, last_graph_star_two])

        return snapshot

    if is_data:
        df = pd.read_csv(file_name, header=None, names=['source', 'dest', 'rating', 'timestamp'])
        print(df.head())
        node_mappings = create_mapping(df)
        snapshot = create_snap(df, node_mappings)
        snap_generation, snap_regression = create_data(snapshot)
        torch.save(snap_generation, path_generation)
        torch.save(snap_regression, path_regression)

    else:
        snap_generation = torch.load(path_generation)
        snap_regression = torch.load(path_regression)

    print('Generation ...')

    amount = len(snap_generation)
    perc_val_test = (1 - perc_train) / 2

    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test)

    train_set_gen = snap_generation[0:train_size]
    val_set_gen = snap_generation[train_size: train_size + val_size]
    test_set_gen = snap_generation[train_size + val_size:]

    x_sequences_train, y_sequences_train = create_sequences(train_set_gen, window_size, window_shift)
    x_sequences_val, y_sequences_val = create_sequences(val_set_gen, window_size, window_shift)
    x_sequences_test, y_sequences_test = create_sequences(test_set_gen, window_size, window_shift)

    snapshot_train_gen = create_loader(x_sequences_train, y_sequences_train, window_size)
    snapshot_val_gen = create_loader(x_sequences_val, y_sequences_val, window_size)
    snapshot_test_gen = create_loader(x_sequences_test, y_sequences_test, window_size)

    amount = len(snap_regression)
    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test)

    train_set_reg = snap_regression[0:train_size]
    val_set_reg = snap_regression[train_size: train_size + val_size]
    test_set_reg = snap_regression[train_size + val_size:]

    return snapshot_train_gen, snapshot_test_gen, snapshot_val_gen, train_set_reg, test_set_reg, val_set_reg


def create_data_eu_core_no_dupl(file_name, window_size, window_shift, is_data=True, path_generation=None,
                                path_regression=None, perc_train=0.7):
    def create_mapping(df):
        timestamp = sorted(df['timestamp'].unique())
        node_mappings = {}
        node_idx = 0

        for ts in timestamp:
            sub_group = df[df['timestamp'] == ts]

            for g in sub_group.iterrows():
                source = g[1]['source']
                dst = g[1]['dest']

                if source not in node_mappings.keys():
                    node_mappings[source] = node_idx
                    node_idx += 1

                if dst not in node_mappings.keys():
                    node_mappings[dst] = node_idx
                    node_idx += 1
        return node_mappings

    def create_snap(df, node_mappings):
        timestamp = sorted(df['timestamp'].unique())
        edge_index, edge_index_t = [[], []], [[], []]
        snapshot = {}
        count = 0
        x = None
        nodes_list = []

        count_occur = 0

        edges = set()

        for ts in tqdm(timestamp, total=len(timestamp)):
            snap = []

            sub_group = df[df['timestamp'] == ts]

            n_operation = 0

            for _, row in sub_group.iterrows():
                source = row['source']
                dst = row['dest']

                source = node_mappings[source]
                dst = node_mappings[dst]

                if source not in nodes_list and dst not in nodes_list:
                    count_occur += 1

                if (source, dst) not in edges:
                    edges.add((source, dst))

                    # Create edge index
                    edge_index[0].extend([source])
                    edge_index[1].extend([dst])

                    # Create edge index T
                    edge_index_t[0].extend([dst])
                    edge_index_t[1].extend([source])

                    # Create features
                    n_nodes = max(np.concatenate((edge_index[0], edge_index[1]))) + 1

                    if count == 0:
                        x = torch.ones(n_nodes, 1)
                    else:
                        x = torch.cat((x, torch.ones(n_nodes - x.shape[0], 1)))

                    count += 1

                    n_operation += 1

                    data = Data(x=x, edge_index=torch.tensor(edge_index), edge_index_t=torch.tensor(edge_index_t),
                                source=source, dst=dst, n_operation=n_operation)

                    snap.append(data)

                    nodes_list.append(source)
                    nodes_list.append(dst)

            for elem in snap:
                elem.n_operation = n_operation

            if len(snap) != 0:
                snapshot[ts] = snap

        print(f'# occur: {count_occur}, {(count_occur / len(np.unique(nodes_list))) * 100:.2f} %')

        return snapshot

    def create_data(snapshot):
        snap_generation = []
        snap_regression = []

        for key in snapshot.keys():
            snap = snapshot[key]

            snap_generation.extend(snap)
            snap_regression.append(snap[-1])

        return snap_generation, snap_regression

    def create_sequences(data, window_size, window_shift, starting_point=0):
        x_sequences = []
        y_sequences = []

        idxs = range(starting_point, len(data) - window_size, window_shift)

        for idx in tqdm(idxs):
            x_sequences.append(data[idx:idx + window_size])
            y_sequences.append(data[idx + window_size])

        return x_sequences, y_sequences

    def create_loader(x_seq, y_seq, window_size):
        snapshot = []

        for idx, history in tqdm(enumerate(x_seq)):
            next_graph = y_seq[idx]
            last_graph = history[-1]
            last_graph_star = create_graph_star(last_graph.clone())
            last_graph_star_two = create_graph_star_two_nodes(last_graph.clone())

            loader = DataLoader(history, batch_size=window_size)
            snapshot.append([loader, next_graph, last_graph, last_graph_star, last_graph_star_two])

        return snapshot

    if is_data:
        df = pd.read_csv(file_name, header=None, names=['source', 'dest', 'timestamp'], sep=' ')
        print(df.head())
        node_mappings = create_mapping(df)
        snapshot = create_snap(df, node_mappings)
        snap_generation, snap_regression = create_data(snapshot)
        torch.save(snap_generation, path_generation)
        torch.save(snap_regression, path_regression)

    else:
        snap_generation = torch.load(path_generation)
        snap_regression = torch.load(path_regression)

    perc_val_test = (1. - perc_train) / 2
    amount = len(snap_regression)
    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test)

    train_set_reg = snap_regression[0:train_size]
    val_set_reg = snap_regression[train_size:train_size + val_size]
    test_set_reg = snap_regression[train_size + val_size:]

    ####

    train_size_regression = 0
    for d in train_set_reg:
        n_operations = d.n_operation
        train_size_regression += n_operations

    val_size_regression = train_size_regression
    for d in val_set_reg:
        n_operations = d.n_operation
        val_size_regression += n_operations

    train_set_gen = snap_generation[0:train_size_regression]
    val_set_gen = snap_generation[train_size_regression:train_size_regression + val_size_regression]
    test_set_gen = snap_generation[train_size_regression + val_size_regression:]

    x_sequences_train, y_sequences_train = create_sequences(train_set_gen, window_size, window_shift)
    x_sequences_val, y_sequences_val = create_sequences(val_set_gen, window_size, window_shift)
    x_sequences_test, y_sequences_test = create_sequences(test_set_gen, window_size, window_shift)

    snapshot_train_gen = create_loader(x_sequences_train, y_sequences_train, window_size)
    snapshot_val_gen = create_loader(x_sequences_val, y_sequences_val, window_size)
    snapshot_test_gen = create_loader(x_sequences_test, y_sequences_test, window_size)

    print('Last train gen: ', train_set_gen[-1])
    print('Last train reg: ', train_set_reg[-1])

    return snapshot_train_gen, snapshot_test_gen, snapshot_val_gen, train_set_reg, test_set_reg, val_set_reg


def create_data_uci_forum(file_name, window_size, window_shift, is_data=True, path_generation=None,
                          path_regression=None, perc_train=0.7):
    def create_mapping(df):
        timestamp = sorted(df['timestamp'].unique())
        node_mappings = {}
        node_idx = 0

        for ts in timestamp:
            sub_group = df[df['timestamp'] == ts]

            for g in sub_group.iterrows():
                source = g[1]['source']
                dst = g[1]['dest']

                if source not in node_mappings.keys():
                    node_mappings[source] = node_idx
                    node_idx += 1

                if dst not in node_mappings.keys():
                    node_mappings[dst] = node_idx
                    node_idx += 1
        return node_mappings

    def create_snap(df, node_mappings):
        timestamp = sorted(df['timestamp'].unique())
        edge_index, edge_index_t = [[], []], [[], []]
        snapshot = {}
        count = 0
        x = None
        nodes_list = []

        count_occur = 0

        edges = set()

        for ts in tqdm(timestamp, total=len(timestamp)):
            snap = []

            sub_group = df[df['timestamp'] == ts]

            n_operation = 0

            for _, row in sub_group.iterrows():
                source = row['source']
                dst = row['dest']

                source = node_mappings[source]
                dst = node_mappings[dst]

                if source not in nodes_list and dst not in nodes_list:
                    count_occur += 1

                if (source, dst) not in edges:
                    edges.add((source, dst))

                    # Create edge index
                    edge_index[0].extend([source])
                    edge_index[1].extend([dst])

                    # Create edge index T
                    edge_index_t[0].extend([dst])
                    edge_index_t[1].extend([source])

                    # Create features
                    n_nodes = max(np.concatenate((edge_index[0], edge_index[1]))) + 1

                    if count == 0:
                        x = torch.ones(n_nodes, 1)
                    else:
                        x = torch.cat((x, torch.ones(n_nodes - x.shape[0], 1)))

                    count += 1
                    n_operation += 1

                    data = Data(x=x, edge_index=torch.tensor(edge_index), edge_index_t=torch.tensor(edge_index_t),
                                source=source, dst=dst, n_operation=n_operation)

                    snap.append(data)

                    nodes_list.append(source)
                    nodes_list.append(dst)

            for elem in snap:
                elem.n_operation = n_operation

            if len(snap) != 0:
                snapshot[ts] = snap

        print(f'# occur: {count_occur}, {(count_occur / len(np.unique(nodes_list))) * 100:.2f} %')

        return snapshot

    def create_data(snapshot):
        snap_generation = []
        snap_regression = []

        for key in snapshot.keys():
            snap = snapshot[key]

            snap_generation.extend(snap)
            snap_regression.append(snap[-1])

        return snap_generation, snap_regression

    def create_sequences(data, window_size, window_shift, starting_point=0):
        x_sequences = []
        y_sequences = []

        idxs = range(starting_point, len(data) - window_size, window_shift)

        for idx in tqdm(idxs):
            x_sequences.append(data[idx:idx + window_size])
            y_sequences.append(data[idx + window_size])

        return x_sequences, y_sequences

    def create_loader(x_seq, y_seq, window_size):
        snapshot = []

        for idx, history in tqdm(enumerate(x_seq)):
            next_graph = y_seq[idx]
            last_graph = history[-1]
            last_graph_star = create_graph_star(last_graph.clone())
            last_graph_star_two = create_graph_star_two_nodes(last_graph.clone())

            loader = DataLoader(history, batch_size=window_size)
            snapshot.append([loader, next_graph, last_graph, last_graph_star, last_graph_star_two])

        return snapshot

    if is_data:
        df = pd.read_csv(file_name, header=None, names=['source', 'dest', 'edge_attr', 'timestamp'], sep=' ')
        print(df.head())
        node_mappings = create_mapping(df)
        snapshot = create_snap(df, node_mappings)
        snap_generation, snap_regression = create_data(snapshot)
        torch.save(snap_generation, path_generation)
        torch.save(snap_regression, path_regression)

    else:
        snap_generation = torch.load(path_generation)
        snap_regression = torch.load(path_regression)

    perc_val_test = (1. - perc_train) / 2

    amount = len(snap_regression)
    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test)

    train_set_reg = snap_regression[0:train_size]
    val_set_reg = snap_regression[train_size:train_size + val_size]
    test_set_reg = snap_regression[train_size + val_size:]

    ####

    train_size_regression = 0
    for d in train_set_reg:
        n_operations = d.n_operation
        train_size_regression += n_operations

    val_size_regression = train_size_regression
    for d in val_set_reg:
        n_operations = d.n_operation
        val_size_regression += n_operations

    train_set_gen = snap_generation[0:train_size_regression]
    val_set_gen = snap_generation[train_size_regression:train_size_regression + val_size_regression]
    test_set_gen = snap_generation[train_size_regression + val_size_regression:]

    x_sequences_train, y_sequences_train = create_sequences(train_set_gen, window_size, window_shift)
    x_sequences_val, y_sequences_val = create_sequences(val_set_gen, window_size, window_shift)
    x_sequences_test, y_sequences_test = create_sequences(test_set_gen, window_size, window_shift)

    snapshot_train_gen = create_loader(x_sequences_train, y_sequences_train, window_size)
    snapshot_val_gen = create_loader(x_sequences_val, y_sequences_val, window_size)
    snapshot_test_gen = create_loader(x_sequences_test, y_sequences_test, window_size)

    print('Last train gen: ', train_set_gen[-1])
    print('Last train reg: ', train_set_reg[-1])

    return snapshot_train_gen, snapshot_test_gen, snapshot_val_gen, train_set_reg, test_set_reg, val_set_reg


def create_data_bitcoin(file_name, window_size, window_shift, is_data=True, path_generation=None,
                        path_regression=None, perc_train=0.7, features=False):
    def create_mapping(df):
        timestamp = sorted(df['timestamp'].unique())
        node_mappings = {}
        node_idx = 0

        for ts in timestamp:
            sub_group = df[df['timestamp'] == ts]

            for g in sub_group.iterrows():
                source = g[1]['source']
                dst = g[1]['dest']

                if source not in node_mappings.keys():
                    node_mappings[source] = node_idx
                    node_idx += 1

                if dst not in node_mappings.keys():
                    node_mappings[dst] = node_idx
                    node_idx += 1
        return node_mappings

    def create_snap(df, node_mappings):
        timestamp = sorted(df['timestamp'].unique())
        edge_index, edge_index_t = [[], []], [[], []]
        snapshot = {}
        count = 0
        x = None
        nodes_list = []

        count_occur = 0

        for ts in tqdm(timestamp, total=len(timestamp)):
            snap = []

            sub_group = df[df['timestamp'] == ts]

            n_operation = len(sub_group)

            for _, row in sub_group.iterrows():
                source = row['source']
                dst = row['dest']
                rating = row['rating']

                source = node_mappings[source]
                dst = node_mappings[dst]

                if source not in nodes_list and dst not in nodes_list:
                    count_occur += 1

                # Create edge index
                edge_index[0].extend([source])
                edge_index[1].extend([dst])

                # Create edge index T
                edge_index_t[0].extend([dst])
                edge_index_t[1].extend([source])

                # Create features
                n_nodes = max(np.concatenate((edge_index[0], edge_index[1]))) + 1

                if count == 0:

                    if features:
                        x = torch.zeros(n_nodes, 1)
                        x[dst] = rating
                        x[source] = 1e-2
                    else:
                        x = torch.ones(n_nodes, 1)
                else:
                    if features:
                        x = torch.cat((x, torch.zeros(n_nodes - x.shape[0], 1)))
                        x[dst] = (x[dst] + rating) / 2.

                        if x[source] == 0.:
                            x[source] = 1e-2
                    else:
                        x = torch.cat((x, torch.ones(n_nodes - x.shape[0], 1)))

                count += 1

                data = Data(x=x, edge_index=torch.tensor(edge_index), edge_index_t=torch.tensor(edge_index_t),
                            source=source, dst=dst, n_operation=n_operation)

                snap.append(data)

                nodes_list.append(source)
                nodes_list.append(dst)

            snapshot[ts] = snap

        print(f'# occur: {count_occur}, {(count_occur / len(np.unique(nodes_list))) * 100:.2f} %')

        return snapshot

    def create_data(snapshot):
        snap_generation = []
        snap_regression = []

        for key in snapshot.keys():
            snap = snapshot[key]

            snap_generation.extend(snap)
            snap_regression.append(snap[-1])

        return snap_generation, snap_regression

    def create_sequences(data, window_size, window_shift, starting_point=0):
        x_sequences = []
        y_sequences = []

        idxs = range(starting_point, len(data) - window_size, window_shift)

        for idx in tqdm(idxs):
            x_sequences.append(data[idx:idx + window_size])
            y_sequences.append(data[idx + window_size])

        return x_sequences, y_sequences

    def create_loader(x_seq, y_seq, window_size):
        snapshot = []

        for idx, history in tqdm(enumerate(x_seq)):
            next_graph = y_seq[idx]
            last_graph = history[-1]
            last_graph_star = create_graph_star(last_graph.clone())
            last_graph_star_two = create_graph_star_two_nodes(last_graph.clone())

            loader = DataLoader(history, batch_size=window_size)
            snapshot.append([loader, next_graph, last_graph, last_graph_star, last_graph_star_two])

        return snapshot

    if is_data:
        df = pd.read_csv(file_name, header=None, names=['source', 'dest', 'rating', 'timestamp'])
        print(df.head())
        node_mappings = create_mapping(df)
        snapshot = create_snap(df, node_mappings)
        snap_generation, snap_regression = create_data(snapshot)
        torch.save(snap_generation, path_generation)
        torch.save(snap_regression, path_regression)

    else:
        snap_generation = torch.load(path_generation)
        snap_regression = torch.load(path_regression)

    print('Generation ...')

    amount = len(snap_generation)
    perc_val_test = (1 - perc_train) / 2

    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test)

    train_set_gen = snap_generation[0:train_size]
    val_set_gen = snap_generation[train_size: train_size + val_size]
    test_set_gen = snap_generation[train_size + val_size:]

    x_sequences_train, y_sequences_train = create_sequences(train_set_gen, window_size, window_shift)
    x_sequences_val, y_sequences_val = create_sequences(val_set_gen, window_size, window_shift)
    x_sequences_test, y_sequences_test = create_sequences(test_set_gen, window_size, window_shift)

    snapshot_train_gen = create_loader(x_sequences_train, y_sequences_train, window_size)
    snapshot_val_gen = create_loader(x_sequences_val, y_sequences_val, window_size)
    snapshot_test_gen = create_loader(x_sequences_test, y_sequences_test, window_size)

    amount = len(snap_regression)
    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test)

    train_set_reg = snap_regression[0:train_size]
    val_set_reg = snap_regression[train_size: train_size + val_size]
    test_set_reg = snap_regression[train_size + val_size:]

    print('Last train gen: ', train_set_gen[-1])
    print('Last train reg: ', train_set_reg[-1])

    return snapshot_train_gen, snapshot_test_gen, snapshot_val_gen, train_set_reg, test_set_reg, val_set_reg


def create_data_bitcoin_alpha(file_name, window_size, window_shift, is_data=True, path_generation=None,
                              path_regression=None, perc_train=0.7, features=False):
    def create_mapping(df):
        timestamp = sorted(df['timestamp'].unique())
        node_mappings = {}
        node_idx = 0

        for ts in timestamp:
            sub_group = df[df['timestamp'] == ts]

            for g in sub_group.iterrows():
                source = g[1]['source']
                dst = g[1]['dest']

                if source not in node_mappings.keys():
                    node_mappings[source] = node_idx
                    node_idx += 1

                if dst not in node_mappings.keys():
                    node_mappings[dst] = node_idx
                    node_idx += 1
        return node_mappings

    def create_snap(df, node_mappings):
        timestamp = sorted(df['timestamp'].unique())
        edge_index, edge_index_t = [[], []], [[], []]
        snapshot = {}
        count = 0
        x = None
        nodes_list = []

        count_occur = 0

        for ts in tqdm(timestamp, total=len(timestamp)):
            snap = []

            sub_group = df[df['timestamp'] == ts]

            n_operation = len(sub_group)

            for _, row in sub_group.iterrows():
                source = row['source']
                dst = row['dest']
                rating = row['rating']

                source = node_mappings[source]
                dst = node_mappings[dst]

                if source not in nodes_list and dst not in nodes_list:
                    count_occur += 1

                # Create edge index
                edge_index[0].extend([source])
                edge_index[1].extend([dst])

                # Create edge index T
                edge_index_t[0].extend([dst])
                edge_index_t[1].extend([source])

                # Create features
                n_nodes = max(np.concatenate((edge_index[0], edge_index[1]))) + 1

                if count == 0:

                    if features:
                        x = torch.zeros(n_nodes, 1)
                        x[dst] = rating
                        x[source] = 1e-2
                    else:
                        x = torch.ones(n_nodes, 1)
                else:
                    if features:
                        x = torch.cat((x, torch.zeros(n_nodes - x.shape[0], 1)))
                        x[dst] = (x[dst] + rating) / 2.

                        if x[source] == 0.:
                            x[source] = 1e-2
                    else:
                        x = torch.cat((x, torch.ones(n_nodes - x.shape[0], 1)))

                count += 1

                data = Data(x=x, edge_index=torch.tensor(edge_index), edge_index_t=torch.tensor(edge_index_t),
                            source=source, dst=dst, n_operation=n_operation)

                snap.append(data)

                nodes_list.append(source)
                nodes_list.append(dst)

            snapshot[ts] = snap

        print(f'# occur: {count_occur}, {(count_occur / len(np.unique(nodes_list))) * 100:.2f} %')

        return snapshot

    def create_data(snapshot):
        snap_generation = []
        snap_regression = []

        for key in snapshot.keys():
            snap = snapshot[key]

            snap_generation.extend(snap)
            snap_regression.append(snap[-1])

        return snap_generation, snap_regression

    def create_sequences(data, window_size, window_shift, starting_point=0):
        x_sequences = []
        y_sequences = []

        idxs = range(starting_point, len(data) - window_size, window_shift)

        for idx in tqdm(idxs):
            x_sequences.append(data[idx:idx + window_size])
            y_sequences.append(data[idx + window_size])

        return x_sequences, y_sequences

    def create_loader(x_seq, y_seq, window_size):
        snapshot = []

        for idx, history in tqdm(enumerate(x_seq)):
            next_graph = y_seq[idx]
            last_graph = history[-1]
            last_graph_star = create_graph_star(last_graph.clone())
            last_graph_star_two = create_graph_star_two_nodes(last_graph.clone())

            loader = DataLoader(history, batch_size=window_size)
            snapshot.append([loader, next_graph, last_graph, last_graph_star, last_graph_star_two])

        return snapshot

    if is_data:
        df = pd.read_csv(file_name, header=None, names=['source', 'dest', 'rating', 'timestamp'])
        print(df.head())
        node_mappings = create_mapping(df)
        snapshot = create_snap(df, node_mappings)
        snap_generation, snap_regression = create_data(snapshot)
        torch.save(snap_generation, path_generation)
        torch.save(snap_regression, path_regression)

    else:
        snap_generation = torch.load(path_generation)
        snap_regression = torch.load(path_regression)

    print('Generation ...')

    perc_val_test = (1. - perc_train) / 2

    amount = len(snap_regression)
    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test)

    train_set_reg = snap_regression[0:train_size]
    val_set_reg = snap_regression[train_size:train_size + val_size]
    test_set_reg = snap_regression[train_size + val_size:]

    ####

    train_size_regression = 0
    for d in train_set_reg:
        n_operations = d.n_operation
        train_size_regression += n_operations

    val_size_regression = train_size_regression
    for d in val_set_reg:
        n_operations = d.n_operation
        val_size_regression += n_operations

    train_set_gen = snap_generation[0:train_size_regression]
    val_set_gen = snap_generation[train_size_regression:train_size_regression + val_size_regression]
    test_set_gen = snap_generation[train_size_regression + val_size_regression:]

    x_sequences_train, y_sequences_train = create_sequences(train_set_gen, window_size, window_shift)
    x_sequences_val, y_sequences_val = create_sequences(val_set_gen, window_size, window_shift)
    x_sequences_test, y_sequences_test = create_sequences(test_set_gen, window_size, window_shift)

    snapshot_train_gen = create_loader(x_sequences_train, y_sequences_train, window_size)
    snapshot_val_gen = create_loader(x_sequences_val, y_sequences_val, window_size)
    snapshot_test_gen = create_loader(x_sequences_test, y_sequences_test, window_size)

    print('Last train gen: ', train_set_gen[-1])
    print('Last train reg: ', train_set_reg[-1])

    return snapshot_train_gen, snapshot_test_gen, snapshot_val_gen, train_set_reg, test_set_reg, val_set_reg
