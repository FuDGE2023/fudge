import torch
from torch_geometric.data import Data

from tqdm import tqdm
import pandas as pd

from utils.utils import create_sequences, create_loader
import numpy as np
import random


def create_data_bitcoin_edge_weights(file_name, window_size, window_shift, is_data=True, path_generation=None,
                                     path_regression=None, perc_train=0.7, batch_size=32, seed=42):
    random.seed(seed)
    np.random.seed(seed)

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

    val_size_regression = 0
    for d in val_set_reg:
        n_operations = d.n_operation
        val_size_regression += n_operations

    train_set_gen = snap_generation[0:train_size_regression]
    val_set_gen = snap_generation[train_size_regression:train_size_regression + val_size_regression]
    test_set_gen = snap_generation[train_size_regression + val_size_regression:]

    x_sequences_train, y_sequences_train = create_sequences(train_set_gen, window_size, window_shift)
    x_sequences_val, y_sequences_val = create_sequences(val_set_gen, window_size, window_shift)
    x_sequences_test, y_sequences_test = create_sequences(test_set_gen, window_size, window_shift)

    # Shuffle for batch
    temp = list(zip(x_sequences_train, y_sequences_train))
    random.shuffle(temp)
    x_sequences_train, y_sequences_train = zip(*temp)
    x_sequences_train, y_sequences_train = list(x_sequences_train), list(y_sequences_train)

    temp = list(zip(x_sequences_val, y_sequences_val))
    random.shuffle(temp)
    x_sequences_val, y_sequences_val = zip(*temp)
    x_sequences_val, y_sequences_val = list(x_sequences_val), list(y_sequences_val)

    snapshot_train_gen = create_loader(x_sequences_train, y_sequences_train, window_size, batch_size)
    snapshot_val_gen = create_loader(x_sequences_val, y_sequences_val, window_size, batch_size)
    snapshot_test_gen = create_loader(x_sequences_test, y_sequences_test, window_size, batch_size)

    print('Last train gen: ', train_set_gen[-1])
    print('Last train reg: ', train_set_reg[-1])

    return x_sequences_train, y_sequences_train, snapshot_train_gen, snapshot_test_gen, snapshot_val_gen, \
           train_set_reg, test_set_reg, val_set_reg, train_set_gen, test_set_gen, val_set_gen


def create_data_eu_core_no_dupl(file_name, window_size, window_shift, is_data=True, path_generation=None,
                        path_regression=None, learning='generation', test=False, perc_train=0.7, batch_size=32, seed=42):
    random.seed(seed)
    np.random.seed(seed)

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


    if is_data:
        df = pd.read_csv(file_name, header=None, names=['source', 'dest', 'timestamp'], sep=' ')
        print(df.head())
        node_mappings = create_mapping(df)
        snapshot = create_snap(df, node_mappings)
        snap_generation, snap_regression = create_data(snapshot)
        # torch.save(snap_generation, path_generation)
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

    val_size_regression = 0
    for d in val_set_reg:
        n_operations = d.n_operation
        val_size_regression += n_operations

    train_set_gen = snap_generation[0:train_size_regression]
    val_set_gen = snap_generation[train_size_regression:train_size_regression + val_size_regression]
    test_set_gen = snap_generation[train_size_regression + val_size_regression:]

    print(len(train_set_gen))

    x_sequences_train, y_sequences_train = create_sequences(train_set_gen, window_size, window_shift)
    x_sequences_val, y_sequences_val = create_sequences(val_set_gen, window_size, window_shift)
    x_sequences_test, y_sequences_test = create_sequences(test_set_gen, window_size, window_shift)

    print(len(x_sequences_train))

    # Shuffle for batch
    temp = list(zip(x_sequences_train, y_sequences_train))
    random.shuffle(temp)
    x_sequences_train, y_sequences_train = zip(*temp)
    x_sequences_train, y_sequences_train = list(x_sequences_train), list(y_sequences_train)

    temp = list(zip(x_sequences_val, y_sequences_val))
    random.shuffle(temp)
    x_sequences_val, y_sequences_val = zip(*temp)
    x_sequences_val, y_sequences_val = list(x_sequences_val), list(y_sequences_val)

    snapshot_train_gen = create_loader(x_sequences_train, y_sequences_train, window_size, batch_size)
    snapshot_val_gen = create_loader(x_sequences_val, y_sequences_val, window_size, batch_size)
    snapshot_test_gen = create_loader(x_sequences_test, y_sequences_test, window_size, batch_size)

    print('Last train gen: ', train_set_gen[-1])
    print('Last train reg: ', train_set_reg[-1])

    print(len(snapshot_train_gen))

    return x_sequences_train, y_sequences_train, snapshot_train_gen, snapshot_test_gen, snapshot_val_gen, train_set_reg, test_set_reg, val_set_reg, train_set_gen, test_set_gen, val_set_gen


def create_data_uci_forum(file_name, window_size, window_shift, is_data=True, path_generation=None,
                          path_regression=None, perc_train=0.7, test=False, batch_size=32, seed=42):
    random.seed(seed)
    np.random.seed(seed)

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

    if is_data:
        df = pd.read_csv(file_name, header=None, names=['source', 'dest', 'edge_attr', 'timestamp'], sep=' ')
        print(df.head())
        node_mappings = create_mapping(df)
        snapshot = create_snap(df, node_mappings)
        snap_generation, snap_regression = create_data(snapshot)
        # torch.save(snap_generation, path_generation)
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

    val_size_regression = 0
    for d in val_set_reg:
        n_operations = d.n_operation
        val_size_regression += n_operations

    train_set_gen = snap_generation[0:train_size_regression]
    val_set_gen = snap_generation[train_size_regression:train_size_regression + val_size_regression]
    test_set_gen = snap_generation[train_size_regression + val_size_regression:]

    x_sequences_train, y_sequences_train = create_sequences(train_set_gen, window_size, window_shift)
    x_sequences_val, y_sequences_val = create_sequences(val_set_gen, window_size, window_shift)
    x_sequences_test, y_sequences_test = create_sequences(test_set_gen, window_size, window_shift)

    # Shuffle for batch
    temp = list(zip(x_sequences_train, y_sequences_train))
    random.shuffle(temp)
    x_sequences_train, y_sequences_train = zip(*temp)
    x_sequences_train, y_sequences_train = list(x_sequences_train), list(y_sequences_train)

    temp = list(zip(x_sequences_val, y_sequences_val))
    random.shuffle(temp)
    x_sequences_val, y_sequences_val = zip(*temp)
    x_sequences_val, y_sequences_val = list(x_sequences_val), list(y_sequences_val)

    snapshot_train_gen = create_loader(x_sequences_train, y_sequences_train, window_size, batch_size)
    snapshot_val_gen = create_loader(x_sequences_val, y_sequences_val, window_size, batch_size)
    snapshot_test_gen = create_loader(x_sequences_test, y_sequences_test, window_size, batch_size)

    print('Last train gen: ', train_set_gen[-1])
    print('Last train reg: ', train_set_reg[-1])

    return x_sequences_train, y_sequences_train, snapshot_train_gen, snapshot_test_gen, snapshot_val_gen, train_set_reg, test_set_reg, val_set_reg, train_set_gen, test_set_gen, val_set_gen


def create_data_bitcoin(file_name, window_size, window_shift, is_data=True, path_generation=None,
                        path_regression=None, perc_train=0.7, features=False, batch_size=32, seed=42):
    random.seed(seed)
    np.random.seed(seed)

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

    amount = len(snap_regression)
    perc_val_test = (1 - perc_train) / 2
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

    val_size_regression = 0
    for d in val_set_reg:
        n_operations = d.n_operation
        val_size_regression += n_operations

    train_set_gen = snap_generation[0:train_size_regression]
    val_set_gen = snap_generation[train_size_regression:train_size_regression + val_size_regression]
    test_set_gen = snap_generation[train_size_regression + val_size_regression:]

    x_sequences_train, y_sequences_train = create_sequences(train_set_gen, window_size, window_shift)
    x_sequences_val, y_sequences_val = create_sequences(val_set_gen, window_size, window_shift)
    x_sequences_test, y_sequences_test = create_sequences(test_set_gen, window_size, window_shift)

    # Shuffle for batch
    temp = list(zip(x_sequences_train, y_sequences_train))
    random.shuffle(temp)
    x_sequences_train, y_sequences_train = zip(*temp)
    x_sequences_train, y_sequences_train = list(x_sequences_train), list(y_sequences_train)

    temp = list(zip(x_sequences_val, y_sequences_val))
    random.shuffle(temp)
    x_sequences_val, y_sequences_val = zip(*temp)
    x_sequences_val, y_sequences_val = list(x_sequences_val), list(y_sequences_val)

    snapshot_train_gen = create_loader(x_sequences_train, y_sequences_train, window_size, batch_size)
    snapshot_val_gen = create_loader(x_sequences_val, y_sequences_val, window_size, batch_size)
    snapshot_test_gen = create_loader(x_sequences_test, y_sequences_test, window_size, batch_size)

    print('Last train gen: ', train_set_gen[-1])
    print('Last train reg: ', train_set_reg[-1])

    return x_sequences_train, y_sequences_train, snapshot_train_gen, snapshot_test_gen, snapshot_val_gen, train_set_reg, test_set_reg, val_set_reg, train_set_gen, test_set_gen, val_set_gen


def create_data_bitcoin_alpha(file_name, window_size, window_shift, is_data=True, path_generation=None,
                        path_regression=None, perc_train=0.7, features=False, batch_size=32, seed=42):
    random.seed(seed)
    np.random.seed(seed)

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

    val_size_regression = 0
    for d in val_set_reg:
        n_operations = d.n_operation
        val_size_regression += n_operations

    train_set_gen = snap_generation[0:train_size_regression]
    val_set_gen = snap_generation[train_size_regression:train_size_regression + val_size_regression]
    test_set_gen = snap_generation[train_size_regression + val_size_regression:]

    x_sequences_train, y_sequences_train = create_sequences(train_set_gen, window_size, window_shift)
    x_sequences_val, y_sequences_val = create_sequences(val_set_gen, window_size, window_shift)
    x_sequences_test, y_sequences_test = create_sequences(test_set_gen, window_size, window_shift)

    # Shuffle for batch
    temp = list(zip(x_sequences_train, y_sequences_train))
    random.shuffle(temp)
    x_sequences_train, y_sequences_train = zip(*temp)
    x_sequences_train, y_sequences_train = list(x_sequences_train), list(y_sequences_train)

    temp = list(zip(x_sequences_val, y_sequences_val))
    random.shuffle(temp)
    x_sequences_val, y_sequences_val = zip(*temp)
    x_sequences_val, y_sequences_val = list(x_sequences_val), list(y_sequences_val)

    snapshot_train_gen = create_loader(x_sequences_train, y_sequences_train, window_size, batch_size)
    snapshot_val_gen = create_loader(x_sequences_val, y_sequences_val, window_size, batch_size)
    snapshot_test_gen = create_loader(x_sequences_test, y_sequences_test, window_size, batch_size)

    print('Last train gen: ', train_set_gen[-1])
    print('Last train reg: ', train_set_reg[-1])

    return x_sequences_train, y_sequences_train, snapshot_train_gen, snapshot_test_gen, snapshot_val_gen, train_set_reg, test_set_reg, val_set_reg, train_set_gen, test_set_gen, val_set_gen


def load_synthetic(params, seed=42):
    from utils.utils import loadList

    random.seed(seed)
    np.random.seed(seed)

    dataset_path = params['dataset_path']
    snap = loadList(dataset_path)

    window_size = params['window_size']
    window_shift = params['window_shift']
    perc_train = params['perc_train']
    batch_size = params['batch_size']
    print(f'batch_size: {batch_size}')

    amount = len(snap)
    perc_val_test = (1. - perc_train) / 2

    print(f'% train: {perc_train}, % val: {perc_val_test}')

    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test)

    train_set = snap[0:train_size]
    val_set = snap[train_size:train_size + val_size]
    test_set = snap[train_size + val_size:]

    x_sequences_train, y_sequences_train = create_sequences(train_set, window_size, window_shift)
    x_sequences_val, y_sequences_val = create_sequences(val_set, window_size, window_shift)
    x_sequences_test, y_sequences_test = create_sequences(test_set, window_size, window_shift)

    # Shuffle for batch
    temp = list(zip(x_sequences_train, y_sequences_train))
    random.shuffle(temp)
    x_sequences_train, y_sequences_train = zip(*temp)
    x_sequences_train, y_sequences_train = list(x_sequences_train), list(y_sequences_train)

    temp = list(zip(x_sequences_val, y_sequences_val))
    random.shuffle(temp)
    x_sequences_val, y_sequences_val = zip(*temp)
    x_sequences_val, y_sequences_val = list(x_sequences_val), list(y_sequences_val)

    snapshot_train = create_loader(x_sequences_train, y_sequences_train, window_size, batch_size, is_synthetic=True)
    snapshot_val = create_loader(x_sequences_val, y_sequences_val, window_size, batch_size, is_synthetic=True)
    snapshot_test = create_loader(x_sequences_test, y_sequences_test, window_size, batch_size, is_synthetic=True)

    return x_sequences_train, y_sequences_train, snapshot_train, snapshot_val, snapshot_test, train_set, val_set, test_set
