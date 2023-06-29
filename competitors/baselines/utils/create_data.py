import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import pandas as pd
import numpy as np

def create_data_bitcoin_edge_weights(file_name, path=None):

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
        snap_regression = []

        for key in snapshot.keys():
            snap = snapshot[key]

            snap_regression.append(snap[-1])

        return snap_regression


    df = pd.read_csv(file_name, header=None, names=['source', 'dest', 'rating', 'timestamp'])
    node_mappings = create_mapping(df)
    snapshot = create_snap(df, node_mappings)
    snapshot = create_data(snapshot)
    torch.save(snapshot, path)


def create_data_eu_core_no_dupl(file_name, path=None):
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
        snap_regression = []

        for key in snapshot.keys():
            snap = snapshot[key]

            snap_regression.append(snap[-1])

        return snap_regression

    df = pd.read_csv(file_name, header=None, names=['source', 'dest', 'timestamp'], sep=' ')
    print(df.head())
    node_mappings = create_mapping(df)
    snapshot = create_snap(df, node_mappings)
    snapshot = create_data(snapshot)
    torch.save(snapshot, path)

def create_data_uci_forum(file_name, path=None):
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
        snap_regression = []

        for key in snapshot.keys():
            snap = snapshot[key]

            snap_regression.append(snap[-1])

        return snap_regression

    df = pd.read_csv(file_name, header=None, names=['source', 'dest', 'edge_attr', 'timestamp'], sep=' ')
    print(df.head())
    node_mappings = create_mapping(df)
    snapshot = create_snap(df, node_mappings)
    snapshot = create_data(snapshot)
    torch.save(snapshot, path)

