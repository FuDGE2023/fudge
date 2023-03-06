import sys

import torch
from torch_geometric.loader import DataLoader

from utils.plotter import plot_degree_dist
from torch_geometric.utils.convert import to_networkx

import networkx as nx

from grakel.kernels import WeisfeilerLehman

import numpy as np
from grakel.kernels.vertex_histogram import VertexHistogram

from tqdm import tqdm
import pandas as pd


def wl_test_last(g_true, g_gen, n_iter):
    print('WL-Test subtree kernel ...')

    print('True graph ...')

    edges = list(zip(g_true.edge_index[0].numpy(), g_true.edge_index[1].numpy()))

    starting_label = {}

    for node in range(g_true.x.shape[0]):
        starting_label[node] = 0

    G_true = [edges, starting_label]

    print('*' * 30)
    print('Generated graph before post-processing ...')

    edges_g = list(zip(g_gen.edge_index[0].numpy(), g_gen.edge_index[1].numpy()))

    starting_label_g = {}

    for node in range(g_gen.num_nodes):
        starting_label_g[node] = 0

    G_pred = [edges_g, starting_label_g]
    print('*' * 30)

    graphs = [G_true, G_pred]

    gk = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram, normalize=True)
    K = gk.fit_transform(graphs)

    print(K[0][1])

    return K


def run_generation(trainer, history, num_generation, params, count_checkpoint=0):
    trainer.gnn_history_source.load_state_dict(torch.load(params['gnn_history_source']))
    trainer.rnn_history_source.load_state_dict(torch.load(params['rnn_history_source']))

    trainer.gnn_aux_add_source.load_state_dict(torch.load(params['gnn_aux_add_source']))
    trainer.net_source.load_state_dict(torch.load(params['net_source']))

    trainer.gnn_history_dest.load_state_dict(torch.load(params['gnn_history_dest']))
    trainer.rnn_history_dest.load_state_dict(torch.load(params['rnn_history_dest']))

    trainer.gnn_aux_add_dest.load_state_dict(torch.load(params['gnn_aux_add_dest']))
    trainer.net_dest.load_state_dict(torch.load(params['net_dest']))

    models_add_source = (trainer.gnn_history_source, trainer.rnn_history_source, trainer.gnn_aux_add_source,
                         trainer.net_source)
    models_add_dest = (trainer.gnn_history_dest, trainer.rnn_history_dest, trainer.gnn_aux_add_dest,
                       trainer.net_dest)

    generated_snapshot = trainer.generate_graph_rnd_alternative(history, num_generation, models_add_source, models_add_dest,
                                                    count_checkpoint)
    return generated_snapshot


def run_generation_regressor(trainer, snap_train, history, num_generation, params, count_checkpoint=0):
    from model.regressor import DiscreteRegressor
    regressor = DiscreteRegressor(snap_train, seed=12345)

    trainer.gnn_history_source.load_state_dict(torch.load(params['gnn_history_source']))
    trainer.rnn_history_source.load_state_dict(torch.load(params['rnn_history_source']))

    trainer.gnn_aux_add_source.load_state_dict(torch.load(params['gnn_aux_add_source']))
    trainer.net_source.load_state_dict(torch.load(params['net_source']))

    trainer.gnn_history_dest.load_state_dict(torch.load(params['gnn_history_dest']))
    trainer.rnn_history_dest.load_state_dict(torch.load(params['rnn_history_dest']))

    trainer.gnn_aux_add_dest.load_state_dict(torch.load(params['gnn_aux_add_dest']))
    trainer.net_dest.load_state_dict(torch.load(params['net_dest']))

    models_add_source = (trainer.gnn_history_source, trainer.rnn_history_source, trainer.gnn_aux_add_source,
                         trainer.net_source)
    models_add_dest = (trainer.gnn_history_dest, trainer.rnn_history_dest, trainer.gnn_aux_add_dest,
                       trainer.net_dest)

    trainer.generate_graph_rnd_alternative_regressor(history, num_generation, models_add_source,
                                                                          models_add_dest, regressor, count_checkpoint)


def run_generation_regressor_sampling(trainer, snap_train, history, num_generation, params, count_checkpoint=0):
    from model.regressor import DiscreteRegressor
    regressor = DiscreteRegressor(snap_train, seed=12345)

    trainer.gnn_history_source.load_state_dict(torch.load(params['gnn_history_source']))
    trainer.rnn_history_source.load_state_dict(torch.load(params['rnn_history_source']))

    trainer.gnn_aux_add_source.load_state_dict(torch.load(params['gnn_aux_add_source']))
    trainer.net_source.load_state_dict(torch.load(params['net_source']))

    trainer.gnn_history_dest.load_state_dict(torch.load(params['gnn_history_dest']))
    trainer.rnn_history_dest.load_state_dict(torch.load(params['rnn_history_dest']))

    trainer.gnn_aux_add_dest.load_state_dict(torch.load(params['gnn_aux_add_dest']))
    trainer.net_dest.load_state_dict(torch.load(params['net_dest']))

    models_add_source = (trainer.gnn_history_source, trainer.rnn_history_source, trainer.gnn_aux_add_source,
                         trainer.net_source)
    models_add_dest = (trainer.gnn_history_dest, trainer.rnn_history_dest, trainer.gnn_aux_add_dest,
                       trainer.net_dest)

    trainer.generate_graph_rnd_alternative_regressor_sampling(history, num_generation, models_add_source,
                                                              models_add_dest, regressor, count_checkpoint)


def graph_statistics(G, torch_g=False):
    if not torch_g:
        G = to_networkx(G, to_undirected=False)

    print('Max diameter in G: ')
    diameter = max([max(j.values()) for (i, j) in nx.shortest_path_length(G)])
    print(diameter)
    print('-----' * 20)

    plot_degree_dist(G)

    print('-----' * 20)

    print('Average coef. clustering: ', nx.average_clustering(G))

    return diameter


def graph_statistics_all_nx(G_true, G_pred):
    from collections import Counter
    import matplotlib.pyplot as plt
    print('In degrees')
    degrees_true = [G_true.in_degree(n) for n in G_true.nodes()]

    c_true = Counter(degrees_true)

    degrees_pred = [G_pred.in_degree(n) for n in G_pred.nodes()]

    c_pred = Counter(degrees_pred)

    plt.figure()
    plt.plot(*zip(*sorted(c_pred.items())), '+', label='Generated Test set')
    plt.plot(*zip(*sorted(c_true.items())), '+', label='True Test set')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('degree')
    plt.ylabel('frequency')
    plt.legend()
    plt.show()

    print('Out degrees')
    degrees_true = [G_true.out_degree(n) for n in G_true.nodes()]

    c_true = Counter(degrees_true)

    degrees_pred = [G_pred.out_degree(n) for n in G_pred.nodes()]

    c_pred = Counter(degrees_pred)

    plt.figure()
    plt.plot(*zip(*sorted(c_true.items())), '+', label='Generated Test set')
    plt.plot(*zip(*sorted(c_pred.items())), '+', label='True Test set')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('degree')
    plt.ylabel('frequency')
    plt.legend()
    plt.show()

    print('-----' * 20)

    print('Max diameter in G_true: ')
    diameter_true = max([max(j.values()) for (i, j) in nx.shortest_path_length(G_true)])
    print(diameter_true)

    print('Max diameter in G_pred: ')
    diameter_pred = max([max(j.values()) for (i, j) in nx.shortest_path_length(G_pred)])
    print(diameter_pred)

    print('-----' * 20)

    print('True: Average coef. clustering: ', nx.average_clustering(G_true))
    print('Pred: Average coef. clustering: ', nx.average_clustering(G_pred))

    return diameter_true


def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray = np.load(filename, allow_pickle=True)
    return tempNumpyArray.tolist()


def saveList(myList, filename):
    # the filename should mention the extension 'npy'
    np.save(filename, myList)
    print("Saved successfully!")


def create_sequences(data, window_size, window_shift, starting_point=0):
    x_sequences = []
    y_sequences = []

    idxs = range(starting_point, len(data) - window_size, window_shift)

    for idx in idxs:
        x_sequences.append(data[idx:idx + window_size])
        y_sequences.append(data[idx + window_size])

    return x_sequences, y_sequences


def create_graph_star(graph):
    graph_x = torch.cat((graph.x, torch.ones(1, graph.x.shape[1])))
    graph.x = graph_x
    return graph


def create_graph_star_two_nodes(graph):
    graph_x = torch.cat((graph.x, torch.ones(2, graph.x.shape[1])))
    graph.x = graph_x
    return graph


def create_loader(x_seq, y_seq, window_size):
    snapshot = []

    for idx, history in enumerate(x_seq):
        next_graph = y_seq[idx]
        last_graph = history[-1][4]
        last_graph_star = create_graph_star(last_graph.clone())

        h = []
        for e in history:
            h.append(e[4])

        loader = DataLoader(h, batch_size=window_size)
        snapshot.append([loader, next_graph, last_graph, last_graph_star])

    return snapshot


def create_loader_remove(x_seq, y_seq, window_size):
    snapshot = []

    for idx, history in enumerate(x_seq):
        next_graph = y_seq[idx]

        if next_graph[7] == 'Add':
            next_graph[7] = torch.tensor([-1])
        else:
            next_graph[7] = torch.tensor([1])

        last_graph = history[-1][4]
        last_graph_star = create_graph_star(last_graph.clone())

        h = []
        for e in history:
            h.append(e[4])

        loader = DataLoader(h, batch_size=window_size)
        snapshot.append([loader, next_graph, last_graph, last_graph_star])

    return snapshot


def load_synthetic(params):
    path_file = params['path_file']
    snap = loadList(path_file)

    print('% All dataset ...')
    count_add = 0
    count_remove = 0
    for elem in snap:
        if elem[7] == 'Add':
            count_add += 1
        else:
            count_remove += 1

    print(count_add / (count_add + count_remove))
    print(count_remove / (count_add + count_remove))

    amount = len(snap)

    perc_train = params['perc_train']
    perc_val_test = (1 - perc_train) / 2

    print(f'% train: {perc_train}, % val: {perc_val_test}')

    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test)

    train_set = snap[0:train_size]
    val_set = snap[train_size:train_size + val_size]
    test_set = snap[train_size + val_size:]

    print('% Training set ...')
    count_add = 0
    count_remove = 0
    for elem in train_set:
        if elem[7] == 'Add':
            count_add += 1
        else:
            count_remove += 1

    print(count_add / (count_add + count_remove))
    print(count_remove / (count_add + count_remove))

    print('% Val set ...')
    count_add = 0
    count_remove = 0
    for elem in val_set:
        if elem[7] == 'Add':
            count_add += 1
        else:
            count_remove += 1

    print(count_add / (count_add + count_remove))
    print(count_remove / (count_add + count_remove))

    print('% Test set ...')
    count_add = 0
    count_remove = 0
    for elem in test_set:
        if elem[7] == 'Add':
            count_add += 1
        else:
            count_remove += 1

    print(count_add / (count_add + count_remove))
    print(count_remove / (count_add + count_remove))

    window_size = params['window_size']
    window_shift = params['window_shift']

    x_sequences_train, y_sequences_train = create_sequences(train_set, window_size, window_shift)
    x_sequences_val, y_sequences_val = create_sequences(val_set, window_size, window_shift)
    x_sequences_test, y_sequences_test = create_sequences(test_set, window_size, window_shift)

    snapshot_train = create_loader_remove(x_sequences_train, y_sequences_train, window_size)
    snapshot_val = create_loader_remove(x_sequences_val, y_sequences_val, window_size)
    snapshot_test = create_loader_remove(x_sequences_test, y_sequences_test, window_size)

    return snapshot_train, snapshot_test, snapshot_val, train_set, test_set, val_set


def load_data(params):
    file_name = params['file_name']
    window_size = params['window_size']
    window_shift = params['window_shift']
    path_generation = params['path_generation']
    path_regression = params['path_regression']
    create_data = params['create_data']
    perc_train = params['perc_train']
    dataset = params['dataset']

    if dataset == 'bitcoin_alpha_no_features':
        from utils.create_data import create_data_bitcoin_alpha

        print('Loading data ...')
        snapshot_train_gen, _, snapshot_val_gen, train_set, test_set, val_set = create_data_bitcoin_alpha(
            file_name, window_size=window_size, window_shift=window_shift, is_data=create_data,
            path_generation=path_generation, path_regression=path_regression, perc_train=perc_train, features=False
        )

    elif dataset == 'bitcoin_OTC_no_features':
        from utils.create_data import create_data_bitcoin

        snapshot_train_gen, _, snapshot_val_gen, train_set, test_set, val_set = create_data_bitcoin(
            file_name, window_size=window_size, window_shift=window_shift, is_data=create_data,
            path_generation=path_generation, path_regression=path_regression, perc_train=perc_train, features=False
        )

    elif dataset == 'eu_core_no_dupl':
        from utils.create_data import create_data_eu_core_no_dupl
        snapshot_train_gen, _, snapshot_val_gen, train_set, test_set, val_set = create_data_eu_core_no_dupl(
            file_name, window_size=window_size, window_shift=window_shift, is_data=create_data,
            path_generation=path_generation, path_regression=path_regression, perc_train=perc_train
        )

    elif dataset == 'synthetic_graph_power_law_with_remove_no_dupl':
        from utils.utils import load_synthetic
        snapshot_train_gen, _, snapshot_val_gen, train_set, test_set, val_set = load_synthetic(params)

    elif dataset == 'uci_forum_no_dupl':
        from utils.create_data import create_data_uci_forum

        snapshot_train_gen, _, snapshot_val_gen, train_set, test_set, val_set = create_data_uci_forum(
            file_name, window_size=window_size, window_shift=window_shift, is_data=create_data,
            path_generation=path_generation, path_regression=path_regression, perc_train=perc_train
        )

    else:
        print('No dataset available')
        sys.exit(-1)

    return snapshot_train_gen, snapshot_val_gen, train_set, test_set, val_set


def wl_test_run(WL_start, G_real, G_pred, diameter):
    from utils.wl_test import compute_similarity
    labels = {}

    if WL_start == 'degree':
        degrees = {}
        color_count = 0
        for g in [G_real, G_pred]:
            for i in range(g.number_of_nodes()):
                # print(g.degree(i))
                if g.degree(i) not in degrees.keys():
                    degrees[g.degree(i)] = color_count
                    color_count += 1

        for g in [G_real, G_pred]:
            labels[g] = {}
            for i in range(g.number_of_nodes()):
                labels[g][i] = degrees[g.degree(i)]

        X = []

        for g in [G_real, G_pred]:
            X.append([list(g.edges()), labels[g].copy()])

    else:
        max_nodes = max(G_real.number_of_nodes(), G_pred.number_of_nodes())
        for i in range(max_nodes):
            labels[i] = 0

        X = []
        for g in [G_real, G_pred]:
            X.append([list(g.edges()), labels.copy()])

    all_steps, same_coloring, max_diameter = compute_similarity(X, n_iter=diameter * 2,
                                                                graph_format="dictionary", max_diameter=diameter)

    return all_steps, same_coloring, max_diameter


def compute_wl_similarity(params, true_snapshot):
    space_wl_results = params['space_wl_results']
    dataset = params['dataset']
    exp = params['exp']
    num_generation = params['num_generation']
    type_set = params['type_set']
    diameter = params['diameter']

    path_degree = f'{space_wl_results}/sampling_wl_degree_{dataset}_{exp}_{type_set}.txt'
    open(path_degree, 'w').close()

    start, end, step, count = 0, 500, 500, 0
    n_chunk = int(num_generation / 500)

    for i in tqdm(range(n_chunk), total=n_chunk):
        generated_snapshot = torch.load(params['path_generated_graphs'] + '_' + str(i) + '_regressor.pt')
        true_snapshot_tmp = true_snapshot[start:end]

        for g_true, g_pred in zip(true_snapshot_tmp, generated_snapshot):
            g_true = to_networkx(g_true)
            g_pred = g_pred[0]

            _, same_coloring, max_diameter = wl_test_run('degree', g_true, g_pred, diameter)

            with open(path_degree, 'a') as filehandle:
                filehandle.write(f'\n {count}, {same_coloring[0][1]}, {max_diameter[0][1]} \n')

            count += 1

        start = end
        end += step

    generated_snapshot = torch.load(params['path_generated_graphs'] + '_last_regressor.pt')
    true_snapshot_tmp = true_snapshot[start:end]

    for g_true, g_pred in zip(true_snapshot_tmp, generated_snapshot):
        g_true = to_networkx(g_true)
        g_pred = g_pred[0]

        _, same_coloring, max_diameter = wl_test_run('degree', g_true, g_pred, diameter)

        with open(path_degree, 'a') as filehandle:
            filehandle.write(f'\n {count}, {same_coloring[0][1]}, {max_diameter[0][1]} \n')

        count += 1

    degree = pd.read_csv(path_degree, names=['count', 'same', 'max'], header=None)

    res_same_col = list(degree['same'].values)
    res_max_diam = list(degree['max'].values)

    print(f'Same coloring --> Mean: {np.mean(res_same_col)}, 90-perc: {np.percentile(res_same_col, 90)}')
    print(f'Max diameter --> Mean: {np.mean(res_max_diam)}, 90-perc: {np.percentile(res_max_diam, 90)}')

    print('--' * 30)


def run_generation_synthetic(trainer, params, history, num_generation):

    if params['model_name'] == 'fudge':
        trainer.gnn_history_event.load_state_dict(torch.load(params['gnn_history_event']))
        trainer.rnn_history_event.load_state_dict(torch.load(params['rnn_history_event']))
        trainer.event.load_state_dict(torch.load(params['event_network']))

        trainer.gnn_history_source.load_state_dict(torch.load(params['gnn_history_source']))
        trainer.rnn_history_source.load_state_dict(torch.load(params['rnn_history_source']))

        trainer.gnn_aux_add_source.load_state_dict(torch.load(params['gnn_aux_add_source']))
        trainer.net_source.load_state_dict(torch.load(params['net_source']))

        trainer.gnn_history_dest.load_state_dict(torch.load(params['gnn_history_dest']))
        trainer.rnn_history_dest.load_state_dict(torch.load(params['rnn_history_dest']))

        trainer.gnn_aux_add_dest.load_state_dict(torch.load(params['gnn_aux_add_dest']))
        trainer.net_dest.load_state_dict(torch.load(params['net_dest']))

        models_event = (trainer.gnn_history_event, trainer.rnn_history_event, trainer.event)
        models_add_source = (trainer.gnn_history_source, trainer.rnn_history_source, trainer.gnn_aux_add_source,
                             trainer.net_source)
        models_add_dest = (trainer.gnn_history_dest, trainer.rnn_history_dest, trainer.gnn_aux_add_dest,
                           trainer.net_dest)

    generated_snapshot = trainer.generate_graph_rnd_alternative(history, num_generation, models_event,
                                                                models_add_source, models_add_dest)

    torch.save(generated_snapshot, params['path_generated_graphs'])


