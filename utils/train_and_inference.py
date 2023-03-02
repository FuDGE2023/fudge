import torch

from model.trainer import Trainer
from utils.utils import run_generation_regressor_sampling, load_data

from tqdm import tqdm
import numpy as np
import pandas as pd


def run_real_world_datasets(params):
    window_size = params['window_size']
    window_shift = params['window_shift']
    perc_train = params['perc_train']
    f = params['f']
    checkpoint = params['checkpoint']
    dataset = params['dataset']
    folder_generation_graphs = params['folder_generation_graphs']
    space_plot_losses = params['space_plot_losses']

    split = f'split_{perc_train}_w_size_{window_size}_w_shift_{window_shift}_nlayers3'
    exp = f'fact_{f}_{split}_new'

    params['out_dim_encoder'] = params['z_dim_encoder'] + (params['num_layer_encoder'] - 1) * params[
        'h_dim_encoder']

    params['exp'] = exp

    params['gnn_history_source'] = f'{checkpoint}/gnn_history_source_' + dataset + '_' + exp + '.pt'
    params['rnn_history_source'] = f'{checkpoint}/rnn_history_source_' + dataset + '_' + exp + '.pt'
    params['gnn_aux_add_source'] = f'{checkpoint}/gnn_aux_add_source_' + dataset + '_' + exp + '.pt'
    params['net_source'] = f'{checkpoint}/net_source_' + dataset + '_' + exp + '.pt'
    params['gnn_history_dest'] = f'{checkpoint}/gnn_history_dest_' + dataset + '_' + exp + '.pt'
    params['rnn_history_dest'] = f'{checkpoint}/rnn_history_dest_' + dataset + '_' + exp + '.pt'
    params['gnn_aux_add_dest'] = f'{checkpoint}/gnn_aux_add_dest_' + dataset + '_' + exp + '.pt'
    params['net_dest'] = f'{checkpoint}/net_dest_' + dataset + '_' + exp + '.pt'
    params['plot_title'] = f'{space_plot_losses}/losses_{dataset}_{exp}.pdf'

    snapshot_train_gen, snapshot_val_gen, train_set, test_set, val_set = load_data(params)

    trainer = Trainer(params)
    print('Training ...')

    if params['train']:
        trainer.fit(snapshot_train_gen, snapshot_val_gen)

    if params['generate_validation']:
        print('Generate validation set ...')

        params[
            'path_generated_graphs'] = f'{folder_generation_graphs}/sampling_generated_{dataset}_{exp}_validation_set'

        count_check = 0
        num_generation = len(val_set)
        history = train_set[-window_size:]

        print('# generation: ', {num_generation})

        run_generation_regressor_sampling(trainer, train_set, history, num_generation, params,
                                          count_checkpoint=count_check)

    if params['generate_test']:
        print('Generate test set ...')
        params[
            'path_generated_graphs'] = f'{folder_generation_graphs}/sampling_generated_{dataset}_{exp}_test_set'

        count_check = 0
        num_generation = len(test_set)
        history = val_set[-window_size:]

        print('# generation: ', {num_generation})

        run_generation_regressor_sampling(trainer, train_set, history, num_generation, params,
                                          count_checkpoint=count_check)


def compute_metrics(params):
    from torch_geometric.utils.convert import to_networkx
    from utils.utils import graph_statistics, compute_wl_similarity

    window_size = params['window_size']
    window_shift = params['window_shift']
    perc_train = params['perc_train']
    f = params['f']
    dataset = params['dataset']
    folder_generation_graphs = params['folder_generation_graphs']

    split = f'split_{perc_train}_w_size_{window_size}_w_shift_{window_shift}_nlayers3'
    exp = f'fact_{f}_{split}_new'
    params['exp'] = exp

    _, _, _, test_set, val_set = load_data(params)

    if params['compute_wl_val']:
        print('Compute WL similarity on validation set ...')
        params[
            'path_generated_graphs'] = f'{folder_generation_graphs}/sampling_generated_{dataset}_{exp}_validation_set'

        num_generation = len(val_set)
        params['num_generation'] = num_generation

        ts = to_networkx(val_set[-1])
        diameter = graph_statistics(ts, True)
        params['diameter'] = diameter

        params['type_set'] = 'validation'
        compute_wl_similarity(params, val_set)

    if params['compute_wl_test']:
        print('Compute WL similarity on test set ...')
        params[
            'path_generated_graphs'] = f'{folder_generation_graphs}/sampling_generated_{dataset}_{exp}_test_set'

        num_generation = len(test_set)
        params['num_generation'] = num_generation

        ts = to_networkx(test_set[-1])
        diameter = graph_statistics(ts, True)
        params['diameter'] = diameter

        params['type_set'] = 'test'
        compute_wl_similarity(params, test_set)


def compute_metrics_synthetic(params):
    from torch_geometric.utils.convert import to_networkx, from_networkx
    from utils.utils import graph_statistics, graph_statistics_all_nx, wl_test_run
    from utils.plotter import cumulative_distribution_all

    window_size = params['window_size']
    window_shift = params['window_shift']
    perc_train = params['perc_train']
    f = params['f']
    dataset = params['dataset']
    folder_generation_graphs = params['folder_generation_graphs']
    space_wl_results = params['space_wl_results']

    split = f'split_{perc_train}_w_size_{window_size}_w_shift_{window_shift}_nlayers3'
    exp = f'fact_{f}_{split}_new'
    params['exp'] = exp

    print('Loading data ...')
    _, _, _, test_set, val_set = load_data(params)

    if params['compute_wl_val']:
        print('Compute WL similarity on validation set ...')
        params[
            'path_generated_graphs'] = f'{folder_generation_graphs}/sampling_generated_{dataset}_{exp}_validation_set'

        generated_snapshot = torch.load(params['path_generated_graphs'])

        pyg_g = from_networkx(generated_snapshot[-1][1])
        print('Last generated graph')
        print(pyg_g)

        print('Last test set graph')
        print(val_set[-1][4])

        print('Last generated graph')
        graph_statistics(generated_snapshot[-1][1], True)
        print('Last test set graph')
        diameter = graph_statistics(val_set[-1][1], True)

        graph_statistics_all_nx(val_set[-1][1], generated_snapshot[-1][1])

        cumulative_distribution_all(generated_snapshot[-1][1], val_set[-1][1], dataset)

        pyg_g = from_networkx(generated_snapshot[-1][1])
        ts = val_set[-1][4]

        _, same_coloring, max_diameter = wl_test_run('degree', val_set[-1][1], generated_snapshot[-1][1],
                                                             diameter)

        print(f'same col: {same_coloring[0][1]}, max diameter: {max_diameter[0, 1]}')

        path_degree = f'{space_wl_results}/wl_degree_{dataset}_{exp}_validation.txt'

        open(path_degree, 'w').close()

        count = 0

        for g_true, g_pred in tqdm(zip(val_set, generated_snapshot), total=len(test_set)):
            _, same_coloring, max_diameter = wl_test_run('degree', g_true[1], g_pred[1], diameter)

            with open(path_degree, 'a') as filehandle:
                filehandle.write(f'\n {count}, {same_coloring[0][1]}, {max_diameter[0][1]} \n')

            count += 1

        degree = pd.read_csv(path_degree, names=['count', 'same', 'max'], header=None)

        res_same_col = list(degree['same'].values)
        res_max_diam = list(degree['max'].values)

        print(f'Same coloring --> Mean: {np.mean(res_same_col)}, 90-perc: {np.percentile(res_same_col, 90)}')
        print(f'Max diameter --> Mean: {np.mean(res_max_diam)}, 90-perc: {np.percentile(res_max_diam, 90)}')

        print('--' * 30)

    if params['compute_wl_test']:
        print('Compute WL similarity on test set ...')
        params[
            'path_generated_graphs'] = f'{folder_generation_graphs}/sampling_generated_{dataset}_{exp}_test_set'

        generated_snapshot = torch.load(params['path_generated_graphs'])

        pyg_g = from_networkx(generated_snapshot[-1][1])
        print('Last generated graph')
        print(pyg_g)

        print('Last test set graph')
        print(test_set[-1][4])

        print('Last generated graph')
        graph_statistics(generated_snapshot[-1][1], True)
        print('Last test set graph')
        diameter = graph_statistics(test_set[-1][1], True)

        graph_statistics_all_nx(test_set[-1][1], generated_snapshot[-1][1])

        cumulative_distribution_all(generated_snapshot[-1][1], test_set[-1][1], dataset)

        pyg_g = from_networkx(generated_snapshot[-1][1])
        ts = test_set[-1][4]

        _, same_coloring, max_diameter = wl_test_run('degree', test_set[-1][1], generated_snapshot[-1][1],
                                                             diameter)

        print(f'same col: {same_coloring[0][1]}, max diameter: {max_diameter[0, 1]}')

        path_degree = f'{space_wl_results}/wl_degree_{dataset}_{exp}_test.txt'

        open(path_degree, 'w').close()

        count = 0

        for g_true, g_pred in tqdm(zip(test_set, generated_snapshot), total=len(test_set)):
            _, same_coloring, max_diameter = wl_test_run('degree', g_true[1], g_pred[1], diameter)

            with open(path_degree, 'a') as filehandle:
                filehandle.write(f'\n {count}, {same_coloring[0][1]}, {max_diameter[0][1]} \n')

            count += 1

        degree = pd.read_csv(path_degree, names=['count', 'same', 'max'], header=None)

        res_same_col = list(degree['same'].values)
        res_max_diam = list(degree['max'].values)

        print(f'Same coloring --> Mean: {np.mean(res_same_col)}, 90-perc: {np.percentile(res_same_col, 90)}')
        print(f'Max diameter --> Mean: {np.mean(res_max_diam)}, 90-perc: {np.percentile(res_max_diam, 90)}')

        print('--' * 30)


def run_synthetic_data(params):
    from model.trainer_remove import Trainer
    from utils.utils import run_generation_synthetic
    window_size = params['window_size']
    window_shift = params['window_shift']
    perc_train = params['perc_train']
    f = params['f']
    checkpoint = params['checkpoint']
    dataset = params['dataset']
    folder_generation_graphs = params['folder_generation_graphs']
    space_plot_losses = params['space_plot_losses']

    split = f'split_{perc_train}_w_size_{window_size}_w_shift_{window_shift}_nlayers3'
    exp = f'fact_{f}_{split}_new'

    params['out_dim_encoder'] = params['z_dim_encoder'] + (params['num_layer_encoder'] - 1) * params[
        'h_dim_encoder']

    params['exp'] = exp

    params['gnn_history_event'] = f'{checkpoint}/gnn_history_event_' + dataset + '_' + exp + '.pt'
    params['rnn_history_event'] = f'{checkpoint}/rnn_history_event_' + dataset + '_' + exp + '.pt'
    params['event_network'] = f'{checkpoint}/event_network' + dataset + '_' + exp + '.pt'
    params['gnn_history_source'] = f'{checkpoint}/gnn_history_source_' + dataset + '_' + exp + '.pt'
    params['rnn_history_source'] = f'{checkpoint}/rnn_history_source_' + dataset + '_' + exp + '.pt'
    params['gnn_aux_add_source'] = f'{checkpoint}/gnn_aux_add_source_' + dataset + '_' + exp + '.pt'
    params['net_source'] = f'{checkpoint}/net_source_' + dataset + '_' + exp + '.pt'
    params['gnn_history_dest'] = f'{checkpoint}/gnn_history_dest_' + dataset + '_' + exp + '.pt'
    params['rnn_history_dest'] = f'{checkpoint}/rnn_history_dest_' + dataset + '_' + exp + '.pt'
    params['gnn_aux_add_dest'] = f'{checkpoint}/gnn_aux_add_dest_' + dataset + '_' + exp + '.pt'
    params['net_dest'] = f'{checkpoint}/net_dest_' + dataset + '_' + exp + '.pt'
    params['plot_title'] = f'{space_plot_losses}/losses_{dataset}_{exp}.pdf'

    print('Loading data ...')
    snapshot_train_gen, snapshot_val_gen, train_set, test_set, val_set = load_data(params)

    trainer = Trainer(params)
    print('Training ...')
    if params['train']:
        trainer.fit(snapshot_train_gen, snapshot_val_gen)

    if params['generate_validation']:
        print('Generate validation set ...')

        params[
            'path_generated_graphs'] = f'{folder_generation_graphs}/sampling_generated_{dataset}_{exp}_validation_set'

        history = [snap[4] for snap in train_set[-window_size:]]
        num_generation = len(val_set)
        print('# generation: ', {num_generation})
        run_generation_synthetic(trainer, params, history, num_generation)

    if params['generate_test']:
        print('Generate test set ...')
        params[
            'path_generated_graphs'] = f'{folder_generation_graphs}/sampling_generated_{dataset}_{exp}_test_set'

        history = [snap[4] for snap in val_set[-window_size:]]
        num_generation = len(test_set)
        print('# generation: ', {num_generation})
        run_generation_synthetic(trainer, params, history, num_generation)


def get_results(params):
    if params['dataset'] != 'synthetic_graph_power_law_with_remove_no_dupl':
        compute_metrics(params)
    else:
        compute_metrics_synthetic(params)


def traininig_generation_phase(params):
    if params['dataset'] != 'synthetic_graph_power_law_with_remove_no_dupl':
        run_real_world_datasets(params)
    else:
        run_synthetic_data(params)





