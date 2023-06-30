import sys

import numpy as np
import torch
import networkx as nx
import torch_geometric

from scipy.stats import ks_2samp

np.random.seed(12345)

'''
To run:
python3 compute_ks.py dataset_name extensions perc_train f bs metrics models

dataset_name: 
    - snapshots_regression_{dataset} --> dataset: [uci_forum_no_dupl, bitcoin_alpha_no_features, bitcoin_OTC_no_features, eu_core_no_dupl] 
    - synthetic_graph_power_law_with_remove_no_dupl

extensions:
    - for all the dataset .pt, except for synthetic (.npy)

perc_train: [0.7, 0.6, 0.5]

f: 
    - FUDGE: [4, 5, 6]
    - NAT: see configuration file. Example 2_16_16
    - TGL: [DySAT, TGN, TGAT, JODIE]
    - TGN: [identity, time, graph_attention, graph_sum]
    
bs:
    - 0 for the competitors
    - for FUDGE depends on the datasets (see configuration file)
    
metrics: [in_degree, out_degree, clustering]

models: [FUDGE, NAT, TGL, TGN]

'''


def get_distributions(distribution, test_graph, model_graph):
    if distribution == "in_degree":
        degrees = [test_graph.in_degree(n) for n in test_graph.nodes()]
        counts_test = np.bincount(degrees)

        degrees = [model_graph.in_degree(n) for n in model_graph.nodes()]
        counts_model = np.bincount(degrees)

        if len(counts_model) < len(counts_test):
            counts_model = np.pad(counts_model, (0, len(counts_test) - len(counts_model)))
        elif len(counts_test) < len(counts_model):
            counts_test = np.pad(counts_test, (0, len(counts_model) - len(counts_test)))

        cdf_test = np.cumsum(counts_test[::-1])[::-1].astype('float64')
        cdf_model = np.cumsum(counts_model[::-1])[::-1].astype('float64')

        cdf_test /= cdf_test[0]
        cdf_model /= cdf_model[0]

        return cdf_test, cdf_model
    elif distribution == "out_degree":
        degrees = [test_graph.out_degree(n) for n in test_graph.nodes()]
        counts_test = np.bincount(degrees)

        degrees = [model_graph.out_degree(n) for n in model_graph.nodes()]
        counts_model = np.bincount(degrees)

        if len(counts_model) < len(counts_test):
            counts_model = np.pad(counts_model, (0, len(counts_test) - len(counts_model)))
        elif len(counts_test) < len(counts_model):
            counts_test = np.pad(counts_test, (0, len(counts_model) - len(counts_test)))

        cdf_test = np.cumsum(counts_test[::-1])[::-1].astype('float64')
        cdf_model = np.cumsum(counts_model[::-1])[::-1].astype('float64')

        cdf_test /= cdf_test[0]
        cdf_model /= cdf_model[0]

        return cdf_test, cdf_model
    elif distribution == "clustering":
        clusterings_test = list(nx.clustering(test_graph).values())
        clusterings_model = list(nx.clustering(model_graph).values())

        return clusterings_test, clusterings_model


def compute_ks(params, test_graph, model_graph):

    stats = []
    p_values = []
    results = []
    alpha = 0.05

    for i in range(len(params['distributions'])):
        sample1, sample2 = get_distributions(params['distributions'][i], test_graph, model_graph)


        if params['distributions'][i] == "clustering":
            stat, p_value = ks_2samp(sample1, sample2)
        else:
            stat = abs(sample1-sample2).max()
            p_value = 0

        result = "SAME DISTRIBUTIONS" if p_value >= alpha else "DIFFERENT DISTRIBUTIONS"

        print("model:{}\tdistribution:{}\tstat: {}".format(params['distributions'][i], params['model'], stat))
        stats.append(stat)
        p_values.append(p_value)
        results.append(result)


def loadList(filename):

    if "synthetic" in filename:
        # the filename should mention the extension 'npy'
        tempNumpyArray = np.load(filename, allow_pickle=True)
        snap = tempNumpyArray.tolist()
    else:
        snap = torch.load(filename)

    return snap


def load_models_graphs(model, dataset, split_percentage, factor, id_last_graph, bs=32):
    if model != "FUDGE":
        if 'generation_bitcoin' in dataset:
            if model in ['TGN', 'TGL', 'NAT']:
                dataset = "snapshots_generation_bitcoin_otc_edge_weights"
            else:
                dataset = "snapshots_generation_bitcoin_OTC_no_features"
        elif 'regression_bitcoin' in dataset:
            if model in ['TGN', 'TGL', 'NAT']:
                dataset = "snapshots_regression_bitcoin_alpha_edge_weights"
            else:
                dataset = "snapshots_regression_bitcoin_alpha_no_features"

        if model in ['TGN', 'TGL', 'NAT']:
            path = "./generated_data/{}/generated_{}_graphs/{}/{}_test/graph_{}.pt" \
                .format(split_percentage, dataset, model, factor, id_last_graph)
        else:
            path = "./generated_data/{}/generated_{}_graphs/{}/graph_{}.pt" \
                .format(split_percentage, dataset, model, id_last_graph)

        generated_snapshot = torch.load(path)
    else:
        generated_graphs = './generated_graphs'

        split = f'split_{perc_train}_w_size_30_w_shift_10_bs_{bs}_is_sampling_1'
        exp = f'fact_{factor}_{split}_ICDM2023'

        dataset_only_name = ""

        if dataset == "snapshots_regression_eu_core_no_dupl":
            dataset_only_name = "eu_core"
            dataset = "eu_core_no_dupl"
        elif dataset == "snapshots_regression_uci_forum_no_dupl":
            dataset_only_name = "uci_forum"
            dataset = "uci_forum_no_dupl"
        elif dataset == "snapshots_regression_bitcoin_alpha_no_features":
            dataset_only_name = "bitcoin_alpha"
            dataset = "bitcoin_alpha_no_features"
        elif dataset == "snapshots_regression_bitcoin_OTC_no_features":
            dataset_only_name = "bitcoin_otc"
            dataset = "bitcoin_OTC_no_features"

        if dataset == "synthetic_graph_power_law_with_remove_no_dupl":
            path_generated = f'{generated_graphs}/{dataset}/sampling_generated_{dataset}_{exp}_test_set.pt'
            generated_snapshot = torch.load(path_generated)[-1][1]
        else:
            params['path_generated_test_set'] = f'{generated_graphs}/{dataset}/sampling_generated_{dataset}_{exp}_test_set'

            generated_snapshot = torch.load(params['path_generated_test_set'] + '_last_regressor.pt')[-1][2]

    return generated_snapshot


if __name__ == '__main__':
    datasets = sys.argv[1].split("-")
    dataset_extensions_list = sys.argv[2].split("-")
    split_percentages = [float(f) for f in sys.argv[3].split("-")]
    factors_list = sys.argv[4].split("-")
    bs = [int(f) for f in sys.argv[5].split("-")][0]
    distributions = sys.argv[6].split("-")
    models = sys.argv[7].split("-")

    for i in range(len(split_percentages)):
        for j in range(len(datasets)):
            dataset = datasets[j]
            dataset_extension = dataset_extensions_list[j]
            split_percentage = split_percentages[i]
            print(dataset, dataset_extension, split_percentage, factors_list, bs, distributions)

            snap = loadList('./data/{}'.format(dataset + dataset_extension))

            seed = 12345

            amount = len(snap)
            print(amount)

            perc_train = split_percentage
            perc_val_test = (1. - perc_train) / 2.0

            train_size = int(amount * perc_train)
            val_size = int(amount * perc_val_test)

            train_set = snap[0:train_size]
            val_set = snap[train_size:train_size + val_size]
            test_set = snap[train_size + val_size:]

            if type(snap[0]) == list:
                snap = np.array(snap)[:, 4]
                train_set = snap[0:train_size]
                val_set = snap[train_size:train_size + val_size]
                test_set = snap[train_size + val_size:]

            params = {}
            params['dataset'] = dataset
            params['split_percentage'] = split_percentage
            params['factors'] = factors_list
            params['distributions'] = distributions

            id_last_graph = len(test_set) - 1
            print(id_last_graph)

            for k in range(len(models)):
                params['model'] = models[k]
                model_graph = load_models_graphs(models[k], dataset, split_percentage, factors_list[k], id_last_graph,
                                                 bs=bs)
                if "synthetic" not in dataset or ("synthetic" in dataset and models[k] != "FUDGE"):
                    model_graph = torch_geometric.utils.convert.to_networkx(model_graph)

                compute_ks(params, torch_geometric.utils.convert.to_networkx(test_set[-1]), model_graph)
