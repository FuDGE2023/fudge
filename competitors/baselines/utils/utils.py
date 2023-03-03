import numpy as np
import torch
import networkx as nx
from time import time
import torch_geometric
from utils.wl_test import compute_similarity
import os
from joblib import Parallel, delayed
from collections import Counter
from tqdm import tqdm
from torch_geometric.utils.convert import to_networkx

def graph_statistics(G, torch_g=False):
    if not torch_g:
        G = to_networkx(G, to_undirected=False)

    print('Max diameter in G: ')
    diameter = max([max(j.values()) for (i, j) in nx.shortest_path_length(G)])
    print(diameter)
    print('-----' * 20)

    print('Average coef. clustering: ', nx.average_clustering(G))

def loadList(filename):

    if "synthetic" in filename:
        # the filename should mention the extension 'npy'
        tempNumpyArray = np.load(filename, allow_pickle=True)
        snap = tempNumpyArray.tolist()
    else:
        snap = torch.load(filename)

    return snap

def create_histogram(input_dict):
    counts = Counter(d for n, d in input_dict.items())
    # print(counts)
    for i in range(max(counts) + 1):
        counts.get(i, 0)
    return [counts.get(i, 0) for i in range(max(counts) + 1)]


def parallel_WL_test(params, graphs_indexes, test_set):

    max_diameter = params['max_diameter']
    baseline = params['baseline']
    WL_start = params['WL_start']
    full_generation_path = params['full_generation_path']

    results = []
    for graph_index in tqdm(graphs_indexes):
    #     print(graph_index)

        G_pred = torch.load("{}/{}/graph_{}.pt".format(full_generation_path, baseline, graph_index))
        G_pred = torch_geometric.utils.convert.to_networkx(G_pred)

        if params['WL_type'] == 'last_graph':
            G_real = test_set[-1]
        else:
            G_real = test_set[graph_index]
        G_real = torch_geometric.utils.convert.to_networkx(G_real)

        max_nodes = max([G_real.number_of_nodes(), G_pred.number_of_nodes()])

        labels = {}

        if WL_start == 'degree':

            degrees = {}
            color_count = 0
            for g in [G_real, G_pred]:
                for i in np.unique(g.nodes()):
                    # print(g.degree(i))
                    if g.degree(i) not in degrees.keys():
                        degrees[g.degree(i)] = color_count
                        color_count += 1

            for g in [G_real, G_pred]:
                labels[g] = {}
                for i in np.unique(g.nodes()):
                    labels[g][i] = degrees[g.degree(i)]

            X = []

            for g in [G_real, G_pred]:
                X.append([list(g.edges()), labels[g].copy()])

        elif WL_start == 'uniform':
            for i in range(max_nodes):
                labels[i] = 0

            X = []
            for g in [G_real, G_pred]:
                X.append([list(g.edges()), labels.copy()])


        res = compute_similarity(X, n_iter=max_diameter * 2, graph_format="dictionary", similarity_metric="cosine",
                                 max_diameter=max_diameter)

        final_res = (res[0][0][1], res[1][0][1], res[2][0][1])
        results.append(final_res)

    return results

def compute_wl(params, test_set):

    if params['WL_type'] == 'last_graph':
        params['num_generations'] = 1

    n_jobs = 1

    graphs_indexes = np.array_split(np.arange(params['num_generations']), n_jobs)

    last_g_networkx = torch_geometric.utils.convert.to_networkx(test_set[-1])
    max_diameter = max([max(j.values()) for (i, j) in nx.shortest_path_length(last_g_networkx)])
    params['max_diameter'] = max_diameter

    t = time()

    results = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(parallel_WL_test)(params, i, test_set) for i in graphs_indexes)
    print("End wl test", time() - t)

    results_all_steps = []
    results_coloring_stable = []
    results_diameter = []
    for x in results:
        x = np.array(x)
        results_all_steps.extend(x[:, 0])
        results_coloring_stable.extend(x[:, 1])
        results_diameter.extend(x[:, 2])

    results_all_steps = np.array(results_all_steps)
    mean_all_steps = np.around(np.mean(results_all_steps), decimals=4)
    percentile_all_steps = np.around(np.percentile(results_all_steps, 90), decimals=4)

    results_coloring_stable = np.array(results_coloring_stable)
    mean_coloring_stable = np.around(np.mean(results_coloring_stable), decimals=4)
    percentile_coloring_stable = np.around(np.percentile(results_coloring_stable, 90), decimals=4)

    results_diameter = np.array(results_diameter)
    mean_diameter = np.around(np.mean(results_diameter), decimals=4)
    percentile_diameter = np.around(np.percentile(results_diameter, 90), decimals=4)

    print("All Steps -> Mean: {}\t Percentile: {}\n"
          "Coloring Stable -> Mean: {}\t Percentile: {}\n"
          "Diameter -> Mean: {}\t Percentile: {}\n".format(mean_all_steps, percentile_all_steps,
                                                           mean_coloring_stable, percentile_coloring_stable,
                                                           mean_diameter, percentile_diameter))


    if not os.path.exists("{}/{}".format(params['space_wl_results'], params['perc_train'])):
        os.makedirs("{}/{}".format(params['space_wl_results'], params['perc_train']))


    f = open("{}/{}/{}_{}.tsv".format(params['space_wl_results'], params['perc_train'], params['dataset'], params['baseline']), "w")
    f.write("Count\tAll_Step\tColoring_Stable\tDiameter\n")
    for i in range(len(results_all_steps)):
        f.write("{}\t{}\t{}\t{}\n".format(i, results_all_steps[i], results_coloring_stable[i], results_diameter[i]))
    f.close()


