import sys
import numpy as np
import torch
import networkx as nx
from time import time
import torch_geometric
import os
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.data_processing import loadList
import argparse

import numpy as np
from six import itervalues

from grakel.kernels.vertex_histogram import VertexHistogram
from grakel.graph import Graph

from scipy.special import rel_entr

def generate_graphs(label_count, WL_labels_inverse, n_iter, nx, L, Gs_ed, extras, _inv_labels, in_edges_mapping=False, max_diameter=5):
    count_colors_before = 0
    count_colors_now = 0
    count_colors = True
    count_colors_iteration_stop = -1

    check_diameter = True
    diameter_iteration_stop = -1
    new_graphs = list()
    for j in range(nx):
        new_labels = dict()
        for k in L[j].keys():
            new_labels[k] = WL_labels_inverse[L[j][k]]
        L[j] = new_labels
        # add new labels
        new_graphs.append((Gs_ed[j], new_labels) + extras[j])
    yield new_graphs, count_colors_iteration_stop, diameter_iteration_stop

    for i in range(1, n_iter):
        if i == max_diameter + 1 and check_diameter:  # max_diameter+1 because the loop starts from 1
            # print("DIAMETER ARRIVED")
            diameter_iteration_stop = i
            check_diameter = False
        label_set, WL_labels_inverse, L_temp = set(), dict(), dict()
        for j in range(nx):
            # Find unique labels and sort
            # them for both graphs
            # Keep for each node the temporary
            L_temp[j] = dict()

            for v in Gs_ed[j].keys():
                credential = str(L[j][v]) + "," + str(sorted([L[j][n] for n in Gs_ed[j][v].keys()]))

                if in_edges_mapping:
                    in_edges = []
                    for n in Gs_ed[j].keys():
                        if v in Gs_ed[j][n].keys():
                            in_edges.append(L[j][n])
                    credential += "," + str(sorted(in_edges))

                L_temp[j][v] = credential
                label_set.add(credential)

        label_list = sorted(list(label_set))
        for dv in label_list:
            WL_labels_inverse[dv] = label_count
            label_count += 1

        count_colors_now = len(WL_labels_inverse.keys())
        if count_colors_now == count_colors_before and count_colors:
            # print("SAME COLOUR MAPPINGS")
            count_colors_iteration_stop = i
            count_colors = False

        count_colors_before = count_colors_now

        # Recalculate labels
        new_graphs = list()
        for j in range(nx):
            new_labels = dict()
            for k in L_temp[j].keys():
                new_labels[k] = WL_labels_inverse[L_temp[j][k]]
            L[j] = new_labels
            # relabel
            new_graphs.append((Gs_ed[j], new_labels) + extras[j])
        _inv_labels[i] = WL_labels_inverse
        yield new_graphs, count_colors_iteration_stop, diameter_iteration_stop


def compute_metric(kernels, similarity_metric="cosine", steps=50):
    K = np.sum(np.array(kernels), axis=0)
    similarities = np.zeros(K.shape)

    # normalizer = Normalizer()

    if similarity_metric == "cosine":
        _X_diag = np.diagonal(K)
        similarities = K / np.sqrt(np.outer(_X_diag, _X_diag))
        similarities = np.around(similarities, decimals=4)

    elif similarity_metric == "kl":
        for i in range(len(K)):
            for j in range(len(K)):
                kl_1 = sum(rel_entr(K[i], K[j]))
                kl_2 = sum(rel_entr(K[j], K[i]))
                similarities[i][j] = np.around((kl_1 + kl_2) / 2, decimals=4)
    #         similarities = normalizer.fit_transform(similarities)

    elif similarity_metric == "l2":
        for i in range(len(K)):
            for j in range(len(K)):
                similarities[i][j] = np.around(np.linalg.norm(K[i] - K[j]), decimals=4)
    #         similarities = normalizer.fit_transform(similarities)

    return similarities


def compute_similarity(X, n_iter=10, graph_format="dictionary", similarity_metric="cosine", max_diameter=5, in_edges_mapping=False):
    nx = 0
    Gs_ed, L, distinct_values, extras = dict(), dict(), set(), dict()
    Xs = []

    for (idx, x) in enumerate(iter(X)):

        if len(x) > 2:
            extra = tuple()
            if len(x) > 3:
                extra = tuple(x[3:])
            x = Graph(x[0], x[1], x[2], graph_format=graph_format)
            extra = (x.get_labels(purpose=graph_format,
                                  label_type="edge", return_none=True),) + extra
        else:
            x = Graph(x[0], x[1], {}, graph_format=graph_format)
            extra = tuple()

        Xs.append(x)
        Gs_ed[nx] = x.get_edge_dictionary()
        L[nx] = x.get_labels(purpose="dictionary")
        extras[nx] = extra
        distinct_values |= set(itervalues(L[nx]))
        nx += 1

    # Save the number of "fitted" graphs.
    _nx = nx

    # get all the distinct values of current labels
    WL_labels_inverse = dict()

    # assign a number to each label
    label_count = 0
    for dv in sorted(list(distinct_values)):
        WL_labels_inverse[dv] = label_count
        label_count += 1

    # Initalize an inverse dictionary of labels for all iterations
    _inv_labels = dict()
    _inv_labels[0] = WL_labels_inverse

    base_kernels = {i: VertexHistogram() for i in range(n_iter)}
    generated_graphs_kernels = []

    same_coloring_step = 1
    max_diameter_step = 1


    for (i, values) in enumerate(generate_graphs(label_count, WL_labels_inverse, n_iter, nx, L, Gs_ed, extras, _inv_labels,
                                                 in_edges_mapping=in_edges_mapping, max_diameter=max_diameter)):

        g, same_coloring_step_temp, max_diameter_step_temp = values

        if same_coloring_step_temp != -1:
            same_coloring_step = same_coloring_step_temp
        if max_diameter_step_temp != -1:
            max_diameter_step = max_diameter_step_temp

        k = base_kernels[i].fit_transform(g)
        generated_graphs_kernels.append(k)

    all_steps = compute_metric(kernels=generated_graphs_kernels, similarity_metric=similarity_metric, steps=n_iter)
    same_coloring = compute_metric(kernels=generated_graphs_kernels[:same_coloring_step], similarity_metric=similarity_metric, steps=n_iter)
    max_diameter = compute_metric(kernels=generated_graphs_kernels[:max_diameter_step], similarity_metric=similarity_metric, steps=n_iter)

    return all_steps, same_coloring, max_diameter

def parallel_WL_test(params, graphs_indexes, data):

    max_diameter = params['max_diameter']
    dataset = params['dataset']
    perc_train = params['perc_train']
    factor = params['factor']
    end_filename = params['end_filename']

    results = []
    for graph_index in tqdm(graphs_indexes):
    #     print(graph_index)

        G_pred = torch.load("./generated_data/{}/generated_{}_graphs/TGN/{}_{}/graph_{}.pt".format(perc_train, dataset, factor, end_filename, graph_index))
        G_pred = torch_geometric.utils.convert.to_networkx(G_pred)

        G_real = data[graph_index]
        G_real = torch_geometric.utils.convert.to_networkx(G_real)

        max_nodes = max([G_real.number_of_nodes(), G_pred.number_of_nodes()])

        labels = {}

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


        res = compute_similarity(X, n_iter=max_diameter*2, graph_format="dictionary", similarity_metric="cosine",
                                 max_diameter=max_diameter)

        final_res = (res[0][0][1], res[1][0][1], res[2][0][1])
        results.append(final_res)

    return results

def compute_wl(params, data):


    n_jobs = 1

    graphs_indexes = np.array_split(np.arange(params['num_generations']), n_jobs)

    last_g_networkx = torch_geometric.utils.convert.to_networkx(data[-1])
    max_diameter = max([max(j.values()) for (i, j) in nx.shortest_path_length(last_g_networkx)])
    params['max_diameter'] = max_diameter

    t = time()

    results = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(parallel_WL_test)(params, i, data) for i in graphs_indexes)  # range(params['num_generations']))
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


    if not os.path.exists("./wl_results/{}/".format(params['perc_train'])):
        os.makedirs("./wl_results/{}".format(params['perc_train']))

    f = open("./wl_results/{}/{}_{}_{}.tsv".format(params['perc_train'], params['dataset'], params['factor'], params['end_filename']), "w")
    f.write("Count\tAll_Step\tColoring_Stable\tDiameter\n")
    for i in range(len(results_all_steps)):
        f.write("{}\t{}\t{}\t{}\n".format(i, results_all_steps[i], results_coloring_stable[i], results_diameter[i]))
    f.close()


parser = argparse.ArgumentParser('Interface for TGN framework')
parser.add_argument('--data', type=str)
parser.add_argument('--extension', type=str)
parser.add_argument('--perc_train', type=float)
parser.add_argument('--factor', type=str)
parser.add_argument('--validation_set', type=str)


args = parser.parse_args()

dataset = args.data
dataset_extension = args.extension
perc_train = args.perc_train
factor = args.factor
end_filename = "val" if args.validation_set == "True" else "test"

print(sys.argv)

snap=loadList('../../data/{}'.format(dataset+dataset_extension))

seed = 12345

amount = len(snap)

perc_val_test = (1 - perc_train) / 2.0

train_size = int(amount * perc_train)
val_size = int(amount * perc_val_test)

train_set = snap[0:train_size]
val_set = snap[train_size:train_size + val_size]
test_set = snap[train_size + val_size: ]

# print(test_set[0])
if type(snap[0]) == list:
    snap = np.array(snap)[:, 4]
    train_set = snap[0:train_size]
    val_set = snap[train_size:train_size + val_size]
    test_set = snap[train_size + val_size:]

# print(test_set[0])
# print(test_set)

params = {}
params['seed'] = seed
params['dataset'] = dataset
params['num_generations'] = len(val_set) if end_filename == "val" else len(test_set)
params['perc_train'] = perc_train
params['factor'] = factor
params['end_filename'] = end_filename

if end_filename == "val":
    compute_wl(params, val_set)
else:
    compute_wl(params, test_set)
