import sys
sys.path.append("/home/mungari/Desktop/evograph/components/") # todo relative

import numpy as np
import torch
import networkx as nx
from time import time
import torch_geometric
from wl_test import compute_similarity
import os
from joblib import Parallel, delayed
from tqdm import tqdm

def loadList(filename):

    if "synthetic" in filename:
        # the filename should mention the extension 'npy'
        tempNumpyArray = np.load(filename, allow_pickle=True)
        snap = tempNumpyArray.tolist()
    else:
        snap = torch.load(filename)

    return snap

def parallel_WL_test(params, graphs_indexes, data):

    max_diameter = params['max_diameter']
    dataset = params['dataset']
    WL_start = params['WL_start']
    split_percentage = params['split_percentage']
    factor = params['factor']
    end_filename = params['end_filename']

    results = []
    for graph_index in tqdm(graphs_indexes):
    #     print(graph_index)

        G_pred = torch.load("/mnt/nas/mungari/evograph/generated_data/{}/generated_{}_graphs/NAT/{}_{}/graph_{}.pt".format(split_percentage, dataset, factor, end_filename, graph_index))
        G_pred = torch_geometric.utils.convert.to_networkx(G_pred)

        if params['WL_type'] == 'last_graph':
            G_real = data[-1]
        else:
            G_real = data[graph_index]
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


        res = compute_similarity(X, n_iter=max_diameter*2, graph_format="dictionary", similarity_metric="cosine",
                                 max_diameter=max_diameter)

        final_res = (res[0][0][1], res[1][0][1], res[2][0][1])
        results.append(final_res)

    return results

def compute_wl(params, data):

    if params['WL_type'] == 'last_graph':
        params['num_generations'] = 1

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


    if not os.path.exists("/mnt/nas/mungari/evograph/wl_results/{}/".format(params['split_percentage'])):
        os.makedirs("/mnt/nas/mungari/evograph/wl_results/{}".format(params['split_percentage']))

    if not os.path.exists("results/"):
        os.makedirs("results/")

    f = open("/mnt/nas/mungari/evograph/wl_results/{}/{}_NAT_{}_{}.tsv".format(params['split_percentage'], params['dataset'], params['factor'], params['end_filename']), "w")
    f.write("Count\tAll_Step\tColoring_Stable\tDiameter\n")
    for i in range(len(results_all_steps)):
        f.write("{}\t{}\t{}\t{}\n".format(i, results_all_steps[i], results_coloring_stable[i], results_diameter[i]))
    f.close()

    if not os.path.exists("{}_{}_{}_results_{}.tsv".format(params['split_percentage'], params['dataset'], params['factor'], params['end_filename'])):
        f = open("{}_{}_{}_results_{}.tsv".format(params['split_percentage'], params['dataset'], params['factor'], params['end_filename']), "w")
        f.write("Baseline\tWL_Start\tWL_Type\tStopCondition\tMean\t90Percentile\n")
        f.close()

    f = open("{}_{}_{}_results_{}.tsv".format(params['split_percentage'], params['dataset'], params['factor'], params['end_filename']), "a")
    f.write("NAT\t{}\t{}\t{}\t{}\t{}\n".format(params['WL_start'], params['WL_type'],
                                              "All Steps", mean_all_steps, percentile_all_steps))
    f.write("NAT\t{}\t{}\t{}\t{}\t{}\n".format(params['WL_start'], params['WL_type'],
                                              "Coloring Stable", mean_coloring_stable, percentile_coloring_stable))
    f.write("NAT\t{}\t{}\t{}\t{}\t{}\n".format(params['WL_start'], params['WL_type'],
                                              "Diameter", mean_diameter, percentile_diameter))
    f.close()

datasets = sys.argv[1].split("-")
dataset_extensions_list = sys.argv[2].split("-")
users = sys.argv[3]
split_percentages = [float(f) for f in sys.argv[4].split("-")]
ngh_dims = sys.argv[6].split("-")
n_degrees = [x.split(",") for x in sys.argv[5].split("-")]
end_filename = "val" if sys.argv[10] == "True" else "test"

print(sys.argv)

for i in range(len(split_percentages)):
    train_percentage = float(split_percentages[i])
    for j in range(len(datasets)):
        for ngh_dim in ngh_dims:
            ngh_dim = int(ngh_dim)
            for n_degree in n_degrees:
                n_degree = [int(x) for x in n_degree]

                factor = "{}_{}_{}".format(ngh_dim, n_degree[0], n_degree[1])
                dataset = datasets[j]
                dataset_extension = dataset_extensions_list[j]
                split_percentage = split_percentages[i]
                # factor = factors_list[k]
                print(dataset, dataset_extension, split_percentage, factor, end_filename)

                snap=loadList('/mnt/nas/{}/evograph/data/{}'.format(users, dataset+dataset_extension))

                seed = 12345


                #DEBUG
                # snap = snap[:1000]
                # snap = np.array(snap)

                amount = len(snap)
                print(amount)

                perc_train = split_percentage
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
                params['metric'] = sys.argv[7] # WL, MMD
                params['WL_start'] = sys.argv[8] #'degree' # uniform, degree
                params['WL_type'] = sys.argv[9] #'averaged' # averaged, last_graph
                params['seed'] = seed
                params['dataset'] = dataset
                params['num_generations'] = len(val_set) if end_filename == "val" else len(test_set)
                params['split_percentage'] = split_percentage
                params['factor'] = factor
                params['end_filename'] = end_filename

                if end_filename == "val":
                    compute_wl(params, val_set)
                else:
                    compute_wl(params, test_set)
