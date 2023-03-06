import networkx as nx
import numpy as np
from utils.wl_test import compute_similarity
from tqdm import tqdm
import torch
import torch_geometric
from time import time

def Loss(x, n, m, G_real):

    G_pred = powerlaw_cluster_graph(n, m, x, G_real.copy(), "", None, save_graphs=False)[0]
    #G_pred = torch_geometric.utils.convert.to_networkx(G_pred)

    # print(G_pred)
    max_nodes = max([G_real.number_of_nodes(), G_pred.number_of_nodes()])

    labels = {}
    for i in range(max_nodes):
        labels[i] = 0

    X = []
    for g in [G_real, G_pred]:
        X.append([list(g.edges()), labels.copy()])

    diameters = []
    for G in [G_real, G_pred]:
        diameter = max([max(j.values()) for (i, j) in nx.shortest_path_length(G)])
        diameters.append(diameter)

    max_diameter = max(diameters)
    # print(x)

    res = compute_similarity(X, n_iter=10, graph_format="dictionary", similarity_metric="cosine",
                                     max_diameter=max_diameter)
    return res[0][0][1] # max iterations

def optimizer_brute(x_min, x_max, x_step, n, m, G_real):
    loss_all = []
    x_list = np.arange(x_min,x_max,x_step)
    for x_test in tqdm(x_list):
        loss_all.append(Loss(x_test, n, m, G_real))

    x_optim = x_list[np.argmin(np.array(loss_all))]
    return x_optim

def _random_subset(seq, m):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = np.random.choice(seq)
        targets.add(x)
    return targets

def powerlaw_cluster_graph(num_generations, m, p, G, full_generation_path, regressor, save_graphs=True):

    graphs = []

    repeated_nodes = list(G.nodes())  # list of existing nodes to sample from
    # with nodes repeated once for each adjacent edge
    source = G.number_of_nodes()  # next node is m
    count_graphs = 0
    # while source < n:  # Now add the other n-1 nodes
    if regressor is None:
        n_operations_list = np.ones(num_generations).astype(int)
    else:
        n_operations_list = regressor.sample(num_generations)

    for i in range(num_generations):

        for j in range(n_operations_list[i]):
            possible_targets = _random_subset(repeated_nodes, m)
            # do one preferential attachment for new node
            target = possible_targets.pop()
            G.add_edge(source, target)
            repeated_nodes.append(target)  # add one node to list for each new link
            count = 1
            while count < m:  # add m-1 more new links
                if seed.random() < p:  # clustering step: add triangle
                    neighborhood = [
                        nbr
                        for nbr in G.neighbors(target)
                        if not G.has_edge(source, nbr) and not nbr == source
                    ]
                    if neighborhood:  # if there is a neighbor without a link
                        nbr = seed.choice(neighborhood)
                        G.add_edge(source, nbr)  # add triangle
                        repeated_nodes.append(nbr)
                        count = count + 1
                        continue  # go to top of while loop
                # else do preferential attachment step if above fails
                target = possible_targets.pop()
                G.add_edge(source, target)
                repeated_nodes.append(target)
                count = count + 1
            repeated_nodes.extend([source] * m)  # add source node to list m times
            source += 1

        if save_graphs:
            pyg = torch_geometric.utils.convert.from_networkx(G)

            pyg.x = torch.ones((G.number_of_nodes(), 1))
            edge_index_t = torch.cat((pyg.edge_index[1].reshape(1, pyg.edge_index[1].shape[0]),
                                      pyg.edge_index[0].reshape(1, pyg.edge_index[0].shape[0])),
                                     dim=0)
            pyg.edge_index_t = edge_index_t
            pyg.n_operations = n_operations_list[i]

            torch.save(pyg, "{}/POWER/graph_{}.pt".format(full_generation_path, count_graphs))
        else:
            graphs.append(G.copy())

        count_graphs += 1

    return graphs

def power_generator(train_graph, history, params):

    seed = params['seed']
    num_generations = params['num_generations']
    regressor = params['regressor']
    full_generation_path = params['full_generation_path']

    torch.manual_seed(seed)
    np.random.seed(seed)


    num_generations_opt = 1
    m = 1
    p = optimizer_brute(0.0, 1.0, 0.1, num_generations_opt, m, train_graph)

    generated_graphs = powerlaw_cluster_graph(num_generations, m, p, history.copy(), full_generation_path, regressor)

    return generated_graphs
