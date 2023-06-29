import networkx as nx
import torch
import torch_geometric
from utils.wl_test import compute_similarity
import numpy as np
from tqdm import tqdm

def Loss(x, n, G_real, seed):

    G_pred = nx.fast_gnp_random_graph(n,x, seed=seed, directed=True)

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

def optimizer_brute(x_min, x_max, x_step, n, G_real, seed):
    loss_all = []
    x_list = np.arange(x_min,x_max,x_step)
    for x_test in tqdm(x_list):
        loss_all.append(Loss(x_test, n, G_real, seed))

    x_optim = x_list[np.argmin(np.array(loss_all))]
    return x_optim

def erdos_renyi_generator(train_graph, history, params):

    seed = params['seed']
    num_generations = params['num_generations']
    regressor = params['regressor']
    full_generation_path = params['full_generation_path']

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = train_graph.number_of_nodes()
    p = optimizer_brute(0.001, 0.02, 0.001, n, train_graph, seed)

    n_operations_list = regressor.sample(num_generations)

    for i in range(num_generations):

        n = history.number_of_nodes() + n_operations_list[i]

        graph = nx.fast_gnp_random_graph(n, p, seed=seed, directed=True)

        pyg = torch_geometric.utils.convert.from_networkx(graph)

        pyg.x = torch.ones((graph.number_of_nodes(), 1))
        edge_index_t = torch.cat((pyg.edge_index[1].reshape(1, pyg.edge_index[1].shape[0]),
                                  pyg.edge_index[0].reshape(1, pyg.edge_index[0].shape[0])),
                                 dim=0)
        pyg.edge_index_t = edge_index_t
        pyg.n_operations = n_operations_list[i]

        history = graph

        torch.save(pyg, "{}/ER/graph_{}.pt".format(full_generation_path, i))

    print("Graphs saved")


