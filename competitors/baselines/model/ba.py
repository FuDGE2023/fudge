import torch
import torch_geometric
import numpy as np
from time import time

def _random_subset(seq, m, rng):
    targets = set()
    while len(targets) < m:
        x = np.random.choice(seq)
        targets.add(x)
    return targets

def barabasi_albert_graph(num_generations, m, seed, initial_graph, full_generation_path, regressor):

    G = initial_graph.copy()
    graphs = []

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    # Start adding the other n - m0 nodes.
    source = len(G)

    n_operations_list = regressor.sample(num_generations)

    for i in range(num_generations):
        t = time()
        for j in range(n_operations_list[i]):
            # Now choose m unique nodes from the existing nodes
            # Pick uniformly from repeated_nodes (preferential attachment)
            targets = _random_subset(repeated_nodes, m, seed)
            # Add edges to m nodes from the source.
            G.add_edges_from(zip([source] * m, targets))
            # Add one node to the list for each new edge just created.
            repeated_nodes.extend(targets)
            # And the new node "source" has m edges to add to the list.
            repeated_nodes.extend([source] * m)
            source += 1

        pyg = torch_geometric.utils.convert.from_networkx(G)

        pyg.x = torch.ones((G.number_of_nodes(), 1))
        edge_index_t = torch.cat((pyg.edge_index[1].reshape(1, pyg.edge_index[1].shape[0]),
                                  pyg.edge_index[0].reshape(1, pyg.edge_index[0].shape[0])),
                                 dim=0)
        pyg.edge_index_t = edge_index_t
        pyg.n_operations = n_operations_list[i]

        torch.save(pyg, "{}/BA/graph_{}.pt".format(full_generation_path, i))

        i += 1

    return graphs

def barabasi_albert_generator(history, params):

    num_generations = params['num_generations']
    seed = params['seed']
    regressor = params['regressor']
    full_generation_path = params['full_generation_path']

    torch.manual_seed(seed)
    np.random.seed(seed)

    m = 1
    graphs = barabasi_albert_graph(num_generations, m, seed, history, full_generation_path, regressor)

    return graphs

