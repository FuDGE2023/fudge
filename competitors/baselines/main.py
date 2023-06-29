import json
import os
import sys
import torch
from utils.utils import *
from utils.regressor import DiscreteRegressor
from utils.create_data import *
from model import ba, er, power

def main(fname):
    with open(fname) as fp:
        params = json.load(fp)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = params['n_gpu']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device

    seed = 12345

    params['seed'] = seed

    dataset = params['dataset']
    space_wl_results = params['space_wl_results']
    folder_generation_graphs = params['folder_generation_graphs']
    need_regressor = params['need_regressor']
    baseline = params['baseline']
    seed = params['seed']

    perc_train = params['perc_train']
    params['full_generation_path'] = "{}/{}/generated_{}_graphs".format(folder_generation_graphs, perc_train,
                                                                         dataset)

    for dir in [space_wl_results, params['full_generation_path']]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    if params['create_data']:
        if params['dataset'] == "bitcoin_alpha_edge_weights" or params['dataset'] == "bitcoin_otc_edge_weights":
            create_data_bitcoin_edge_weights(params['file_name'], params['data_path'])
        elif params['dataset'] == "uci_forum_no_dupl":
            create_data_uci_forum(params['file_name'], params['data_path'])
        elif params['dataset'] == "eu_core_no_dupl":
            create_data_eu_core_no_dupl(params['file_name'], params['data_path'])
        # elif params['dataset'] == "synthetic_graph_power_law_with_remove_no_dupl":
        #     create_data_eu_core_no_dupl(file_name, path=None)

    print(params['data_path'])
    snap = loadList(params['data_path'])

    amount = len(snap)

    perc_val_test = (1 - perc_train) / 2.0

    train_size = int(amount * perc_train)
    val_size = int(amount * perc_val_test)

    train_set = snap[0:train_size]
    val_set = snap[train_size:train_size + val_size]
    test_set = snap[train_size + val_size:]

    regressor = DiscreteRegressor(train_set, need_regressor=need_regressor, seed=seed)
    params['regressor'] = regressor

    if type(snap[0]) == list:
        snap = np.array(snap)[:, 4]
        train_set = snap[0:train_size]
        val_set = snap[train_size:train_size + val_size]
        test_set = snap[train_size + val_size:]

    params['num_generations'] = len(test_set)

    if params['generate_test']:

        if not os.path.exists("{}/{}/".format(params['full_generation_path'], params['baseline'])):
            os.makedirs("{}/{}/".format(params['full_generation_path'], params['baseline']))

        t = time()

        train_graph = torch_geometric.utils.convert.to_networkx(train_set[-1])
        history = torch_geometric.utils.convert.to_networkx(val_set[-1])

        if params['baseline'] == "POWER":
            power.power_generator(train_graph, history, params)

        elif params['baseline'] == "BA":
            ba.barabasi_albert_generator(history, params)

        elif params['baseline'] == "ER":
            er.erdos_renyi_generator(train_graph, history, params)

        print("End generation", time() - t)

    if params['compute_wl']:
        compute_wl(params, test_set)

    if params['compute_statistics']:
        G_real = test_set[-1]

        G_pred = torch.load("{}/{}/graph_{}.pt".format(params['full_generation_path'], params['baseline'], len(test_set)-1))

        print("G real")
        graph_statistics(G_real)
        print("G pred")
        graph_statistics(G_pred)


if __name__ == '__main__':
    main(sys.argv[1])

