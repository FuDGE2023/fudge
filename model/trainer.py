import torch
import torch.nn as nn
from time import time
from tqdm import tqdm

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.loader import DataLoader

from model.model import EncoderGIN, RecurrentNetwork, netEmbedding
from utils.utils import create_graph_star, create_graph_star_two_nodes

from collections import defaultdict


def sampling(source_dist, dest_dist, edge_dict, all_nodes: set):
    for source in np.random.choice(list(all_nodes), size=len(all_nodes), replace=False, p=source_dist):
        candidates = np.array(sorted(all_nodes - edge_dict[source]))

        if len(candidates) > 0:
            cand_dist = dest_dist[candidates]
            cand_dist /= cand_dist.sum()
            destination = np.random.choice(candidates, size=1, replace=False, p=cand_dist)[0]
            edge_dict[source].add(destination)

            return source, destination

    return None


class Trainer(nn.Module):
    def __init__(self, params):
        super(Trainer, self).__init__()

        self.params = params
        self.device = self.params['device']
        self.in_feat = self.params['in_feat']
        self.h_dim_encoder = self.params['h_dim_encoder']
        self.z_dim_encoder = self.params['z_dim_encoder']
        self.num_layer_encoder = self.params['num_layer_encoder']
        self.out_dim_encoder = self.params['out_dim_encoder']
        self.learning_rate = self.params['learning_rate']
        self.window_size = self.params['window_size']

        self.h_dim_rnn = self.params['h_dim_rnn']
        self.num_layer_rnn = self.params['num_layer_rnn']

        self.h_dim_embedding = self.params['h_dim_embedding']
        self.h_dim_event = self.params['h_dim_event']

        # Init model add source
        self.gnn_history_source = EncoderGIN(self.in_feat, self.h_dim_encoder, self.z_dim_encoder).to(self.device)
        self.rnn_history_source = RecurrentNetwork(self.z_dim_encoder, self.h_dim_rnn, self.num_layer_rnn).to(
            self.device)

        self.gnn_aux_add_source = EncoderGIN(self.in_feat, self.h_dim_encoder, self.z_dim_encoder).to(self.device)
        self.net_source = netEmbedding(self.num_layer_rnn * self.h_dim_rnn, self.h_dim_embedding,
                                       self.z_dim_encoder).to(self.device)

        self.optim_gnn_history_source = torch.optim.Adam(self.gnn_history_source.parameters(), lr=self.learning_rate)
        self.optim_rnn_history_source = torch.optim.Adam(self.rnn_history_source.parameters(), lr=self.learning_rate)

        self.optim_gnn_aux_add_source = torch.optim.Adam(self.gnn_aux_add_source.parameters(), lr=self.learning_rate)
        self.optim_net_source = torch.optim.Adam(self.net_source.parameters(), lr=self.learning_rate)

        # Init model add dest
        self.gnn_history_dest = EncoderGIN(self.in_feat, self.h_dim_encoder, self.z_dim_encoder).to(self.device)
        self.rnn_history_dest = RecurrentNetwork(self.z_dim_encoder, self.h_dim_rnn, self.num_layer_rnn).to(self.device)

        self.gnn_aux_add_dest = EncoderGIN(self.in_feat, self.h_dim_encoder, self.z_dim_encoder).to(self.device)
        self.net_dest = netEmbedding(self.num_layer_rnn * self.h_dim_rnn + self.z_dim_encoder, self.h_dim_embedding,
                                     self.z_dim_encoder).to(self.device)

        self.optim_gnn_history_dest = torch.optim.Adam(self.gnn_history_dest.parameters(), lr=self.learning_rate)
        self.optim_rnn_history_dest = torch.optim.Adam(self.rnn_history_dest.parameters(), lr=self.learning_rate)
        self.optim_gnn_aux_add_dest = torch.optim.Adam(self.gnn_aux_add_dest.parameters(), lr=self.learning_rate)
        self.optim_net_dest = torch.optim.Adam(self.net_dest.parameters(), lr=self.learning_rate)

        self.cel = nn.CrossEntropyLoss(reduction='mean')
        self.cosine_similarity = nn.CosineSimilarity()

    def train_add_source(self, snapshot_train):
        start_train_add_source = time()
        self.gnn_history_source.train()
        self.rnn_history_source.train()
        self.gnn_aux_add_source.train()
        self.net_source.train()

        loss_source_batch = 0
        idx = 0

        for idx, snapshot_seq in tqdm(enumerate(snapshot_train), total=len(snapshot_train)):
            self.optim_gnn_history_source.zero_grad()
            self.optim_rnn_history_source.zero_grad()

            self.optim_gnn_aux_add_source.zero_grad()
            self.optim_net_source.zero_grad()

            loader = snapshot_seq[0]
            next_graph = snapshot_seq[1]
            last_graph_star = snapshot_seq[3]

            f_tot = None
            for _time, snapshot in enumerate(loader):
                edge_index = snapshot.edge_index.to(self.device)
                x = snapshot.x.to(self.device)
                edge_index_t = snapshot.edge_index_t.to(self.device)
                batch = snapshot.batch.to(self.device)

                f_tot, _ = self.gnn_history_source(edge_index, x, edge_index_t, batch)

            _, h_t = self.rnn_history_source(f_tot)
            h_t = h_t.flatten()
            h_t = h_t.reshape(1, h_t.shape[0])

            edge_index = last_graph_star.edge_index.to(self.device)
            x = last_graph_star.x.to(self.device)
            edge_index_t = last_graph_star.edge_index_t.to(self.device)

            h_last, z_last = self.gnn_aux_add_source(edge_index, x, edge_index_t)
            z_source = self.net_source(h_t)
            logit_source = torch.mm(z_last, z_source.T).flatten()

            y_source = torch.zeros_like(logit_source)
            y_source[next_graph.source] = 1
            y_source = y_source.to(self.device)

            loss_source = self.cel(logit_source, y_source)
            loss_source.backward()

            self.optim_gnn_history_source.step()
            self.optim_rnn_history_source.step()

            self.optim_gnn_aux_add_source.step()
            self.optim_net_source.step()

            loss_source_batch += loss_source.item()

        idx = (idx + 1)
        loss_source_batch /= idx

        end_train_add_source = time()

        print(
            f'loss source: {loss_source_batch:.4f}')
        print(f'Elapsed time: {end_train_add_source - start_train_add_source}')

        return loss_source_batch

    def val_add_source(self, snapshot_val):
        start_train_add_source = time()
        self.gnn_history_source.eval()
        self.rnn_history_source.eval()
        self.gnn_aux_add_source.eval()
        self.net_source.eval()

        loss_source_batch = 0
        idx = 0

        with torch.no_grad():
            for idx, snapshot_seq in tqdm(enumerate(snapshot_val), total=len(snapshot_val)):

                loader = snapshot_seq[0]
                next_graph = snapshot_seq[1]
                last_graph_star = snapshot_seq[3]

                f_tot = None
                for _time, snapshot in enumerate(loader):
                    edge_index = snapshot.edge_index.to(self.device)
                    x = snapshot.x.to(self.device)
                    edge_index_t = snapshot.edge_index_t.to(self.device)
                    batch = snapshot.batch.to(self.device)

                    f_tot, _ = self.gnn_history_source(edge_index, x, edge_index_t, batch)

                _, h_t = self.rnn_history_source(f_tot)
                h_t = h_t.flatten()
                h_t = h_t.reshape(1, h_t.shape[0])

                edge_index = last_graph_star.edge_index.to(self.device)
                x = last_graph_star.x.to(self.device)
                edge_index_t = last_graph_star.edge_index_t.to(self.device)

                h_last, z_last = self.gnn_aux_add_source(edge_index, x, edge_index_t)
                z_source = self.net_source(h_t)
                logit_source = torch.mm(z_last, z_source.T).flatten()

                y_source = torch.zeros_like(logit_source)
                y_source[next_graph.source] = 1
                y_source = y_source.to(self.device)

                loss_source = self.cel(logit_source, y_source)
                loss_source_batch += loss_source.item()

        idx = (idx + 1)
        loss_source_batch /= idx

        end_train_add_source = time()

        print(
            f'loss source: {loss_source_batch:.4f}')
        print(f'Elapsed time: {end_train_add_source - start_train_add_source}')

        return loss_source_batch

    def train_add_dest(self, snapshot_train):
        start_train_add_dest = time()

        self.gnn_history_dest.train()
        self.rnn_history_dest.train()

        self.gnn_aux_add_dest.train()
        self.net_dest.train()

        loss_dest_batch = 0
        idx = 0

        for idx, snapshot_seq in tqdm(enumerate(snapshot_train), total=len(snapshot_train)):
            self.optim_gnn_history_dest.zero_grad()
            self.optim_rnn_history_dest.zero_grad()

            self.optim_gnn_aux_add_dest.zero_grad()
            self.optim_net_dest.zero_grad()

            loader = snapshot_seq[0]
            next_graph = snapshot_seq[1]
            last_graph_star = snapshot_seq[4]

            f_tot = None
            for _time, snapshot in enumerate(loader):
                edge_index = snapshot.edge_index.to(self.device)
                x = snapshot.x.to(self.device)
                edge_index_t = snapshot.edge_index_t.to(self.device)
                batch = snapshot.batch.to(self.device)

                f_tot, _ = self.gnn_history_dest(edge_index, x, edge_index_t, batch)

            _, h_t = self.rnn_history_dest(f_tot)
            h_t = h_t.flatten()
            h_t = h_t.reshape(1, h_t.shape[0])

            edge_index = last_graph_star.edge_index.to(self.device)
            x = last_graph_star.x.to(self.device)
            edge_index_t = last_graph_star.edge_index_t.to(self.device)

            h_last, z_last = self.gnn_aux_add_dest(edge_index, x, edge_index_t)

            # Find source
            source = next_graph.source
            z_source = z_last[source].to(self.device)
            z_source = z_source.reshape(1, z_source.shape[0])

            z_dest = self.net_dest(torch.cat((h_t, z_source), -1))
            logit_dest = torch.mm(z_last, z_dest.T).flatten()

            y_dest = torch.zeros_like(logit_dest)
            y_dest[next_graph.dst] = 1
            y_dest = y_dest.to(self.device)

            loss_dest = self.cel(logit_dest, y_dest)
            loss_dest.backward()

            self.optim_gnn_history_dest.step()
            self.optim_rnn_history_dest.step()

            self.optim_gnn_aux_add_dest.step()
            self.optim_net_dest.step()

            loss_dest_batch += loss_dest.item()

        idx = (idx + 1)
        loss_dest_batch /= idx

        end_train_add_dest = time()

        print(
            f'loss dest: {loss_dest_batch:.4f}')
        print(f'Elapsed time: {end_train_add_dest - start_train_add_dest}')

        return loss_dest_batch

    def val_add_dest(self, snapshot_val):
        start_train_add_dest = time()

        self.gnn_history_dest.eval()
        self.rnn_history_dest.eval()

        self.gnn_aux_add_dest.eval()
        self.net_dest.eval()

        loss_dest_batch = 0
        idx = 0

        with torch.no_grad():
            for idx, snapshot_seq in tqdm(enumerate(snapshot_val), total=len(snapshot_val)):

                loader = snapshot_seq[0]
                next_graph = snapshot_seq[1]
                last_graph_star = snapshot_seq[4]

                f_tot = None
                for _time, snapshot in enumerate(loader):
                    edge_index = snapshot.edge_index.to(self.device)
                    x = snapshot.x.to(self.device)
                    edge_index_t = snapshot.edge_index_t.to(self.device)
                    batch = snapshot.batch.to(self.device)

                    f_tot, _ = self.gnn_history_dest(edge_index, x, edge_index_t, batch)

                _, h_t = self.rnn_history_dest(f_tot)
                h_t = h_t.flatten()
                h_t = h_t.reshape(1, h_t.shape[0])

                edge_index = last_graph_star.edge_index.to(self.device)
                x = last_graph_star.x.to(self.device)
                edge_index_t = last_graph_star.edge_index_t.to(self.device)

                h_last, z_last = self.gnn_aux_add_dest(edge_index, x, edge_index_t)

                # Find source
                source = next_graph.source
                z_source = z_last[source].to(self.device)
                z_source = z_source.reshape(1, z_source.shape[0])

                z_dest = self.net_dest(torch.cat((h_t, z_source), -1))
                logit_dest = torch.mm(z_last, z_dest.T).flatten()

                y_dest = torch.zeros_like(logit_dest)
                y_dest[next_graph.dst] = 1
                y_dest = y_dest.to(self.device)

                loss_dest = self.cel(logit_dest, y_dest)
                loss_dest_batch += loss_dest.item()

        idx = (idx + 1)
        loss_dest_batch /= idx

        end_train_add_dest = time()

        print(
            f'loss dest: {loss_dest_batch:.4f}')
        print(f'Elapsed time: {end_train_add_dest - start_train_add_dest}')

        return loss_dest_batch

    def train_step(self, snapshot_train, snapshot_val):
        dataset = self.params['dataset']
        exp = self.params['exp']
        space_results_losses = self.params['space_results_losses']

        best_loss_source = np.inf
        best_loss_dest = np.inf

        path_losses = f'{space_results_losses}/losses_{dataset}_{exp}.txt'
        path_val_losses = f'{space_results_losses}/val_losses_{dataset}_{exp}.txt'

        open(path_losses, 'w').close()
        open(path_val_losses, 'w').close()

        losses_source, losses_dest = [], []
        losses_source_val, losses_dest_val = [], []

        try:
            for epoch in range(self.params['n_epochs']):
                print(f'Epoch: {epoch + 1}')
                print('\n Training \n')
                loss_source_batch = self.train_add_source(snapshot_train)
                loss_dest_batch = self.train_add_dest(snapshot_train)

                losses_source.append(loss_source_batch)
                losses_dest.append(loss_dest_batch)

                print('\n Validation \n')

                val_loss_source_batch = self.val_add_source(snapshot_val)
                val_loss_dest_batch = self.val_add_dest(snapshot_val)

                losses_source_val.append(val_loss_source_batch)
                losses_dest_val.append(val_loss_dest_batch)

                with open(path_losses, 'a') as filehandle:
                    filehandle.write(f'\n {epoch}, {loss_source_batch}, {loss_dest_batch} \n')

                with open(path_val_losses, 'a') as filehandle:
                    filehandle.write(f'\n {epoch}, {val_loss_source_batch}, {val_loss_dest_batch} \n')

                if best_loss_source > val_loss_source_batch:
                    best_loss_source = val_loss_source_batch
                    torch.save(self.gnn_history_source.state_dict(), self.params['gnn_history_source'])
                    torch.save(self.rnn_history_source.state_dict(), self.params['rnn_history_source'])

                    torch.save(self.gnn_aux_add_source.state_dict(), self.params['gnn_aux_add_source'])
                    torch.save(self.net_source.state_dict(), self.params['net_source'])

                if best_loss_dest > val_loss_dest_batch:
                    best_loss_dest = val_loss_dest_batch

                    torch.save(self.gnn_history_dest.state_dict(), self.params['gnn_history_dest'])
                    torch.save(self.rnn_history_dest.state_dict(), self.params['rnn_history_dest'])

                    torch.save(self.gnn_aux_add_dest.state_dict(), self.params['gnn_aux_add_dest'])
                    torch.save(self.net_dest.state_dict(), self.params['net_dest'])

        except KeyboardInterrupt:
            print('---' * 30)

        plt.figure(figsize=(15, 8))
        plt.plot(losses_source, label='source')
        plt.plot(losses_dest, label='destination')
        plt.legend()
        plt.savefig(self.params['plot_title'], bbox_inches='tight')
        plt.show()

    def fit(self, snapshot_train, snapshot_val):
        self.train_step(snapshot_train, snapshot_val)

    def generate_graph_rnd_alternative_regressor(self, history, num_generation, models_add_source,
                                                 models_add_dest, regressor, count_check):
        dataset = self.params['dataset']
        path_generated_graphs = self.params['path_generated_graphs']

        gnn_history_source, rnn_history_source, gnn_aux_add_source, net_source = models_add_source
        gnn_history_dest, rnn_history_dest, gnn_aux_add_dest, net_dest = models_add_dest

        device = self.device

        generated_snap_checkpoint = []

        gnn_history_source.eval()
        rnn_history_source.eval()
        gnn_aux_add_source.eval()
        net_source.eval()

        gnn_history_dest.eval()
        rnn_history_dest.eval()
        gnn_aux_add_dest.eval()
        net_dest.eval()

        count = 0
        count_checkpoint = count_check

        n_operation_list = regressor.sample(num_generation)

        with torch.no_grad():

            pbar = tqdm(total=num_generation)
            while count < num_generation:

                last_graph = history[-1]
                last_graph_star = create_graph_star(last_graph.clone())
                last_graph_star_two = create_graph_star_two_nodes(last_graph.clone())

                last_graph_tmp = history[-1]

                loader = DataLoader(history, batch_size=self.params['window_size'])

                current_pyg = []
                current_G = []
                current_source_dst = []

                for no_idx in range(n_operation_list[count]):

                    f_tot = None
                    for _time, snapshot in enumerate(loader):
                        edge_index = snapshot.edge_index.to(device)
                        x = snapshot.x.to(device)
                        edge_index_t = snapshot.edge_index_t.to(device)
                        batch = snapshot.batch.to(device)

                        f_tot, _ = gnn_history_source(edge_index, x, edge_index_t, batch)

                    _, h_t = rnn_history_source(f_tot)
                    h_t = h_t.flatten()
                    h_t = h_t.reshape(1, h_t.shape[0])

                    edge_index = last_graph_star.edge_index.to(device)
                    x = last_graph_star.x.to(device)
                    edge_index_t = last_graph_star.edge_index_t.to(device)

                    h_last, z_last = gnn_aux_add_source(edge_index, x, edge_index_t)
                    z_source = net_source(h_t)
                    logit_source = torch.mm(z_last, z_source.T).flatten()

                    p_source = F.softmax(logit_source, -1)

                    f_tot = None
                    for _time, snapshot in enumerate(loader):
                        edge_index = snapshot.edge_index.to(device)
                        x = snapshot.x.to(device)
                        edge_index_t = snapshot.edge_index_t.to(device)
                        batch = snapshot.batch.to(device)

                        f_tot, _ = gnn_history_dest(edge_index, x, edge_index_t, batch)

                    _, h_t = rnn_history_dest(f_tot)
                    h_t = h_t.flatten()
                    h_t = h_t.reshape(1, h_t.shape[0])

                    edge_index = last_graph_star_two.edge_index.to(device)
                    x = last_graph_star_two.x.to(device)
                    edge_index_t = last_graph_star_two.edge_index_t.to(device)

                    h_last, z_last = gnn_aux_add_dest(edge_index, x, edge_index_t)

                    # Find source
                    z_dest = net_dest(torch.cat((h_t, z_source), -1))
                    logit_dest = torch.mm(z_last, z_dest.T).flatten()

                    p_dest = F.softmax(logit_dest, -1)

                    nodes = set(range(len(p_source)))

                    pyg = last_graph_tmp.clone()
                    G = to_networkx(pyg, to_undirected=False)
                    feat = pyg.x
                    num_nodes = pyg.num_nodes

                    edge_dict = defaultdict(set)

                    for n in G.nodes():
                        edge_dict[n] = set([dest for _, dest in G.edges(n)])

                    p_source = p_source.cpu().numpy()
                    p_dest = p_dest.cpu().numpy()

                    first_min, second_min, *_ = np.partition(np.unique(p_source), 1)

                    if first_min == 0:
                        idxs = np.where(p_source == first_min)

                        if second_min < 1e-45:
                            p_source[list(idxs[0])] = second_min / 10
                        else:
                            p_source[list(idxs[0])] = second_min

                        p_source /= p_source.sum()

                    first_min, second_min, *_ = np.partition(np.unique(p_dest), 1)

                    if first_min == 0:
                        idxs = np.where(p_dest == first_min)
                        if second_min < 1e-45:
                            p_dest[list(idxs[0])] = second_min / 10
                        else:
                            p_dest[list(idxs[0])] = second_min

                        p_dest /= p_dest.sum()

                    source, dest = sampling(p_source, p_dest, edge_dict, nodes)

                    if source not in G.nodes():
                        x = torch.ones(1, 1)
                        feat = torch.cat((feat, x))

                        G.add_node(source)

                    if dest not in G.nodes():
                        x = torch.ones(1, 1)
                        feat = torch.cat((feat, x))

                        G.add_node(dest)

                    G.add_edge(source, dest)
                    pyg = from_networkx(G)
                    pyg.x = feat

                    edge_index = pyg.edge_index
                    edge_index_t = torch.cat((edge_index[1].reshape(1, edge_index[1].shape[0]),
                                              edge_index[0].reshape(1, edge_index[0].shape[0])),
                                             dim=0)

                    pyg.edge_index_t = edge_index_t
                    pyg.source = source
                    pyg.dst = dest
                    pyg.n_operation = n_operation_list[count]

                    current_pyg.append(pyg)
                    current_G.append(G.copy())
                    current_source_dst.append((source, dest))

                    last_graph_tmp = pyg.clone()

                generated_snap_checkpoint.append([current_G[-1],
                                                  current_source_dst[-1],
                                                  current_pyg[-1]])

                pyg = current_pyg[-1]

                history.pop(0)
                history.append(pyg)

                count += 1
                pbar.update(1)

                if count % 500 == 0:
                    torch.save(generated_snap_checkpoint, path_generated_graphs + '_' + str(count_checkpoint) + '_regressor.pt')
                    generated_snap_checkpoint = []

                    count_checkpoint += 1

                    print('Checkpoint saved ... (generated data)')

            pbar.close()

        torch.save(generated_snap_checkpoint, path_generated_graphs + '_last_regressor.pt')

    def generate_graph_rnd_alternative_regressor_sampling(self, history, num_generation, models_add_source,
                                                 models_add_dest, regressor, count_check):
        dataset = self.params['dataset']
        path_generated_graphs = self.params['path_generated_graphs']

        gnn_history_source, rnn_history_source, gnn_aux_add_source, net_source = models_add_source
        gnn_history_dest, rnn_history_dest, gnn_aux_add_dest, net_dest = models_add_dest

        device = self.device

        generated_snapshot_rnd = []

        generated_snap_checkpoint = []

        gnn_history_source.eval()
        rnn_history_source.eval()
        gnn_aux_add_source.eval()
        net_source.eval()

        gnn_history_dest.eval()
        rnn_history_dest.eval()
        gnn_aux_add_dest.eval()
        net_dest.eval()

        count = 0
        count_checkpoint = count_check

        n_operation_list = regressor.sample(num_generation)

        with torch.no_grad():

            pbar = tqdm(total=num_generation)
            while count < num_generation:

                last_graph = history[-1]
                last_graph_star = create_graph_star(last_graph.clone())
                last_graph_star_two = create_graph_star_two_nodes(last_graph.clone())

                loader = DataLoader(history, batch_size=self.params['window_size'])

                f_tot = None
                for _time, snapshot in enumerate(loader):
                    edge_index = snapshot.edge_index.to(device)
                    x = snapshot.x.to(device)
                    edge_index_t = snapshot.edge_index_t.to(device)
                    batch = snapshot.batch.to(device)

                    f_tot, _ = gnn_history_source(edge_index, x, edge_index_t, batch)

                _, h_t = rnn_history_source(f_tot)
                h_t = h_t.flatten()
                h_t = h_t.reshape(1, h_t.shape[0])

                edge_index = last_graph_star.edge_index.to(device)
                x = last_graph_star.x.to(device)
                edge_index_t = last_graph_star.edge_index_t.to(device)

                h_last, z_last = gnn_aux_add_source(edge_index, x, edge_index_t)
                z_source = net_source(h_t)
                logit_source = torch.mm(z_last, z_source.T).flatten()

                p_source = F.softmax(logit_source, -1)

                f_tot = None
                for _time, snapshot in enumerate(loader):
                    edge_index = snapshot.edge_index.to(device)
                    x = snapshot.x.to(device)
                    edge_index_t = snapshot.edge_index_t.to(device)
                    batch = snapshot.batch.to(device)

                    f_tot, _ = gnn_history_dest(edge_index, x, edge_index_t, batch)

                _, h_t = rnn_history_dest(f_tot)
                h_t = h_t.flatten()
                h_t = h_t.reshape(1, h_t.shape[0])

                edge_index = last_graph_star_two.edge_index.to(device)
                x = last_graph_star_two.x.to(device)
                edge_index_t = last_graph_star_two.edge_index_t.to(device)

                h_last, z_last = gnn_aux_add_dest(edge_index, x, edge_index_t)

                # Find source
                z_dest = net_dest(torch.cat((h_t, z_source), -1))
                logit_dest = torch.mm(z_last, z_dest.T).flatten()

                p_dest = F.softmax(logit_dest, -1)

                nodes = set(range(len(p_source)))

                pyg = last_graph.clone()
                G = to_networkx(pyg, to_undirected=False)
                feat = pyg.x
                num_nodes = pyg.num_nodes

                edge_dict = defaultdict(set)

                for n in G.nodes():
                    edge_dict[n] = set([dest for _, dest in G.edges(n)])

                p_source = p_source.cpu().numpy()
                p_dest = p_dest.cpu().numpy()

                first_min, second_min, *_ = np.partition(np.unique(p_source), 1)

                if first_min == 0:
                    idxs = np.where(p_source == first_min)

                    if second_min < 1e-45:
                        p_source[list(idxs[0])] = second_min / 10
                    else:
                        p_source[list(idxs[0])] = second_min

                    p_source /= p_source.sum()

                first_min, second_min, *_ = np.partition(np.unique(p_dest), 1)

                if first_min == 0:
                    idxs = np.where(p_dest == first_min)
                    if second_min < 1e-45:
                        p_dest[list(idxs[0])] = second_min / 10
                    else:
                        p_dest[list(idxs[0])] = second_min

                    p_dest /= p_dest.sum()

                for no_idx in range(n_operation_list[count]):

                    source, dest = sampling(p_source, p_dest, edge_dict, nodes)

                    if source not in G.nodes():
                        x = torch.ones(1, 1)
                        feat = torch.cat((feat, x))

                        G.add_node(source)

                    if dest not in G.nodes():
                        x = torch.ones(1, 1)
                        feat = torch.cat((feat, x))

                        G.add_node(dest)

                    G.add_edge(source, dest)

                    edge_dict = defaultdict(set)

                    for n in G.nodes():
                        edge_dict[n] = set([dest for _, dest in G.edges(n)])

                pyg = from_networkx(G)
                pyg.x = feat

                edge_index = pyg.edge_index
                edge_index_t = torch.cat((edge_index[1].reshape(1, edge_index[1].shape[0]),
                                          edge_index[0].reshape(1, edge_index[0].shape[0])),
                                         dim=0)

                pyg.edge_index_t = edge_index_t
                pyg.source = source
                pyg.dst = dest
                pyg.n_operation = n_operation_list[count]

                generated_snap_checkpoint.append([G.copy(),
                                                  [(source, dest)],
                                                  pyg])

                history.pop(0)
                history.append(pyg)

                count += 1
                pbar.update(1)

                if count % 500 == 0:
                    torch.save(generated_snap_checkpoint, path_generated_graphs + '_' + str(count_checkpoint) + '_regressor.pt')
                    generated_snap_checkpoint = []

                    count_checkpoint += 1

                    print('Checkpoint saved ... (generated data)')

            pbar.close()

        torch.save(generated_snap_checkpoint, path_generated_graphs + '_last_regressor.pt')
