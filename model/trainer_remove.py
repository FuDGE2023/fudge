import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_networkx, from_networkx

from model.model import EncoderGIN, RecurrentNetwork, netEmbedding, ProbabilityEventNetwork
from utils.utils import create_graph_star, create_dynamic_loader, create_graph_star_two_nodes

from time import time
import matplotlib.pyplot as plt
import numpy as np

from torch_geometric.loader import DataLoader

from tqdm import tqdm

from collections import defaultdict


def sampling(source_dist, dest_dist, edge_dict, all_nodes: set, add=True):
    for source in np.random.choice(list(all_nodes), size=len(all_nodes), replace=False, p=source_dist):

        if add:
            candidates = np.array(sorted(all_nodes - edge_dict[source]))
        else:
            candidates = np.array(sorted(edge_dict[source]))

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

        self.learning_rate_event = self.params['learning_rate_event']
        self.learning_rate_source = self.params['learning_rate_source']
        self.learning_rate_dest = self.params['learning_rate_dest']

        self.window_size = self.params['window_size']
        self.batch_size = self.params['batch_size']

        self.h_dim_rnn = self.params['h_dim_rnn']
        self.num_layer_rnn = self.params['num_layer_rnn']

        self.h_dim_embedding = self.params['h_dim_embedding']
        self.h_dim_event = self.params['h_dim_event']
        self.out_dim_event = self.params['out_dim_event']

        # Init model event
        self.gnn_history_event = EncoderGIN(self.in_feat, self.h_dim_encoder, self.z_dim_encoder).to(self.device)
        self.rnn_history_event = RecurrentNetwork(self.z_dim_encoder, self.h_dim_rnn, self.num_layer_rnn).to(
            self.device)
        self.event = ProbabilityEventNetwork(self.num_layer_rnn * self.h_dim_rnn, self.h_dim_event,
                                             self.out_dim_event).to(self.device)

        self.optim_gnn_history_event = torch.optim.Adam(self.gnn_history_event.parameters(),
                                                        lr=self.learning_rate_event)
        self.optim_rnn_history_event = torch.optim.Adam(self.rnn_history_event.parameters(),
                                                        lr=self.learning_rate_event)
        self.optim_event = torch.optim.Adam(self.event.parameters(), lr=self.learning_rate_event)

        # Init model add source
        self.gnn_history_source = EncoderGIN(self.in_feat, self.h_dim_encoder, self.z_dim_encoder).to(self.device)
        self.rnn_history_source = RecurrentNetwork(self.z_dim_encoder, self.h_dim_rnn, self.num_layer_rnn).to(
            self.device)

        self.gnn_aux_add_source = EncoderGIN(self.in_feat, self.h_dim_encoder, self.z_dim_encoder).to(self.device)
        self.net_source = netEmbedding(self.num_layer_rnn * self.h_dim_rnn, self.h_dim_embedding,
                                       self.z_dim_encoder).to(self.device)

        self.optim_gnn_history_source = torch.optim.Adam(self.gnn_history_source.parameters(),
                                                         lr=self.learning_rate_source)
        self.optim_rnn_history_source = torch.optim.Adam(self.rnn_history_source.parameters(),
                                                         lr=self.learning_rate_source)

        self.optim_gnn_aux_add_source = torch.optim.Adam(self.gnn_aux_add_source.parameters(),
                                                         lr=self.learning_rate_source)
        self.optim_net_source = torch.optim.Adam(self.net_source.parameters(), lr=self.learning_rate_source)

        # Init model add dest
        self.gnn_history_dest = EncoderGIN(self.in_feat, self.h_dim_encoder, self.z_dim_encoder).to(self.device)
        self.rnn_history_dest = RecurrentNetwork(self.z_dim_encoder, self.h_dim_rnn, self.num_layer_rnn).to(self.device)

        self.gnn_aux_add_dest = EncoderGIN(self.in_feat, self.h_dim_encoder, self.z_dim_encoder).to(self.device)
        self.net_dest = netEmbedding(self.num_layer_rnn * self.h_dim_rnn + self.z_dim_encoder, self.h_dim_embedding,
                                     self.z_dim_encoder).to(self.device)

        self.optim_gnn_history_dest = torch.optim.Adam(self.gnn_history_dest.parameters(), lr=self.learning_rate_dest)
        self.optim_rnn_history_dest = torch.optim.Adam(self.rnn_history_dest.parameters(), lr=self.learning_rate_dest)
        self.optim_gnn_aux_add_dest = torch.optim.Adam(self.gnn_aux_add_dest.parameters(), lr=self.learning_rate_dest)
        self.optim_net_dest = torch.optim.Adam(self.net_dest.parameters(), lr=self.learning_rate_dest)

        self.cel = nn.CrossEntropyLoss(reduction='mean')

        self.scaler = torch.cuda.amp.GradScaler()

    def train_event(self, snapshot_train):
        start_train_event = time()
        self.gnn_history_event.train()
        self.rnn_history_event.train()
        self.event.train()

        loss_event_batch = 0
        idx = 0

        for idx, snapshot_seq in tqdm(enumerate(snapshot_train), total=len(snapshot_train)):
            self.optim_gnn_history_event.zero_grad()
            self.optim_rnn_history_event.zero_grad()
            self.optim_event.zero_grad()

            loader = snapshot_seq[0]

            y_event = snapshot_seq[10]
            y_event = y_event.to(self.device)

            it_loader = iter(loader)
            snapshot = next(it_loader)
            edge_index = snapshot.edge_index.to(self.device)
            x = snapshot.x.to(self.device).to(torch.float16)
            edge_index_t = snapshot.edge_index_t.to(self.device)
            batch = snapshot.batch.to(self.device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                f_tot, _ = self.gnn_history_event(edge_index, x, edge_index_t, batch)

                bs_tmp = f_tot.shape[0] // self.window_size
                f_tot = f_tot.reshape(bs_tmp, self.window_size, self.z_dim_encoder)

                _, h_t = self.rnn_history_event(f_tot)

                h_t = h_t.transpose(0, 1)
                h_t = h_t.reshape(h_t.shape[0], h_t.shape[1] * h_t.shape[2])

                p_event = self.event(h_t)

                loss_event = self.cel(p_event, y_event)

            self.scaler.scale(loss_event).backward()
            loss_event_batch += loss_event.item()

            self.scaler.step(self.optim_gnn_history_event)
            self.scaler.step(self.optim_rnn_history_event)
            self.scaler.step(self.optim_event)

            self.scaler.update()

        idx = (idx + 1)
        loss_event_batch /= idx

        end_train_event = time()

        print(
            f'loss event: {loss_event_batch:.4f}')
        print(f'Elapsed time: {end_train_event - start_train_event}')

        return loss_event_batch

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
            loader_last_graphs_star = snapshot_seq[3]
            n_nodes_last_graphs_star = snapshot_seq[5]

            y_sources = snapshot_seq[7]

            it_loader = iter(loader)
            snapshot = next(it_loader)
            edge_index = snapshot.edge_index.to(self.device)
            x = snapshot.x.to(self.device).to(torch.float16)
            edge_index_t = snapshot.edge_index_t.to(self.device)
            batch = snapshot.batch.to(self.device)
            y_sources = y_sources.to(self.device).to(torch.float16)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                f_tot, _ = self.gnn_history_source(edge_index, x, edge_index_t, batch)

                bs_tmp = f_tot.shape[0] // self.window_size
                f_tot = f_tot.reshape(bs_tmp, self.window_size, self.z_dim_encoder)

                _, h_t = self.rnn_history_source(f_tot)

                h_t = h_t.transpose(0, 1)
                h_t = h_t.reshape(h_t.shape[0], h_t.shape[1] * h_t.shape[2])

                z_source = self.net_source(h_t)

                it_loader = iter(loader_last_graphs_star)
                last_graphs_star = next(it_loader)

                edge_index = last_graphs_star.edge_index.to(self.device)
                x = last_graphs_star.x.to(self.device)
                edge_index_t = last_graphs_star.edge_index_t.to(self.device)
                batch = last_graphs_star.batch.to(self.device)

                _, z_last = self.gnn_aux_add_source(edge_index, x, edge_index_t, batch)

                max_n = np.max(n_nodes_last_graphs_star)
                z_last = torch.split(z_last, n_nodes_last_graphs_star)
                z_last = torch.stack([torch.cat((x, torch.zeros(max_n-x.shape[0], x.shape[1]).to(self.device)),
                                                dim=0) for x in z_last])
                z_source = z_source.unsqueeze(1).repeat(1, max_n, 1)
                logit_source = torch.bmm(z_last, torch.transpose(z_source, 2, 1))
                logit_source = logit_source[:, :, 0]

                loss_source = self.cel(logit_source, y_sources)

            self.scaler.scale(loss_source).backward()
            loss_source_batch += loss_source.item()

            self.scaler.step(self.optim_gnn_history_source)
            self.scaler.step(self.optim_rnn_history_source)
            self.scaler.step(self.optim_gnn_aux_add_source)
            self.scaler.step(self.optim_net_source)
            self.scaler.update()

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
            loader_last_graphs_star = snapshot_seq[4]
            n_nodes_last_graphs_star_two = snapshot_seq[6]

            y_destinations = snapshot_seq[8]
            y_destinations = y_destinations.to(self.device).to(torch.float16)

            source_tmp = snapshot_seq[9]

            it_loader = iter(loader)
            snapshot = next(it_loader)

            edge_index = snapshot.edge_index.to(self.device)
            x = snapshot.x.to(self.device).to(torch.float16)
            edge_index_t = snapshot.edge_index_t.to(self.device)
            batch = snapshot.batch.to(self.device)

            with torch.cuda.amp.autocast(dtype=torch.float16):

                f_tot, _ = self.gnn_history_dest(edge_index, x, edge_index_t, batch)

                bs_tmp = f_tot.shape[0] // self.window_size
                f_tot = f_tot.reshape(bs_tmp, self.window_size, self.z_dim_encoder)

                _, h_t = self.rnn_history_dest(f_tot)
                h_t = h_t.transpose(0, 1)
                h_t = h_t.reshape(h_t.shape[0], h_t.shape[1] * h_t.shape[2])

                it_loader = iter(loader_last_graphs_star)
                last_graphs_star = next(it_loader)
                edge_index = last_graphs_star.edge_index.to(self.device)
                x = last_graphs_star.x.to(self.device)
                edge_index_t = last_graphs_star.edge_index_t.to(self.device)
                batch = last_graphs_star.batch.to(self.device)

                _, z_last = self.gnn_aux_add_dest(edge_index, x, edge_index_t, batch)

                z_source = z_last[source_tmp].to(self.device)

                z_dest = self.net_dest(torch.cat((h_t, z_source), -1))

                max_n = np.max(n_nodes_last_graphs_star_two)
                z_last = torch.split(z_last, n_nodes_last_graphs_star_two)
                z_last = torch.stack([torch.cat((x, torch.zeros(max_n - x.shape[0], x.shape[1]).to(self.device)),
                                                dim=0) for x in z_last])
                z_dest = z_dest.unsqueeze(1).repeat(1, max_n, 1)
                logit_dest = torch.bmm(z_last, torch.transpose(z_dest, 2, 1))
                logit_dest = logit_dest[:, :, 0]

                loss_dest = self.cel(logit_dest, y_destinations)

            self.scaler.scale(loss_dest).backward()

            loss_dest_batch += loss_dest.item()

            self.scaler.step(self.optim_gnn_history_dest)
            self.scaler.step(self.optim_rnn_history_dest)
            self.scaler.step(self.optim_gnn_aux_add_dest)
            self.scaler.step(self.optim_net_dest)

            self.scaler.update()

        idx = (idx + 1)
        loss_dest_batch /= idx

        end_train_add_dest = time()

        print(
            f'loss dest: {loss_dest_batch:.4f}')
        print(f'Elapsed time: {end_train_add_dest - start_train_add_dest}')

        return loss_dest_batch

    def val_event(self, snapshot_val):
        start_train_event = time()
        self.gnn_history_event.eval()
        self.rnn_history_event.eval()
        self.event.eval()

        loss_event_batch = 0
        idx = 0

        with torch.no_grad():
            for idx, snapshot_seq in tqdm(enumerate(snapshot_val), total=len(snapshot_val)):

                loader = snapshot_seq[0]

                y_event = snapshot_seq[10]
                y_event = y_event.to(self.device)

                it_loader = iter(loader)
                snapshot = next(it_loader)
                edge_index = snapshot.edge_index.to(self.device)
                x = snapshot.x.to(self.device)
                edge_index_t = snapshot.edge_index_t.to(self.device)
                batch = snapshot.batch.to(self.device)

                f_tot, _ = self.gnn_history_event(edge_index, x, edge_index_t, batch)
                bs_tmp = f_tot.shape[0] // self.window_size
                f_tot = f_tot.reshape(bs_tmp, self.window_size, self.z_dim_encoder)

                _, h_t = self.rnn_history_event(f_tot)
                h_t = h_t.transpose(0, 1)
                h_t = h_t.reshape(h_t.shape[0], h_t.shape[1] * h_t.shape[2])

                p_event = self.event(h_t)

                loss_event = self.cel(p_event, y_event)
                loss_event_batch += loss_event.item()

        idx = (idx + 1)
        loss_event_batch /= idx

        end_train_event = time()

        print(
            f'loss event: {loss_event_batch:.4f}')
        print(f'Elapsed time: {end_train_event - start_train_event}')

        return loss_event_batch

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
                loader_last_graphs_star = snapshot_seq[3]
                n_nodes_last_graphs_star = snapshot_seq[5]

                y_sources = snapshot_seq[7]
                y_sources = y_sources.to(self.device)

                it_loader = iter(loader)
                snapshot = next(it_loader)
                edge_index = snapshot.edge_index.to(self.device)
                x = snapshot.x.to(self.device)
                edge_index_t = snapshot.edge_index_t.to(self.device)
                batch = snapshot.batch.to(self.device)

                f_tot, _ = self.gnn_history_source(edge_index, x, edge_index_t, batch)
                bs_tmp = f_tot.shape[0] // self.window_size
                f_tot = f_tot.reshape(bs_tmp, self.window_size, self.z_dim_encoder)

                _, h_t = self.rnn_history_source(f_tot)
                h_t = h_t.transpose(0, 1)
                h_t = h_t.reshape(h_t.shape[0], h_t.shape[1] * h_t.shape[2])

                z_source = self.net_source(h_t)

                it_loader = iter(loader_last_graphs_star)
                last_graphs_star = next(it_loader)

                edge_index = last_graphs_star.edge_index.to(self.device)
                x = last_graphs_star.x.to(self.device)
                edge_index_t = last_graphs_star.edge_index_t.to(self.device)
                batch = last_graphs_star.batch.to(self.device)

                _, z_last = self.gnn_aux_add_source(edge_index, x, edge_index_t, batch)

                max_n = np.max(n_nodes_last_graphs_star)
                z_last = torch.split(z_last, n_nodes_last_graphs_star)
                z_last = torch.stack([torch.cat((x, torch.zeros(max_n - x.shape[0], x.shape[1]).to(self.device)),
                                                dim=0) for x in z_last])
                z_source = z_source.unsqueeze(1).repeat(1, max_n, 1)
                logit_source = torch.bmm(z_last, torch.transpose(z_source, 2, 1))
                logit_source = logit_source[:, :, 0]

                loss_source = self.cel(logit_source, y_sources)
                loss_source_batch += loss_source.item()

        idx = (idx + 1)
        loss_source_batch /= idx

        end_train_add_source = time()

        print(
            f'loss source: {loss_source_batch:.4f}')
        print(f'Elapsed time: {end_train_add_source - start_train_add_source}')

        return loss_source_batch

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
                loader_last_graphs_star = snapshot_seq[4]
                n_nodes_last_graphs_star_two = snapshot_seq[6]

                y_destinations = snapshot_seq[8]
                y_destinations = y_destinations.to(self.device)

                source_tmp = snapshot_seq[9]

                it_loader = iter(loader)
                snapshot = next(it_loader)

                edge_index = snapshot.edge_index.to(self.device)
                x = snapshot.x.to(self.device)
                edge_index_t = snapshot.edge_index_t.to(self.device)
                batch = snapshot.batch.to(self.device)

                f_tot, _ = self.gnn_history_dest(edge_index, x, edge_index_t, batch)

                bs_tmp = f_tot.shape[0] // self.window_size

                f_tot = f_tot.reshape(bs_tmp, self.window_size, self.z_dim_encoder)

                _, h_t = self.rnn_history_dest(f_tot)
                h_t = h_t.transpose(0, 1)
                h_t = h_t.reshape(h_t.shape[0], h_t.shape[1] * h_t.shape[2])

                it_loader = iter(loader_last_graphs_star)
                last_graphs_star = next(it_loader)
                edge_index = last_graphs_star.edge_index.to(self.device)
                x = last_graphs_star.x.to(self.device)
                edge_index_t = last_graphs_star.edge_index_t.to(self.device)
                batch = last_graphs_star.batch.to(self.device)

                _, z_last = self.gnn_aux_add_dest(edge_index, x, edge_index_t, batch)

                z_source = z_last[source_tmp].to(self.device)

                z_dest = self.net_dest(torch.cat((h_t, z_source), -1))
                max_n = np.max(n_nodes_last_graphs_star_two)
                z_last = torch.split(z_last, n_nodes_last_graphs_star_two)
                z_last = torch.stack([torch.cat((x, torch.zeros(max_n - x.shape[0], x.shape[1]).to(self.device)),
                                                dim=0) for x in z_last])
                z_dest = z_dest.unsqueeze(1).repeat(1, max_n, 1)
                logit_dest = torch.bmm(z_last, torch.transpose(z_dest, 2, 1))
                logit_dest = logit_dest[:, :, 0]

                loss_dest = self.cel(logit_dest, y_destinations)
                loss_dest_batch += loss_dest.item()

        idx = (idx + 1)
        loss_dest_batch /= idx

        end_train_add_dest = time()

        print(
            f'loss dest: {loss_dest_batch:.4f}')
        print(f'Elapsed time: {end_train_add_dest - start_train_add_dest}')

        return loss_dest_batch

    def train_step(self, snapshot_train, snapshot_val, x_seq_train, y_seq_train, is_sampling=False): #
        dataset = self.params['dataset']
        exp = self.params['exp']

        best_loss_source = np.inf
        best_loss_dest = np.inf
        best_loss_event = np.inf

        path_losses = f'./results_losses/losses_{dataset}_{exp}.txt'
        path_val_losses = f'./results_losses/val_losses_{dataset}_{exp}.txt'

        open(path_losses, 'w').close()
        open(path_val_losses, 'w').close()

        losses_event, losses_source, losses_dest = [], [], []
        losses_event_val, losses_source_val, losses_dest_val = [], [], []

        try:

            for epoch in range(self.params['n_epochs']):
                print(f'Epoch: {epoch + 1}')
                print('\n Training \n')

                if is_sampling:
                        snapshot_train = create_dynamic_loader(x_seq_train, y_seq_train, self.params['window_size'],
                                                               self.params['batch_size'], is_synthetic=True)

                loss_event_batch = self.train_event(snapshot_train)
                loss_source_batch = self.train_add_source(snapshot_train)
                loss_dest_batch = self.train_add_dest(snapshot_train)

                losses_event.append(loss_event_batch)
                losses_source.append(loss_source_batch)
                losses_dest.append(loss_dest_batch)

                print('\n Validation \n')
                val_loss_event_batch = self.val_event(snapshot_val)
                val_loss_source_batch = self.val_add_source(snapshot_val)
                val_loss_dest_batch = self.val_add_dest(snapshot_val)

                losses_event_val.append(val_loss_event_batch)
                losses_source_val.append(val_loss_source_batch)
                losses_dest_val.append(val_loss_dest_batch)

                with open(path_losses, 'a') as filehandle:
                    filehandle.write(f'\n {epoch}, {loss_source_batch}, {loss_dest_batch}, {loss_event_batch} \n')

                with open(path_val_losses, 'a') as filehandle:
                    filehandle.write(f'\n {epoch}, {val_loss_source_batch}, {val_loss_dest_batch}, {val_loss_event_batch}  \n')

                if best_loss_event > val_loss_event_batch:
                    best_loss_event = val_loss_event_batch

                    torch.save(self.gnn_history_event.state_dict(), self.params['gnn_history_event'])
                    torch.save(self.rnn_history_event.state_dict(), self.params['rnn_history_event'])
                    torch.save(self.event.state_dict(), self.params['event_network'])

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
        plt.plot(losses_event, label='event')
        plt.plot(losses_source, label='source')
        plt.plot(losses_dest, label='destination')
        plt.legend()
        plt.savefig(self.params['plot_title'], bbox_inches='tight')
        plt.show()

    def fit(self, snapshot_train, snapshot_val, x_sequences_train, y_sequences_train, is_sampling):
        self.train_step(snapshot_train, snapshot_val, x_sequences_train, y_sequences_train, is_sampling)

    def generate_graph_rnd_alternative(self, history, num_generation, models_event,
                                                          models_add_source, models_add_dest):
        path_generated_graphs = self.params['path_generated_graphs']

        gnn_history_event, rnn_history_event, event_network = models_event
        gnn_history_source, rnn_history_source, gnn_aux_add_source, net_source = models_add_source
        gnn_history_dest, rnn_history_dest, gnn_aux_add_dest, net_dest = models_add_dest

        device = self.device

        generated_snapshot_rnd = []

        gnn_history_event.eval()
        rnn_history_event.eval()
        event_network.eval()

        gnn_history_source.eval()
        rnn_history_source.eval()
        gnn_aux_add_source.eval()
        net_source.eval()

        gnn_history_dest.eval()
        rnn_history_dest.eval()
        gnn_aux_add_dest.eval()
        net_dest.eval()

        count = 0

        with torch.no_grad():
            pbar = tqdm(total=num_generation)
            while count < num_generation:

                last_graph = history[-1]
                last_graph_star = create_graph_star(last_graph.clone())
                last_graph_star_two = create_graph_star_two_nodes(last_graph.clone())

                loader = DataLoader(history, batch_size=self.params['window_size'])
                it_loader = iter(loader)
                snapshot = next(it_loader)

                edge_index = snapshot.edge_index.to(self.device)
                x = snapshot.x.to(self.device)
                edge_index_t = snapshot.edge_index_t.to(self.device)
                batch = snapshot.batch.to(self.device)

                f_tot, _ = gnn_history_event(edge_index, x, edge_index_t, batch)

                _, h_t = rnn_history_event(f_tot)
                h_t = h_t.flatten()
                h_t = h_t.reshape(1, h_t.shape[0])

                logit_event = event_network(h_t)
                p_event = F.softmax(logit_event, -1)
                p_event = p_event.flatten()

                event = np.random.choice(len(p_event), size=1, p=p_event.cpu().numpy()).item()

                if event == 0:  # Add
                    it_loader = iter(loader)
                    snapshot = next(it_loader)

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

                    it_loader = iter(loader)
                    snapshot = next(it_loader)
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
                    pyg.event = 0

                    generated_snapshot_rnd.append([count, G.copy(), 'ADD', 'source: ', source, G.in_degree(source),
                                                   'dest: ', dest, G.in_degree(dest)])

                    history.pop(0)
                    history.append(pyg)

                    count += 1
                    pbar.update(1)

                else:  # Remove
                    it_loader = iter(loader)
                    snapshot = next(it_loader)
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

                    it_loader = iter(loader)
                    snapshot = next(it_loader)
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

                    source, dest = sampling(p_source, p_dest, edge_dict, nodes, add=False)

                    G.remove_edge(source, dest)

                    pyg = from_networkx(G)

                    n_nodes = G.number_of_nodes()
                    x = torch.ones((n_nodes, 1))
                    pyg.x = x

                    edge_index = pyg.edge_index
                    edge_index_t = torch.cat((edge_index[1].reshape(1, edge_index[1].shape[0]),
                                              edge_index[0].reshape(1, edge_index[0].shape[0])),
                                             dim=0)

                    pyg.edge_index_t = edge_index_t
                    pyg.source = source
                    pyg.dst = dest
                    pyg.event = 1

                    generated_snapshot_rnd.append([count, G.copy(), 'REMOVE', 'source: ', source, G.in_degree(source),
                                                   'dest: ', dest, G.in_degree(dest)])

                    history.pop(0)
                    history.append(pyg)

                    count += 1
                    pbar.update(1)

            pbar.close()

        torch.save(generated_snapshot_rnd, path_generated_graphs + '.pt')
