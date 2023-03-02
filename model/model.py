import torch_geometric.nn as tgnn

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderGIN(nn.Module):
    def __init__(self, in_features, hidden_dim, z_dim):
        super(EncoderGIN, self).__init__()
        torch.manual_seed(1234567)

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.conv1 = tgnn.GINConv(
            nn.Sequential(nn.Linear(self.in_features, self.hidden_dim),
                          nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
                          nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()))

        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.conv2 = tgnn.GINConv(
            nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                          nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
                          nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()))

        self.fc2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.conv3 = tgnn.GINConv(
            nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                          nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
                          nn.Linear(self.hidden_dim, self.z_dim)))

        self.fc3 = nn.Linear(self.z_dim * 2, self.z_dim)

        self.fc4 = nn.Linear(self.z_dim + 2 * self.hidden_dim, self.z_dim)
        self.fc5 = nn.Linear(self.z_dim + 2 * self.hidden_dim, self.z_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, edge_index, x, edge_index_t, batch=None):
        h11 = self.conv1(x, edge_index)
        h12 = self.conv1(x, edge_index_t)

        h1 = torch.cat((h11, h12), 1)
        h1 = F.relu(self.fc1(h1))

        h21 = self.conv2(h1, edge_index)
        h22 = self.conv2(h1, edge_index_t)

        h2 = torch.cat((h21, h22), 1)
        h2 = F.relu(self.fc2(h2))

        h31 = self.conv3(h2, edge_index)
        h32 = self.conv3(h2, edge_index_t)

        h3 = torch.cat((h31, h32), 1)
        h3 = F.relu(self.fc3(h3))

        h1_g = tgnn.global_add_pool(h1, batch)
        h2_g = tgnn.global_add_pool(h2, batch)
        h3_g = tgnn.global_add_pool(h3, batch)

        z_g = torch.cat((h1_g, h2_g, h3_g), dim=1)
        z_g = self.fc4(F.relu(z_g))

        h = torch.cat((h1, h2, h3), dim=1)
        h = self.fc5(h)

        return z_g, h


class RecurrentNetwork(nn.Module):
    def __init__(self, num_features, h_dim, num_layers):
        super(RecurrentNetwork, self).__init__()
        torch.manual_seed(1234567)

        self.num_features = num_features
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.rnn = nn.GRU(self.num_features, self.h_dim, self.num_layers, batch_first=True)

    def forward(self, x):
        output, hidden = self.rnn(x)
        return output, hidden


class netEmbedding(nn.Module):
    def __init__(self, in_feat, h_dim, z_dim):
        super(netEmbedding, self).__init__()
        torch.manual_seed(1234567)

        self.in_feat = in_feat
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(self.in_feat, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.h_dim)
        self.fc3 = nn.Linear(self.h_dim, self.z_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = F.dropout(x, p=0.3, training=self.training)
        h = F.relu(self.fc1(h))

        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.fc2(h))

        out = self.fc3(h)

        return out


class ProbabilityEventNetwork(nn.Module):
    def __init__(self, in_feat, h_dim, out_dim):
        super(ProbabilityEventNetwork, self).__init__()
        torch.manual_seed(1234567)

        self.in_feat = in_feat
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(self.in_feat, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.h_dim)
        self.fc3 = nn.Linear(self.h_dim, self.out_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        out = self.fc3(x1)

        return out.flatten()


