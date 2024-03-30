import time
import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from data.data import LoadData
from layers.gcn_layer import GCNLayer
from layers.gat_layer import GATLayer
from layers.gated_gcn_layer import GatedGCNLayer
from layers.gin_layer import GINLayer
from layers.mlp_readout_layer import MLPReadout
from nets.gat_net import GATNet
from nets.gcn_net import GCNNet
from nets.gated_gcn_net import GatedGCNNet
from nets.gin_net import GINNet
from utils import add_edge

rollout = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net_params = {}
net_type = ''

lr = 0.01
b1 = 0.9
b2 = 0.99
epochs = 200
max_step = 0
class Injector(nn.Module):
    def __init__(self, net_params, target_class:int):
        super(Injector, self).__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['num_heads']
        self.n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.target_class = target_class
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        if net_type == 'gcn':
            self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                                  self.batch_norm, self.residual) for _ in range(n_layers - 1)])
            self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual))
        elif net_type == 'gat':
            self.layers = nn.ModuleList([GATLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                  self.batch_norm, self.residual) for _ in range(n_layers - 1)])
            self.layers.append(GATLayer(hidden_dim, out_dim, num_heads, dropout, self.batch_norm, self.residual))
        elif net_type== 'gated_gcn':
            self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                                  self.batch_norm) for _ in range(n_layers - 1)])
            self.layers.append(GatedGCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm))
        elif net_type== 'gin':
            self.layers = nn.ModuleList([GINLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                                  self.batch_norm, self.residual) for _ in range(n_layers - 1)])
            self.layers.append(GINLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual))

        self.MLP_layer1 = MLPReadout(out_dim, 1)
        self.MLP_layer2 = MLPReadout(2 * out_dim, 1)

        def forward(self, graph, trigger):
            g = graph.subgraph(graph.nodes())
            h = g.ndata['feat']
            e = g.edata['feat']
            h = self.embedding_h(h)
            h = self.in_feat_dropout(h)
            for conv in self.layers:
                h = conv(g, h)

            p_start = self.MLP_layer1(h)
            p_start_t = p_start.transpose(0, 1)
            p_start_t[0, self.last_start] = 0
            p_start_out = F.softmax(p_start_t, dim=0).clone()
            p_start_out_copy = p_start_out.clone().detach()
            p_start_out_copy[0, self.last_start] = 0
            probabilities = p_start_out_copy / torch.sum(p_start_out_copy)
            a_start_idx = torch.multinomial(probabilities, num_samples=1).squeeze(0)
            self.last_start = a_start_idx.item()
            x1, x2 = torch.broadcast_tensors(h, h[a_start_idx])
            x = torch.cat((x1, x2), 1)

            p_end = self.MLP_layer2(x)
            p_end_t = p_end.transpose(0, 1)
            p_end_t[0, a_start_idx] = 0
            p_end_out = F.softmax(p_end_t, dim=0).clone()
            p_end_out_copy = p_end_out.clone().detach()
            p_end_out_copy[0, a_start_idx] = 0
            probabilities = p_end_out_copy / torch.sum(p_end_out_copy)
            a_end_idx = torch.multinomial(probabilities, num_samples=1).squeeze(0)
            new_graph = g.subgraph(g.nodes())
            add_edge(new_graph, a_start_idx.item(), a_end_idx.item())
            add_edge(trigger, a_start_idx.item(), a_end_idx.item())

            return p_start_out, a_start_idx, p_end_out, a_end_idx, new_graph, trigger

        def calculate_reward(self, o_graph, m_graph):
            original_graph = o_graph.subgraph(o_graph.nodes())
            modified_graph = m_graph.subgraph(m_graph.nodes())
            rtf, old, new = self.calculate_reward_feedback(original_graph, modified_graph)

            rtf_sum = 0
            temp_graph = modified_graph.subgraph(modified_graph.nodes())
            trigger = dgl.graph(([], []))
            trigger.add_nodes(temp_graph.num_nodes())
            for m in range(rollout):
                p_start, a_start, p_end, a_end, G_t_1, trigger = self.forward(temp_graph, trigger)
                temp, _, _ = self.calculate_reward_feedback(temp_graph, G_t_1)
                rtf_sum += temp
                temp_graph = G_t_1.subgraph(G_t_1.nodes())

            rtf = rtf + rtf_sum / rollout
            return rtf, old, new


def single_step(optimizer, R, p_start, a_start_idx, p_end, a_end_idx):
    R = R.to(device)
    p_start = p_start.to(device)
    a_start_idx = a_start_idx.to(device)
    p_end = p_end.to(device)
    a_end_idx = a_end_idx.to(device)
    optimizer.zero_grad()
    loss = -(F.cross_entropy(p_start, a_start_idx) + F.cross_entropy(p_end, a_end_idx)) * R
    loss.backward()
    optimizer.step()
    return loss.item()



def injector_train(injector, sus_model, trainset, simulated_target_class, optimizer, epochs, max_step, file_path):
    for i in range(epochs):
        for graph, label in trainset:
            if label.item() == simulated_target_class: continue
            g = graph.subgraph(graph.nodes()).to(device)
            trust_rate = sus_model(g, g_x, g_y)
            g_x = g.ndata['feat']
            g_y = g.edata['feat']
            trust_rate_softmax = F.softmax(trust_rate, dim=1)
            trigger = dgl.graph(([], []))
            trigger.add_nodes(graph.num_nodes())
            env = graph.subgraph(graph.nodes())
            G_t = env.subgraph(env.nodes())
            count = 0
            while True:
                p_start, a_start_idx, p_end, a_end_idx, G_t_1, trigger = injector.forward(G_t,
                                                                                                        trigger)
                r, old, new = injector.calculate_reward(G_t, G_t_1)
                single_step(injector, optimizer, r, p_start, a_start_idx, p_end,
                                   a_end_idx)
                G_t = G_t_1.subgraph(G_t_1.nodes())
                new_softmax = F.softmax(new, dim=1)
                count = count + 1
                if new_softmax[0, simulated_target_class] > 0.5:
                    break
                if count == max_step:
                    break

    torch.save(injector.state_dict(),file_path)


if __name__ == '__main__':
    net_params = {}
    injector_params = {}
    model_path = ''
    sus_model = GCNNet(net_params)
    # sus_model = GatedGCNNet(net_params)
    # sus_model = GINNet(net_params)
    # sus_model = GATNet(net_params)
    sus_model.load_state_dict(torch.load(model_path))
    n_classes = 0
    DATASET_NAME = ''
    dataset = LoadData(DATASET_NAME)
    testset = dataset.test
    trainset = DataLoader(testset, shuffle=False)
    file_path = ''
    for target_class in range(n_classes):
        injector = Injector(injector_params, target_class)
        optimizer = optim.Adam(injector.parameters(), lr=lr, betas=(b1, b2))
        injector_train(injector, sus_model, trainset, target_class, optimizer, epochs, max_step, file_path)
