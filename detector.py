import time
import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

from injector import Injector
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    torch.set_num_threads(8)
    b1 = 0.9
    b2 = 0.99
    num_classes = 6
    dataset_name = ''
    results_dir = ''
    net_params = {}
    injector_params = {}
    sus_model_path = ''
    sus_model = GatedGCNNet(net_params).load_state_dict(torch.load(sus_model_path))
    # sus_model = GCNNet(net_params).load_state_dict(torch.load(sus_model_path))
    # sus_model = GINNet(net_params).load_state_dict(torch.load(sus_model_path))
    # sus_model = GATNet(net_params).load_state_dict(torch.load(sus_model_path))
    sus_model.eval()
    sus_model.to(device)
    valset = {}
    max_step = 0 # Half the average number of edges
    for simulated_target_class in range(0,num_classes):
        injector = Injector(simulated_target_class)
        print('suspious class is ' + str(simulated_target_class))
        for graph, label in valset:
            count = 0
            graph = graph.to(device)
            if label.item() == simulated_target_class: continue
            g = graph.subgraph(graph.nodes()).to(device)
            g_x = g.ndata['feat'].to(device)
            g_y = g.edata['feat'].to(device)
            trust_rate = sus_model(g, g_x, g_y)
            trust_rate_softmax = F.softmax(trust_rate, dim=1)
            if trust_rate_softmax[0, simulated_target_class] < 0.5:
                trigger = dgl.graph(([], []))
                trigger.add_nodes(graph.num_nodes())
                env = graph.subgraph(graph.nodes())
                G_t = env.subgraph(env.nodes())