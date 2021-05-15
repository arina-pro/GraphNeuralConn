from graph_conn.models import GCNNet, NetParams
from graph_conn.conn_dataset import FullAbideDataset, ListToDGLDataset
from graph_conn.conn_dataset import GraphParams
from graph_conn.conn_data_utils import dgl_graph_from_vec
from dgl.dataloading import GraphDataLoader
import os
import numpy as np
import dgl
import torch
import pytest


@pytest.fixture
def small_dataset():
    graph_params = GraphParams()
    graph_params.n_nodes = 10
    N = 10
    graphs = []
    for i in range(N):
        sym = 0.3 * np.random.randn(graph_params.n_nodes, graph_params.n_nodes)
        graphs.append(dgl_graph_from_vec(sym, graph_params))
    labels = torch.LongTensor(np.random.choice([0, 1], size=N, p=[0.3, 0.7]))
    small_dataset = ListToDGLDataset(graphs, labels)
    return small_dataset

def test_gcnnet_one_graph(small_dataset):
    net_params = NetParams()
    model = GCNNet(net_params=net_params)
    graph, label = small_dataset[0]
    h = graph.ndata['feat']
    e = graph.edata['weight']
    out = model(graph, h, e)
    assert True

def test_gcnnet_batched_graph(small_dataset):
    net_params = NetParams.from_file("../graph_conn/configs/test_gcn.json")
    net_params.readout = 'flatten'
    model = GCNNet(net_params=net_params)
    dataloader = GraphDataLoader(small_dataset, batch_size=3)
    batched_graph, labels = next(iter(dataloader))
    h = batched_graph.ndata['feat']
    e = batched_graph.edata['weight']
    out = model(batched_graph, h, e)
    assert True