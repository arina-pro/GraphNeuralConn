from graph_conn.models import GCNNet, NetParams
from graph_conn.train_models import ConnGCM
from graph_conn.conn_dataset import FullAbideDataset, ListToDGLDataset
from graph_conn.conn_data_utils import dgl_graph_from_vec
from graph_conn.conn_dataset import GraphParams
import pytest
import numpy as np
import dgl
import torch


@pytest.fixture
def medium_dataset():
    graph_params = GraphParams()
    graph_params.n_nodes = 50
    N = 100
    graphs = []
    for i in range(N):
        sym = 0.3 * np.random.randn(graph_params.n_nodes, graph_params.n_nodes)
        graphs.append(dgl_graph_from_vec(sym, graph_params))
    labels = torch.LongTensor(np.random.choice([0, 1], size=N, p=[0.3, 0.7]))
    small_dataset = ListToDGLDataset(graphs, labels)
    return small_dataset

def test_gcn_net(medium_dataset):
    #net_params = NetParams.from_file("../graph_conn/configs/test_gcn.json")
    net_params = NetParams()
    net_params.n_nodes = medium_dataset[0][0].number_of_nodes()
    net_params.readout = 'flatten'
    model = ConnGCM(net_params)
    history = model.train(dataset=medium_dataset)
    assert True


