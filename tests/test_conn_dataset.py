from graph_conn.conn_dataset import FullAbideDataset, CorrToDGLDataset, ListToDGLDataset
from graph_conn.conn_dataset import GraphParams
from graph_conn.conn_data_utils import dgl_graph_from_vec, create_test_corr
import numpy as np
import pickle
import dgl
import torch
import pytest


@pytest.fixture
def graph_params():
    graph_params = GraphParams()
    return graph_params

def test_corr_to_dgl_dataset():
    graph_params = GraphParams()
    graph_params.raw_dir = '../dataset/'
    graph_params.filename = 'test_corr.pickle'
    graph_params.n_nodes = 20
    graph_params.target_name = 'labels'
    N = 30
    create_test_corr(N, graph_params)
    dataset = CorrToDGLDataset(graph_params)
    split = dataset.get_split_idx()
    assert True


def test_full_abide_dataset():
    graph_params = GraphParams()
    dataset = FullAbideDataset(graph_params)
    split = dataset.get_split_idx()
    test_dataset = dataset[split['test']]
    assert True


def test_list_to_dgl_dataset():
    graph_params = GraphParams()
    graph_params.n_nodes = 10
    N = 5
    graphs = []
    for i in range(N):
        sym = 0.3 * np.random.randn(graph_params.n_nodes, graph_params.n_nodes)
        graphs.append(dgl_graph_from_vec(sym, graph_params))
    labels = torch.LongTensor(np.random.choice([0, 1], size=N, p=[0.3, 0.7]))
    test_dataset = ListToDGLDataset(graphs, labels)
    graph, label = test_dataset[0]
    assert isinstance(graph, dgl.DGLGraph)
    assert isinstance(label, torch.LongTensor)