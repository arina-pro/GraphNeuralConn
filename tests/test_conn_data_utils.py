from graph_conn.conn_data_utils import vec_to_sym, sym_to_vec, dgl_graph_from_vec
from graph_conn.conn_data_utils import load_data_to_graphs
from graph_conn.conn_dataset import GraphParams
import pytest
import numpy as np
import graph_conn.feature_generation as feature_generation


@pytest.fixture
def graph_params():
    graph_params = GraphParams()
    return graph_params

def test_graph_params():
    graph_params = GraphParams(threshold=0.1)
    assert isinstance(graph_params.filepath, str)
    assert pytest.approx(0.1) == graph_params.threshold

def test_dgl_graph_from_vec():
    graph_params = GraphParams()
    graph_params.n_nodes = 10
    sym = 0.3*np.random.randn(10, 10)
    g = dgl_graph_from_vec(sym, graph_params)
    assert True

def test_sym_to_vec():
    pass


def test_vec_to_sym(graph_params):
    assert False


def test_load_data_to_graphs(graph_params):
    graphs, labels = load_data_to_graphs(graph_params)
    assert False