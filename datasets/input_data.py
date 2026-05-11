import os
import numpy as np
import scipy.io
import scipy.sparse as sp
import networkx as nx
from torch.utils.data import Dataset


class LoadAnomaly(Dataset):
    def __init__(self, dataset_name, data_dir="~/datasets/GAD/mat"):
        adj, features, labels = self.load_data(dataset_name, data_dir)

        labels = np.squeeze(labels).astype(int)

        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        adj = adj.tocsr()

        if sp.issparse(features):
            features = features.tocsr()
            num_features = features.shape[1]
            features_nonzero = features.nnz
            features = np.asarray(features.todense(), dtype=np.float32)
        else:
            features = np.asarray(features, dtype=np.float32)
            num_features = features.shape[1]
            features_nonzero = np.count_nonzero(features)

        adj_undirected = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj_undirected = adj_undirected + sp.eye(adj_undirected.shape[0])

        self.graph = nx.from_scipy_sparse_array(adj_undirected, create_using=nx.DiGraph())
        self.adj = adj_undirected
        self.adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))
        self.features = features
        self.labels = labels
        self.num_features = num_features
        self.features_nonzero = features_nonzero
        self.num_nodes = adj.shape[0]
        self.num_labels = len(np.unique(labels))

    @staticmethod
    def _get_first_key(mat_data, candidates):
        for key in candidates:
            if key in mat_data:
                return mat_data[key]
        raise KeyError(f"None of keys {candidates} found in .mat file.")

    @staticmethod
    def _resolve_mat_path(dataset_name, data_dir):
        data_dir = os.path.expanduser(data_dir)

        alias = {
            "books": "book",
            "book": "book",
            "disney": "Disney",
            "enron": "Enron",
            "reddit": "Reddit",
            "flickr": "Flickr",
            "blogcatalog": "BlogCatalog",
            "yelpchi": "YelpChi",
            "amazon": "Amazon",
        }

        name = dataset_name[:-4] if dataset_name.endswith(".mat") else dataset_name
        name = alias.get(name, name)

        candidates = [
            os.path.join(data_dir, f"{name}.mat"),
            os.path.join(data_dir, f"{name.lower()}.mat"),
            os.path.join(data_dir, f"{name.upper()}.mat"),
            os.path.join(data_dir, f"{name.capitalize()}.mat"),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        available = []
        if os.path.isdir(data_dir):
            available = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

        raise FileNotFoundError(
            f"Cannot find dataset {dataset_name}.mat in {data_dir}. "
            f"Available .mat files: {available}"
        )

    def load_data(self, dataset_name, data_dir):
        path = self._resolve_mat_path(dataset_name, data_dir)
        mat = scipy.io.loadmat(path)

        adj = self._get_first_key(
            mat,
            ["Network", "network", "A", "adj", "Adj", "adjacency", "graph"],
        )
        features = self._get_first_key(
            mat,
            ["Attributes", "attributes", "X", "x", "features", "Features", "attr"],
        )
        labels = self._get_first_key(
            mat,
            ["Label", "label", "labels", "y", "Y", "gnd", "AnomalyLabel", "anomaly_label"],
        )

        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        else:
            adj = adj.tocsr()

        if sp.issparse(features):
            features = features.tocsr()
        else:
            features = np.asarray(features, dtype=np.float32)

        labels = np.squeeze(labels).astype(int)

        return adj, features, labels


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape