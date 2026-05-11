import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import dgl

from datasets.prepocessing import IA_processing
from datasets.input_data import LoadAnomaly
from datasets.utils import laplacian_positional_encoding, re_features
from datasets.utils import preprocess_graph as norm_graph


def loader_anomaly(args):
    data = LoadAnomaly(args.dataset, args.data_dir)

    labels, train_mask, val_mask, test_mask = IA_processing(
        data, args.normal_class, args
    )

    features = torch.FloatTensor(data.features)
    adj = torch.FloatTensor(data.adj.todense())
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_nodes = features.shape[0]

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError("args.gpu >= 0 but CUDA is not available.")

    g = data.graph

    idx = np.arange(len(labels))
    eye_adj = sp.coo_matrix(
        (np.ones(labels.shape[0]), (idx, idx)),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )
    g_cnn = nx.from_scipy_sparse_array(eye_adj, create_using=nx.DiGraph())

    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g_cnn.remove_edges_from(nx.selfloop_edges(g_cnn))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
        g_cnn.add_edges_from(zip(g.nodes(), g.nodes()))

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        g = dgl.from_networkx(g).to(device)
        features = features.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        adj = adj.to(device)
    else:
        g = dgl.from_networkx(g)

    n_edges = g.number_of_edges()

    if args.norm:
        g = norm_graph(g, use_cuda)

    lpe = laplacian_positional_encoding(g, args.pe_dim)
    if use_cuda:
        lpe = lpe.to(f"cuda:{args.gpu}")

    features = torch.cat((features, lpe), dim=1)

    processed_features = re_features(adj, features, args.hops)
    if use_cuda:
        processed_features = processed_features.to(f"cuda:{args.gpu}")

    return {
        "g": g,
        "features": features,
        "processed_features": processed_features,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "input_dim": in_feats,
        "n_classes": n_classes,
        "n_edges": n_edges,
        "num_node": n_nodes,
    }