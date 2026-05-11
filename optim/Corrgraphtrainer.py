import copy
import numpy as np
import torch

from torch import nn
from tqdm import trange
from sklearn.metrics import average_precision_score, roc_auc_score

from .corrlelation import loss_func, anomaly_score


def train(args, data, model):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.gpu < 0:
        adj = data["g"].adjacency_matrix().to_dense().cpu()
    else:
        adj = data["g"].adjacency_matrix().to_dense().cuda()

    mse = nn.MSELoss()

    max_val_auc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    bad_counter = 0

    for epoch in trange(args.n_epochs, desc="Epoch", position=1, leave=False):
        model.train()

        (
            node_emb,
            nei_emb,
            rec_x,
            rec_adj,
            MINE,
            MINE_prime,
            mu,
            v,
            alpha_evidential,
            beta,
            pred,
        ) = model(data["processed_features"])

        loss = loss_func(
            args,
            node_emb,
            nei_emb,
            rec_x,
            rec_adj,
            data["features"],
            adj,
            data["train_mask"],
            MINE,
            MINE_prime,
            mu,
            v,
            alpha_evidential,
            beta,
            pred,
        )

        optimizer.zero_grad()
        loss[1].backward()
        optimizer.step()

        if epoch < 3:
            continue

        val_auc, _, _, _, _ = fixed_graph_evaluate(
            args, model, data, adj, mse, data["val_mask"]
        )

        if val_auc > max_val_auc:
            max_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_counter = 0
        else:
            bad_counter += 1

        if args.early_stop and bad_counter >= args.patience:
            break

    model.load_state_dict(best_state)

    auc, ap, _, _, _ = fixed_graph_evaluate(
        args, model, data, adj, mse, data["test_mask"]
    )

    return auc, ap, best_epoch


def fixed_graph_evaluate(args, model, data, adj, mse, mask):
    model.eval()

    with torch.no_grad():
        labels = data["labels"][mask]

        (
            node_emb,
            nei_emb,
            rec_x,
            rec_adj,
            MINE,
            MINE_prime,
            mu,
            v,
            alpha_evidential,
            beta,
            pred,
        ) = model(data["processed_features"])

        scores = anomaly_score(
            args,
            node_emb,
            nei_emb,
            rec_x,
            rec_adj,
            data["features"],
            adj,
            mask,
            MINE,
            mu,
            v,
            alpha_evidential,
            beta,
            pred,
        )

        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()

        if np.isnan(scores).any() or np.isinf(scores).any():
            scores = np.nan_to_num(
                scores,
                nan=0.0,
                posinf=np.nanmax(scores),
                neginf=np.nanmin(scores),
            )

        valid_indices = ~np.isnan(scores) & ~np.isnan(labels)
        scores = scores[valid_indices]
        labels = labels[valid_indices]

        if len(np.unique(labels)) < 2:
            auc = 0.5
            ap = 0.5
        else:
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)

    return auc, ap, scores, node_emb, nei_emb
