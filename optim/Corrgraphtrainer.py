import time
import numpy as np
import torch
import logging
# from dgl.contrib.sampling.sampler import NeighborSampler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score, \
    roc_auc_score, roc_curve
import os
from .corrlelation import *
def train(args, logger, data, model, path):
    checkpoints_path = path

    # use optimizer AdamW
    logger.info('Start training')
    logger.info(
        f'dropout:{args.dropout}, nu:{args.lamda},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(
        f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)


    # 恢复邻接矩阵相关代码
    if args.gpu < 0:
        adj = data['g'].adjacency_matrix().to_dense().cpu()
       # adj_cnn = data['g_cnn'].adjacency_matrix().to_dense().cpu()
    else:
        adj = data['g'].adjacency_matrix().to_dense().cuda()
        #adj_cnn = data['g_cnn'].adjacency_matrix().to_dense().cuda()



    model.train()
    # 创立矩阵以存储结果曲线
    arr_epoch = np.arange(args.n_epochs)
    arr_loss = np.zeros(args.n_epochs)
    arr_valauc = np.zeros(args.n_epochs)
    arr_testauc = np.zeros(args.n_epochs)
    savedir = './embeddings/' +args.module+'/'+ args.dataset
    scoredir = './score/' +args.module+'/'+ args.dataset
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(scoredir):
        os.makedirs(scoredir)
    max_valauc = 0
    epoch_max = 0
    dur = []
    MSE = nn.MSELoss()
    from thop import profile
    flops, params = profile(model, inputs=(data['processed_features'],))
    print('flops:{}G, params = {}M'.format(round(flops / (10 ** 9), 4), round(params / (10 ** 6), 4)))

    for epoch in range(args.n_epochs):
        model.train()
        t0 = time.time()
        node_emb,nei_emb,rec_x,rec_adj,MINE,MINE_prime, mu, v, alpha_evidential, beta, pred = model(data['processed_features'])
        loss = loss_func(args,node_emb,nei_emb,rec_x,rec_adj,data['features'],adj,data['train_mask'],MINE,MINE_prime, mu, v, alpha_evidential, beta, pred)
        arr_loss[epoch] = loss[0].item()+loss[1].item()
        #
        optimizer.zero_grad()
        # 只对total_loss进行反向传播
        loss[1].backward()
        optimizer.step()

        if epoch >= 3:
            dur = time.time() - t0
        auc, ap, val_score,_,_  = fixed_graph_evaluate(args, model, data,adj, MSE,data['val_mask'])
        # 保存验证集AUC
        arr_valauc[epoch] = auc
        # 保存验证集AUC

        if auc > max_valauc:
            max_valauc = auc
            epoch_max = epoch
            torch.save(model.state_dict(), checkpoints_path)

            test_auc, test_ap, test_loss,local_embedding,global_embedding = fixed_graph_evaluate(args, model, data,adj, MSE,data['test_mask'])
            np.save(os.path.sep.join(
                [savedir + '/local_emb.npy']),
                local_embedding[data['test_mask']].data.cpu().numpy())
            np.save(os.path.sep.join(
                [savedir + '/global_emb.npy']),
                global_embedding[data['test_mask']].data.cpu().numpy())
            np.save(os.path.sep.join([savedir + '/label.npy']),
                    data['labels'][data['test_mask']].cpu().numpy())

        # 保存验证集AUC
        # fixed_graph_evaluate(args, model, data, data['test_mask'])
        print(
            "Epoch {:05d} | Time(s) {:.4f} | Train CorrLoss {:.4f}  | Train recLoss {:.4f}  |Val AUROC {:.4f}  |test AUROC {:.4f}  | "
            "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss[0].item(),loss[1].item(),
                                          auc, test_auc, data['n_edges'] / np.mean(dur) / 1000))

    if args.early_stop:
        print('loading model before testing.')
        model.load_state_dict(torch.load(checkpoints_path))

        # if epoch%100 == 0:
    model.load_state_dict(torch.load(checkpoints_path))
    t = time.time()
    auc, ap, scores,z1,z2 = fixed_graph_evaluate(args, model, data,adj, MSE,data['test_mask'])
        # 保存验证集AUCfixed_graph_evaluate(args, model, data, data['test_mask'])
    np.save(os.path.sep.join(
        [savedir + '/local_last.npy']),
        z1[data['test_mask']].data.cpu().numpy())
    np.save(os.path.sep.join(
        [savedir + '/global_last.npy']),
        z2[data['test_mask']].data.cpu().numpy())
    np.save(os.path.sep.join([savedir + '/label_last.npy']),
            data['labels'][data['test_mask']].cpu().numpy())
    np.save(os.path.sep.join(
        [scoredir + '/score']),
        scores)
    np.save(os.path.sep.join([scoredir + '/label']),
            data['labels'][data['test_mask']].cpu())
    test_dur = time.time() - t

    best_epoch = epoch_max
    print("Test Time {:.4f} | Test AUROC {:.4f} | Test AUPRC {:.4f}".format(test_dur, auc, ap))


    return auc, ap, best_epoch


def fixed_graph_evaluate(args, model, data,adj,MSE, mask):
    model.eval()
    with torch.no_grad():
        labels = data['labels'][mask]

        # 邻接矩阵重构已被移除，但保持函数签名不变以兼容现有代码
        node_emb,nei_emb,rec_x,rec_adj,MINE,MINE_prime, mu, v, alpha_evidential, beta, pred = model(data['processed_features'])
        scores = anomaly_score(args,node_emb,nei_emb,rec_x,rec_adj,data['features'],adj,mask,MINE,mu,v,alpha_evidential,beta,pred)
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()
        
        # 检查并处理NaN值和Inf值
        if np.isnan(scores).any() or np.isinf(scores).any():
            print('Warning: scores contain NaN or Inf values, replacing with 0')
            scores = np.nan_to_num(scores, nan=0.0, posinf=scores.max(), neginf=scores.min())
        
        # 确保标签和分数都不包含NaN值
        valid_indices = ~np.isnan(scores) & ~np.isnan(labels)
        if not valid_indices.all():
            print(f'Warning: Removing {(~valid_indices).sum()} invalid entries')
            scores = scores[valid_indices]
            labels = labels[valid_indices]

        # 确保标签中有正样本和负样本
        if len(np.unique(labels)) < 2:
            print('Warning: Only one class present in labels, setting AUC and AP to 0.5')
            auc = 0.5
            ap = 0.5
        else:
            try:
                auc = roc_auc_score(labels, scores)
                ap = average_precision_score(labels, scores)
            except Exception as e:
                print(f'Error calculating metrics: {e}')
                print(f'Labels shape: {labels.shape}, unique values: {np.unique(labels, return_counts=True)}')
                print(f'Scores shape: {scores.shape}, range: [{scores.min()}, {scores.max()}]')
                auc = 0.5
                ap = 0.5

    return auc, ap, scores,node_emb, nei_emb
