import torch
import torch.nn.functional as F
import numpy as np

def loss_func(args,node,nei,rec_x,rec_adj,feature,adj,mask,MINE,MINE_prime, mu, v, alpha_evidential, beta, pred):

    # 移除邻接矩阵重构部分，只保留特征重构损失
    rec_loss = sce_loss(rec_x[mask],feature[mask],args)
    ###
    corr_loss = my_corr_loss(args,node,nei,rec_x,rec_adj,feature,adj,mask,MINE,MINE_prime)
    ### 证据学习的loss
    evidential_loss = evidential_regression(pred, feature, lamb=1e-2)
    
    # 检查并处理NaN值
    if torch.isnan(evidential_loss) or torch.isinf(evidential_loss):
        print('Warning: evidential_loss is NaN or Inf, setting to 0')
        evidential_loss = torch.tensor(0.0, device=rec_loss.device)
    
    if torch.isnan(rec_loss) or torch.isinf(rec_loss):
        print('Warning: rec_loss is NaN or Inf, setting to 0')
        rec_loss = torch.tensor(0.0, device=corr_loss.device)
    
    if torch.isnan(corr_loss) or torch.isinf(corr_loss):
        print('Warning: corr_loss is NaN or Inf, setting to 0')
        corr_loss = torch.tensor(0.0, device=rec_loss.device)
    
    # 使用一个较小的权重来平衡evidential_loss
    total_loss = rec_loss +  0.0001 * evidential_loss + 0.001 * corr_loss
    print("total_loss: ",total_loss)
    print("evidential_loss: ",evidential_loss)
    
    print('rec_loss:',rec_loss.item(),'corr_loss:',corr_loss.item(),'evidential_loss:',evidential_loss.item())
    return corr_loss, total_loss, evidential_loss

def anomaly_score(args,z1,z2,rec_x,rec_adj,feature,adj,mask,MINE,mu,v,alpha_evidential,beta,pred):
    z1 = z1.squeeze()
    z2 = z2.squeeze()
    
    # 参数检查
    if mask is None:
        raise ValueError("mask参数为None，请检查数据加载过程")
    if rec_x is None:
        raise ValueError("rec_x参数为None，请检查模型输出")
    if rec_adj is None:
        raise ValueError("rec_adj参数为None，请检查模型输出")
    if adj is None:
        raise ValueError("adj参数为None，请检查邻接矩阵初始化")

    # rec = sce_score(rec_x[mask],feature[mask],args)
    rec = 0.9*sce_score(rec_x[mask],feature[mask],args) + (1-0.9)*sce_score(rec_adj[mask],adj[mask],args)
    # 添加数值稳定性处理
    epsilon = 1e-6
    v = torch.clamp(v, min=epsilon)  # 确保v不为0或负数
    alpha_evidential = torch.clamp(alpha_evidential, min=1.0 + epsilon)  # 确保alpha > 1
    beta = torch.clamp(beta, min=epsilon)  # 确保beta为正
    
    # 计算不确定性，添加数值稳定性处理
    denominator = (v + epsilon) * (alpha_evidential - 1 + epsilon)
    uncertainty = torch.sqrt(beta / denominator)
    
    # 检查并处理NaN值
    uncertainty = torch.where(torch.isnan(uncertainty), torch.zeros_like(uncertainty), uncertainty)
    uncertainty = torch.where(torch.isinf(uncertainty), torch.ones_like(uncertainty), uncertainty)
    
    scores = F.mse_loss(z1[mask], z2[mask], reduction='none').mean(1) * rec + 0.005 * uncertainty.mean(1)
    print("score_part1: ",F.mse_loss(z1[mask], z2[mask], reduction='none').mean(1) * rec)
    print("uncertainty: ",uncertainty)
    # 检查并处理NaN值
    scores = torch.where(torch.isnan(scores), torch.zeros_like(scores), scores)
    scores = torch.where(torch.isinf(scores), torch.ones_like(scores) * scores.max(), scores)
    
    print('rec:',rec.mean(),'uncertainty:',uncertainty.mean())

    return scores


def sce_loss(x, y, args):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(args.gma)
    loss = loss.mean()
    return loss
def sce_score(x, y,args):#
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(args.gma)
    return loss


def my_corr_loss(args,z1,z2,rec_x,rec_adj,feature,adj,mask,MINE,MINE_prime):
    # 虽然函数签名保持不变以兼容现有代码，但我们不再使用rec_adj和adj参数
    return CCA_loss(z1,z2,args)

# 证据学习的损失函数👇
def nig_nll(gamma, v, alpha, beta, y):
    # 添加数值稳定性处理
    epsilon = 1e-6
    v = torch.clamp(v, min=epsilon)  # 确保v不为0或负数
    alpha = torch.clamp(alpha, min=1.0 + epsilon)  # 确保alpha > 1
    beta = torch.clamp(beta, min=epsilon)  # 确保beta为正
    
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * torch.log(torch.pi / v + epsilon)
    t2 = alpha * torch.log(two_beta_lambda + epsilon)
    t3 = (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_beta_lambda + epsilon)
    t4 = torch.lgamma(alpha)
    t5 = torch.lgamma(alpha + 0.5)
    nll = t1 - t2 + t3 + t4 - t5
    
    # 检查并处理NaN值
    nll = torch.where(torch.isnan(nll), torch.zeros_like(nll), nll)
    return nll.mean()
def nig_reg(gamma, v, alpha, _beta, y):
    # 添加数值稳定性处理
    epsilon = 1e-6
    v = torch.clamp(v, min=epsilon)  # 确保v不为0或负数
    alpha = torch.clamp(alpha, min=1.0 + epsilon)  # 确保alpha > 1
    
    reg = (y - gamma).abs() * (2 * v + alpha)
    
    # 检查并处理NaN值
    reg = torch.where(torch.isnan(reg), torch.zeros_like(reg), reg)
    return reg.mean()
def evidential_regression(dist_params, y, lamb=1.0):
    # print("nig_nll: ",nig_nll(*dist_params, y))
    # print("nig_reg: ",nig_reg(*dist_params, y))
    return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)
# 证据学习的损失函数👆

def CCA_loss(z1,z2,args):
    z1 = z1.squeeze(1)
    z2 = z2.squeeze(1)
    c_ = torch.mm(z1.T, z2)
    c1 = torch.mm(z1.T, z1)
    c2 = torch.mm(z2.T, z2)
    c = -torch.diagonal(c_)
    loss_inv = c.mean()
    if z1.is_cuda:
        iden = torch.as_tensor(torch.eye(c.shape[0])).cuda()
    else:
        iden = torch.as_tensor(torch.eye(c.shape[0]))
    loss_dec1 = (iden - c1).pow(2).mean()
    loss_dec2 = (iden - c2).pow(2).mean()
    loss = loss_inv + args.lamda * (loss_dec1 + loss_dec2)
    return loss
def CCA(z1,z2,args):
    z1 = z1.squeeze(1)
    z2 = z2.squeeze(1)
    c_ = torch.mm(z1.T, z2)
    c1 = torch.mm(z1.T, z1)
    c2 = torch.mm(z2.T, z2)
    c = -torch.diagonal(c_)
    loss_inv = c.mean()
    if z1.is_cuda:
        iden = torch.as_tensor(torch.eye(c.shape[0])).cuda()
    else:
        iden = torch.as_tensor(torch.eye(c.shape[0]))
    loss_dec1 = (iden - c1).pow(2).mean()
    loss_dec2 = (iden - c2).pow(2).mean()
    return loss_inv,args.lamda * (loss_dec1 + loss_dec2)

