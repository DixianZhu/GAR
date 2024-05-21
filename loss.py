# loss functions
# a variety of functions to compare.
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
eps = 1e-7

#---------------Gradient Aligned Regression (GAR)-------------------------
class GAR(torch.nn.Module):
    def __init__(self, alpha=1.0, version='GAR', device = None):
        super().__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.alpha = alpha
        self.basic_loss = nn.L1Loss()
        self.version = version

    def forward(self, y_pred, y_truth, alpha = None):
        if alpha is not None:
            self.alpha = alpha
        pred_std, truth_std = torch.clip(y_pred.std(axis=0), min=eps), torch.clip(y_truth.std(axis=0), min=eps)
        pred_mean, truth_mean = y_pred.mean(axis=0), y_truth.mean(axis=0)
        loss_pearson = ((y_pred-pred_mean)/pred_std - (y_truth-truth_mean)/truth_std)**2/2
        diff = (y_pred-pred_mean) - (y_truth-truth_mean)
        loss_cov = diff**2/2
        bloss = self.basic_loss(y_pred, y_truth)+eps
        aloss = loss_pearson.mean()+eps
        closs = loss_cov.mean()+eps
        if self.alpha > 1.0:
            factor = min([aloss, bloss, closs]).detach()
        elif self.alpha < 1.0:
            factor = max([aloss, bloss, closs]).detach()
        else:
            factor = 1.0
        aloss, bloss, closs = aloss/factor, bloss/factor, closs/factor
        loss = (aloss**(1/self.alpha) + bloss**(1/self.alpha) + closs**(1/self.alpha))/3
        if self.version == 'GAR-EXP':
          loss = factor*(loss**self.alpha)
        else:
          loss = loss.log()*self.alpha
        return loss  


#--------------------Losses for other baselines-------------------------
def rank(seq):
    return torch.argsort(torch.argsort(seq).flip(1))


def rank_normalised(seq):
    return (rank(seq) + 1).float() / seq.size()[1]


class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        rank = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    def backward(ctx, grad_output):
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None
    
def batchwise_ranking_regularizer(features, targets, lambda_val):
    loss = 0

    # Reduce ties and boost relative representation of infrequent labels by computing the 
    # regularizer over a subset of the batch in which each label appears at most once
    batch_unique_targets = torch.unique(targets)
    if len(batch_unique_targets) < len(targets):
        sampled_indices = []
        for target in batch_unique_targets:
            sampled_indices.append(random.choice((targets == target).nonzero()[:,0]).item())
        x = features[sampled_indices]
        y = targets[sampled_indices]
    else:
        x = features
        y = targets

    # Compute feature similarities
    xxt = torch.matmul(F.normalize(x.view(x.size(0),-1)), F.normalize(x.view(x.size(0),-1)).permute(1,0))

    # Compute ranking loss
    for i in range(len(y)):
        label_ranks = rank_normalised(-torch.abs(y[i] - y).transpose(0,1))
        feature_ranks = TrueRanker.apply(xxt[i].unsqueeze(dim=0), lambda_val)
        loss += F.mse_loss(feature_ranks, label_ranks)

    return loss

def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mae_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]
        if len(features.shape) > 2:
          features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
          labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss


def ConR_extend(features, targets, preds, w=1, weights=1, t=0.07, e=0.01):
    q = torch.nn.functional.normalize(features, dim=1)
    k = torch.nn.functional.normalize(features, dim=1)

    l_k = targets[:, None, :]  # Nx1xD
    l_q = targets[None, :, :]  # 1xNxD

    p_k = preds[:, None, :]
    p_q = preds[None, :, :]

    l_dist = torch.abs(l_q - l_k).mean(dim=-1)
    p_dist = torch.abs(p_q - p_k).mean(dim=-1)


    #w = targets.std().detach()
    pos_i = l_dist.le(w)
    neg_i = ((~ (l_dist.le(w))) * (p_dist.le(w)))

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0

    prod = torch.einsum("nc,kc->nk", [q, k]) / t
    pos = prod * pos_i
    neg = prod * neg_i

    pushing_w = weights * torch.exp(l_dist * e)
    neg_exp_dot = (pushing_w * (torch.exp(neg)) * neg_i).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()
    no_pos_flag = (pos_i).sum(1).bool()

    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom = torch.clip(pos_i.sum(1), min=1e-5)
    #print(denom)

    loss = ((-torch.log(
        torch.div(torch.exp(pos), (torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1))) * (
                 pos_i)).sum(1) / denom)

    loss = (weights * (loss * no_neg_flag).unsqueeze(-1)).mean()

    return loss

def ConR(features, targets, preds, w=1, weights=1, t=0.07, e=0.01):
    q = torch.nn.functional.normalize(features, dim=1)
    k = torch.nn.functional.normalize(features, dim=1)

    l_k = targets.flatten()[None, :]
    l_q = targets

    p_k = preds.flatten()[None, :]
    p_q = preds

    l_dist = torch.abs(l_q - l_k)
    p_dist = torch.abs(p_q - p_k)


    #w = targets.std().detach()
    pos_i = l_dist.le(w)
    neg_i = ((~ (l_dist.le(w))) * (p_dist.le(w)))

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0

    prod = torch.einsum("nc,kc->nk", [q, k]) / t
    pos = prod * pos_i
    neg = prod * neg_i

    pushing_w = weights * torch.exp(l_dist * e)
    neg_exp_dot = (pushing_w * (torch.exp(neg)) * neg_i).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()
    no_pos_flag = (pos_i).sum(1).bool()

    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom = torch.clip(pos_i.sum(1), min=1e-5)
    #print(denom)

    loss = ((-torch.log(
        torch.div(torch.exp(pos), (torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1))) * (
                 pos_i)).sum(1) / denom)

    loss = (weights * (loss * no_neg_flag).unsqueeze(-1)).mean()

    return loss


