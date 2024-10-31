import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy


class My_Loss(nn.Module):
    def __init__(self, args, mixup_active, num_aug_splits):
        super().__init__()
        self.num_classes = 101
        self.gamm = 1.0
        self.alph = 0.01

        if args.jsd:
            assert num_aug_splits > 1  # JSD only valid with aug splits set
            self.classify_loss_fun = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
        elif mixup_active:
            self.classify_loss_fun = SoftTargetCrossEntropy().cuda()
        elif args.smoothing:
            self.classify_loss_fun = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
        else:
            self.classify_loss_fun = nn.CrossEntropyLoss().cuda()

    def update_sim(self, hash_feature, sim_matrix, count_matrix, epoch, cls_out, target):
        B, D = hash_feature.shape
        x_flat = hash_feature.reshape(B, -1)
        x_normalized = F.normalize(x_flat, p=2, dim=1)
        similarity_matrix = torch.mm(x_normalized, x_normalized.t())

        # 计算预测类别
        predicted_classes = torch.argmax(cls_out, dim=1)

        # 找到分类正确的样本索引
        correct_indices = (predicted_classes == target).nonzero(as_tuple=True)[0]

        for i in range(len(correct_indices)):
            for j in range(i+1, len(correct_indices)):
                idx_i = correct_indices[i]
                idx_j = correct_indices[j]
                sim_matrix[target[idx_i], target[idx_j]] = similarity_matrix[idx_i, idx_j] + sim_matrix[target[idx_i], target[idx_j]]
                sim_matrix[target[idx_j], target[idx_i]] = sim_matrix[target[idx_i], target[idx_j]]
                count_matrix[target[idx_i], target[idx_j]] = count_matrix[target[idx_i], target[idx_j]] + 1
                count_matrix[target[idx_j], target[idx_i]] = count_matrix[target[idx_i], target[idx_j]]
        return sim_matrix, count_matrix

    def hash_loss(self, hash_out, target, sim_matrix):
        theta = torch.einsum('ij,jk->ik', hash_out, hash_out.t()) / 2
        one_hot = torch.nn.functional.one_hot(target, self.num_classes)
        one_hot = one_hot.float()
        Sim = (torch.einsum('ij,jk->ik', one_hot, one_hot.t()) > 0).float()

        for i in range(6):
             for j in range(6):
                     Sim[i, j] = sim_matrix[target[i], target[j]]

        pair_loss = (torch.log(1 + torch.exp(theta)) - Sim * theta)
        mask_positive = Sim == 1
        mask_negative = Sim != 1
        S1 = mask_positive.float().sum() - hash_out.shape[0]
        S0 = mask_negative.float().sum()
        if S0 == 0:
            S0 = 1
        if S1 == 0:
            S1 = 1
        S = S0 + S1

        pair_loss[mask_positive] = pair_loss[mask_positive] * (S / S1)
        pair_loss[mask_negative] = pair_loss[mask_negative] * (S / S0)

        # 调整对角线上的损失，将自身对比损失设置为0
        diag_matrix = torch.tensor(np.diag(torch.diag(pair_loss.detach()).cpu())).cuda()
        pair_loss = pair_loss - diag_matrix

        count = (hash_out.shape[0] * (hash_out.shape[0] - 1) / 2)

        return pair_loss.sum() / 2 / count

    def forward(self, hash_feature, hash_out, cls_out, target, sim_matrix_last, sim_matrix_now, count_matrix, epoch):
        cls_loss = self.classify_loss_fun(cls_out, target)
        hash_loss = self.hash_loss(hash_out, target, sim_matrix_last)
        sim_matrix_now, count_matrix = self.update_sim(hash_feature, sim_matrix_now, count_matrix, epoch, cls_out, target)
        loss = self.gamm * cls_loss + self.alph * hash_loss
        return sim_matrix_now, count_matrix, hash_loss, cls_loss, loss


class My_Loss_eval(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = 101
        self.gamm = 1.0
        self.alph = 0.01
        self.classify_loss_fun = nn.CrossEntropyLoss().cuda()

    def hash_loss(self, hash_out, target):
        theta = torch.einsum('ij,jk->ik', hash_out, hash_out.t()) / 2
        one_hot = torch.nn.functional.one_hot(target, self.num_classes)
        one_hot = one_hot.float()
        Sim = (torch.einsum('ij,jk->ik', one_hot, one_hot.t()) > 0).float()

        pair_loss = (torch.log(1 + torch.exp(theta)) - Sim * theta)
        mask_positive = Sim > 0
        mask_negative = Sim <= 0
        S1 = mask_positive.float().sum() - hash_out.shape[0]
        S0 = mask_negative.float().sum()
        if S0 == 0:
            S0 = 1
        if S1 == 0:
            S1 = 1
        S = S0 + S1
        pair_loss[mask_positive] = pair_loss[mask_positive] * (S / S1)
        pair_loss[mask_negative] = pair_loss[mask_negative] * (S / S0)

        # 调整对角线上的损失，将自身对比损失设置为0
        diag_matrix = torch.tensor(np.diag(torch.diag(pair_loss.detach()).cpu())).cuda()
        pair_loss = pair_loss - diag_matrix

        count = (hash_out.shape[0] * (hash_out.shape[0] - 1) / 2)

        return pair_loss.sum() / 2 / count


    def forward(self, hash_out, cls_out, target):
        cls_loss = self.classify_loss_fun(cls_out, target)
        hash_loss = self.hash_loss(hash_out, target)
        loss = self.gamm * cls_loss + self.alph * hash_loss
        return hash_loss, cls_loss, loss
