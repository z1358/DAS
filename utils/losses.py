import torch
import torch.nn as nn
import torch.nn.functional as F

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1-self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


class KDFromTemplate(nn.Module):
    def __init__(self, template, temperature=0.07, alpha=1.0):
        super().__init__()

        self.temperature = temperature
        self.template = template
        self.alpha = alpha

        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

        normalized_template = torch.nn.functional.normalize(self.template, dim=1)
        self.similarity_map = normalized_template @ normalized_template.T

    def forward(self, features, test_template, labels=None):
        """
        features : batch x dim
        test_template : class_num x dim 
        """

        normalized_test_template = torch.nn.functional.normalize(test_template, dim=1)
        prob = torch.nn.functional.normalize(features, dim=1) @ normalized_test_template.T

        train_logit = self.similarity_map[labels,:]

        loss = self.loss_fn(F.log_softmax(prob / self.temperature, dim=1),
                            F.softmax(train_logit / self.temperature, dim=1))
        loss = loss * self.alpha

        return loss
    
class AugCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(AugCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_aug, x_ema):
        return -(1-self.alpha) * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
                  - self.alpha * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)


class SoftLikelihoodRatio(nn.Module):
    def __init__(self, clip=0.99, eps=1e-5):
        super(SoftLikelihoodRatio, self).__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits):
        probs = logits.softmax(1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return - (probs * torch.log((probs / (torch.ones_like(probs) - probs)) + self.eps)).sum(1)


class GeneralizedCrossEntropy(nn.Module):
    """ Paper: https://arxiv.org/abs/1805.07836 """
    def __init__(self, q=0.8):
        super(GeneralizedCrossEntropy, self).__init__()
        self.q = q

    def __call__(self, logits, targets=None):
        probs = logits.softmax(1)
        if targets is None:
            targets = probs.argmax(dim=1)
        probs_with_correct_idx = probs.index_select(-1, targets).diag()
        return (1.0 - probs_with_correct_idx ** self.q) / self.q

class OPELoss(nn.Module):
    def __init__(self, class_per_task, temperature=0.5, only_old_proto=False):
        super(OPELoss, self).__init__()
        self.class_per_task = class_per_task
        self.temperature = temperature
        self.only_old_proto = only_old_proto

    def cal_prototype(self, z1, z2, y):
        if z2 is None:
            z2 = z1
        start_i = 0
        end_i = self.class_per_task
        dim = z1.shape[1]
        current_classes_mean_z1 = torch.zeros((end_i, dim), device=z1.device)
        current_classes_mean_z2 = torch.zeros((end_i, dim), device=z1.device)
        for i in range(start_i, end_i):
            indices = (y == i)
            if not any(indices):
                continue
            t_z1 = z1[indices]
            t_z2 = z2[indices]

            mean_z1 = torch.mean(t_z1, dim=0)
            mean_z2 = torch.mean(t_z2, dim=0)

            current_classes_mean_z1[i] = mean_z1
            current_classes_mean_z2[i] = mean_z2

        return current_classes_mean_z1, current_classes_mean_z2

    def forward(self, z1, z2, labels, is_new=False, source_protos=None):
        prototype_z1, prototype_z2 = self.cal_prototype(z1, z2, labels)
        if source_protos is not None:
            prototype_z2 = source_protos.squeeze(1)
        if not self.only_old_proto or is_new:
            nonZeroRows = torch.abs(prototype_z1).sum(dim=1) > 0
            nonZero_prototype_z1 = prototype_z1[nonZeroRows]
            nonZero_prototype_z2 = prototype_z2[nonZeroRows]

        nonZero_prototype_z1 = F.normalize(nonZero_prototype_z1)
        nonZero_prototype_z2 = F.normalize(nonZero_prototype_z2)

        device = nonZero_prototype_z1.device

        class_num = nonZero_prototype_z1.size(0)
        z = torch.cat((nonZero_prototype_z1, nonZero_prototype_z2), dim=0)

        logits = torch.einsum("if, jf -> ij", z, z) / self.temperature
        # logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        # logits = logits - logits_max.detach()

        pos_mask = torch.zeros((2 * class_num, 2 * class_num), dtype=torch.bool, device=device)
        pos_mask[:, class_num:].fill_diagonal_(True)
        pos_mask[class_num:, :].fill_diagonal_(True)

        logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logit_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        loss = -mean_log_prob_pos.mean()

        return loss, prototype_z1, prototype_z2