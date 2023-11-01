
from __future__ import print_function

import torch
import torch.nn as nn
from memory_bank import MemoryBankModule

class SupConLoss0(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss0, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask # copy mask and expend
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases (mid ones)
        logits_self_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_self_mask # remove mid ones from mask

        # compute log_prob, use exp to range negative logits to [0,1]
        exp_logits = torch.exp(logits) * logits_self_mask # not add all postive pairs

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss_1 = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss_1.view(anchor_count, batch_size).mean()

        return loss

class SupConLoss1(nn.Module):
    # debug new feature
    # add neutral label (unknown labels)
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss1, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            # add 0.5 to unknown labels
            unknown_labels = (torch.bitwise_and(torch.le(labels, 0), torch.le(labels, 0).T).float()).to(device)
            center = torch.eye(batch_size, dtype=torch.float32).to(device)
            unknown_labels = ((unknown_labels - torch.bitwise_and((unknown_labels).int(),center.int()).to(device))*0.5) # remove center ones
            mask_known = torch.eq(labels, labels.T).float().to(device)
            mask = mask_known + unknown_labels
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask # copy mask and expend
        try:#TODO: fix bug
            mask_known = mask_known.repeat(anchor_count, contrast_count)
            mask_unknown = unknown_labels.repeat(anchor_count, contrast_count)

        except:
            print("no mask known")
        # mask-out self-contrast cases (mid ones)
        # loss 1
        mask = mask.repeat(anchor_count, contrast_count)
        logits_self_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        try:
            logits_mask = torch.square(mask_known - 1) - mask_unknown
        except:
            logits_mask = torch.square(mask - 1)
        logits_mask_org = logits_self_mask
        # loss 2 try not remove mid ones in mask
        mask = mask * logits_self_mask

        # compute log_prob, use exp to range negative logits to [0,1]
        # loss 1
        exp_logits = torch.exp(logits) * logits_mask  # remove all positive pairs

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # torch.maximum(torch.zeros_like(a), a)
        neg_relu_prob = torch.minimum(torch.zeros_like(log_prob), log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * neg_relu_prob).sum(1) / mask.sum(1)

        # TRIPLET LOSS ADVANCE
        # tmp two channel mask
        sim_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        sim_mask = sim_mask.repeat(anchor_count, contrast_count)
        sim_mask = sim_mask * logits_self_mask
        trip_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_neg = torch.exp(logits) * trip_mask
        log_self_diff = logits - torch.log(logits_neg.sum(1, keepdim=True))
        triplet_loss = (sim_mask * log_self_diff).sum(1) / sim_mask.sum(1)
        # TODO: update loss to paper loss
        # loss
        loss = - (self.temperature / self.base_temperature) * (triplet_loss + mean_log_prob_pos) / 2
        loss = loss.view(anchor_count, batch_size).mean()
        self_loss = - (self.temperature / self.base_temperature) * triplet_loss
        class_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss, self_loss.view(anchor_count, batch_size).mean(), class_loss.view(anchor_count, batch_size).mean()


class SupConTupletLoss(nn.Module):
    # remove the part positive samples from negative calculation
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConTupletLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
          if features.is_cuda
          else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask # copy mask and expend
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases (mid ones)
        # loss 1
        logits_self_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_mask = torch.square(mask - 1)

        logits_mask_org = logits_self_mask
        # loss 2 try not remove mid ones in mask
        mask = mask * logits_self_mask

        # compute log_prob, use exp to range negative logits to [0,1]
        # loss 1
        exp_logits = torch.exp(logits) * logits_mask  # remove all positive pairs

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # TRIPLET LOSS ADVANCE
        # tmp two channel mask
        sim_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        sim_mask = sim_mask.repeat(anchor_count, contrast_count)
        sim_mask = sim_mask * logits_self_mask
        trip_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_neg = torch.exp(logits) * trip_mask
        log_self_diff = logits - torch.log(logits_neg.sum(1, keepdim=True))
        triplet_loss = (sim_mask * log_self_diff).sum(1) / sim_mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * (triplet_loss + mean_log_prob_pos) / 2
        loss = loss.view(anchor_count, batch_size).mean()
        self_loss = - (self.temperature / self.base_temperature) * triplet_loss
        class_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss, self_loss.view(anchor_count, batch_size).mean(), class_loss.view(anchor_count, batch_size).mean()


class SupConTupletLoss2(nn.Module):
    # remove the part positive samples from negative calculation
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, self_weight=0.7):
        super(SupConTupletLoss2, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.self_weight = self_weight
    def forward(self, features, labels=None, mask=None, weight=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if weight is None:
            weight = self.self_weight

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        def print_grad(grad):
            print("Features gradient:", grad.abs().sum().item())

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask # copy mask and expend
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases (mid ones)
        # loss 1
        logits_self_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_mask = torch.square(mask - 1)

        logits_mask_org = logits_self_mask
        # loss 2 try not remove mid ones in mask
        mask = mask * logits_self_mask

        # compute log_prob, use exp to range negative logits to [0,1]
        # loss 1
        exp_logits = torch.exp(logits) * logits_mask  # remove all positive pairs

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # TRIPLET LOSS ADVANCE
        # tmp two channel mask
        sim_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        sim_mask = sim_mask.repeat(anchor_count, contrast_count)
        sim_mask = sim_mask * logits_self_mask
        trip_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_neg = torch.exp(logits) * trip_mask
        log_self_diff = logits - torch.log(logits_neg.sum(1, keepdim=True))
        triplet_loss = (sim_mask * log_self_diff).sum(1) / sim_mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * ((1-weight) * triplet_loss + weight * mean_log_prob_pos)
        loss = loss.view(anchor_count, batch_size).mean()
        self_loss = - (self.temperature / self.base_temperature) * triplet_loss
        class_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss, self_loss.view(anchor_count, batch_size).mean(), class_loss.view(anchor_count, batch_size).mean()


class SupConTupletLoss3(nn.Module):
    # neg relu
    # re-orgnize the loss function and run
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, self_weight=0.7):
        super(SupConTupletLoss3, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.self_weight = self_weight

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask # copy mask and expend
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases (mid ones)
        # loss 1
        logits_self_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_mask = torch.square(mask - 1)

        logits_mask_org = logits_self_mask
        # loss 2 try not remove mid ones in mask
        mask = mask * logits_self_mask

        # compute log_prob, use exp to range negative logits to [0,1]
        # loss 1
        exp_logits = torch.exp(logits) * logits_mask  # remove all positive pairs

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # torch.maximum(torch.zeros_like(a), a)
        neg_relu_prob = torch.minimum(torch.zeros_like(log_prob),log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * neg_relu_prob).sum(1) / mask.sum(1)

        # TRIPLET LOSS ADVANCE
        # tmp two channel mask
        sim_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        sim_mask = sim_mask.repeat(anchor_count, contrast_count)
        sim_mask = sim_mask * logits_self_mask
        trip_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_neg = torch.exp(logits) * trip_mask
        log_self_diff = logits - torch.log(logits_neg.sum(1, keepdim=True))
        triplet_loss = (sim_mask * log_self_diff).sum(1) / sim_mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * ((1-self.self_weight) * triplet_loss + self.self_weight * mean_log_prob_pos)
        loss = loss.view(anchor_count, batch_size).mean()
        self_loss = - (self.temperature / self.base_temperature) * triplet_loss
        class_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss, self_loss.view(anchor_count, batch_size).mean(), class_loss.view(anchor_count, batch_size).mean()

class SupConTupletLoss3(nn.Module):
    # neg relu
    # re-orgnize the loss function and run
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, self_weight=0.7):
        super(SupConTupletLoss3, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.self_weight = self_weight

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask # copy mask and expend
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases (mid ones)
        # loss 1
        logits_self_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_mask = torch.square(mask - 1)

        logits_mask_org = logits_self_mask
        # loss 2 try not remove mid ones in mask
        mask = mask * logits_self_mask

        # compute log_prob, use exp to range negative logits to [0,1]
        # loss 1
        exp_logits = torch.exp(logits) * logits_mask  # remove all positive pairs

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # torch.maximum(torch.zeros_like(a), a)
        neg_relu_prob = torch.minimum(torch.zeros_like(log_prob),log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * neg_relu_prob).sum(1) / mask.sum(1)

        # TRIPLET LOSS ADVANCE
        # tmp two channel mask
        sim_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        sim_mask = sim_mask.repeat(anchor_count, contrast_count)
        sim_mask = sim_mask * logits_self_mask
        trip_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_neg = torch.exp(logits) * trip_mask
        log_self_diff = logits - torch.log(logits_neg.sum(1, keepdim=True))
        triplet_loss = (sim_mask * log_self_diff).sum(1) / sim_mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * ((1-self.self_weight) * triplet_loss + self.self_weight * mean_log_prob_pos)
        loss = loss.view(anchor_count, batch_size).mean()
        self_loss = - (self.temperature / self.base_temperature) * triplet_loss
        class_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss, self_loss.view(anchor_count, batch_size).mean(), class_loss.view(anchor_count, batch_size).mean()



class SupConTupletLossSplit(nn.Module):
    # TODO use one view as self contrast, another view as label contrast
    # re-orgnize the loss function and run
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, self_weight=0.5):
        super(SupConTupletLossSplit, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.self_weight = self_weight

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = 2
        label_contrast_feature = torch.cat(torch.unbind(features, dim=1)[:2], dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            label_anchor_feature = label_contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(label_anchor_feature, label_contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        label_logits = anchor_dot_contrast - logits_max.detach()

        # tile mask # copy mask and expend
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases (mid ones)
        # loss 1
        logits_self_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_mask = torch.square(mask - 1)

        logits_mask_org = logits_self_mask
        # loss 2 try not remove mid ones in mask
        mask = mask * logits_self_mask

        # compute log_prob, use exp to range negative logits to [0,1]
        # loss 1
        exp_logits = torch.exp(label_logits) * logits_mask  # remove all positive pairs

        log_prob = label_logits - torch.log(exp_logits.sum(1, keepdim=True))
        # torch.maximum(torch.zeros_like(a), a)
        neg_relu_prob = torch.minimum(torch.zeros_like(log_prob),log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * neg_relu_prob).sum(1) / mask.sum(1)

        # TRIPLET LOSS ADVANCE
        # tmp two channel mask
        sim_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        sim_mask = sim_mask.repeat(anchor_count, contrast_count)
        sim_mask = sim_mask * logits_self_mask
        trip_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        self_contrast_feature = torch.cat(torch.unbind(features, dim=1)[:2], dim=0)
        self_anchor_feature = self_contrast_feature
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(self_anchor_feature, self_contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        self_logits = anchor_dot_contrast - logits_max.detach()

        logits_neg = torch.exp(self_logits) * trip_mask
        log_self_diff = self_logits - torch.log(logits_neg.sum(1, keepdim=True))
        triplet_loss = (sim_mask * log_self_diff).sum(1) / sim_mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * ((1-self.self_weight) * triplet_loss + self.self_weight * mean_log_prob_pos)
        loss = loss.view(anchor_count, batch_size).mean()
        self_loss = - (self.temperature / self.base_temperature) * triplet_loss
        class_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss, self_loss.view(anchor_count, batch_size).mean(), class_loss.view(anchor_count, batch_size).mean()

class SupConLossWithQueue(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07,
                 self_weight=0.5, queue_size=8192, momentum=0.99):
        super(SupConLossWithQueue, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.self_weight = self_weight
        self.queue_size = queue_size
        self.momentum = momentum

        # Create similar feature vectors for the queue
        # base_vector = torch.rand(128)
        # std_dev = 0.01
        # perturbations = torch.randn(queue_size, 128) * std_dev
        # similar_features = base_vector + perturbations
        # similar_features = torch.nn.functional.normalize(similar_features, dim=1)

        # Initialize the feature queue
        self.register_buffer("feature_queue", torch.randn(queue_size, 128))
        self.feature_queue = nn.functional.normalize(self.feature_queue, dim=1)
        self.register_buffer("feature_queue_ptr", torch.zeros(1, dtype=torch.long))

        # perturbations2 = torch.randn(queue_size, 128) * std_dev
        # similar_features2 = base_vector + perturbations2
        # similar_features2 = torch.nn.functional.normalize(similar_features2, dim=1)

        self.register_buffer("feature_queue2", torch.randn(queue_size, 128))#torch.randn(queue_size, 128))
        self.feature_queue2 = nn.functional.normalize(self.feature_queue2, dim=1)
        self.register_buffer("feature_queue_ptr2", torch.zeros(1, dtype=torch.long))

        # Initialize the label queue
        self.register_buffer("label_queue", torch.arange(start=1e5, end=1e5+queue_size, dtype=torch.long))
        self.label_queue_ptr = 0

    def synchronize_queues(self):
        with torch.no_grad():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # Synchronize feature_queue
                torch.distributed.all_reduce(self.feature_queue, op=torch.distributed.ReduceOp.SUM)
                self.feature_queue /= torch.distributed.get_world_size()

                torch.distributed.all_reduce(self.feature_queue2, op=torch.distributed.ReduceOp.SUM)
                self.feature_queue2 /= torch.distributed.get_world_size()
                # Synchronize label_queue; since it's long dtype, first cast to float
                label_queue_float = self.label_queue.to(dtype=torch.float32)
                torch.distributed.all_reduce(label_queue_float, op=torch.distributed.ReduceOp.SUM)
                self.label_queue = (label_queue_float / torch.distributed.get_world_size()).to(dtype=torch.long)

    def forward(self, features, label=None, mask=None):
        assert features.requires_grad, "Features tensor does not require gradients."


        # Let's use the first part of the reshaped features for our computations
        features_reshaped = features.view(-1, features.shape[-1])
        feature1, feature2 = features_reshaped.chunk(2, 0)

        # Normalize the features
        feature1 = nn.functional.normalize(feature1, dim=1)
        feature2 = nn.functional.normalize(feature2, dim=1)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = self.queue_size + label.shape[0]
        if self.label_queue is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif self.label_queue is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif self.label_queue is not None:
            labels = torch.cat([self.label_queue.contiguous().view(-1, 1), label.contiguous().view(-1, 1)], dim=0)

            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        batch1 = torch.cat([self.feature_queue, feature1], dim=0)
        batch2 = torch.cat([self.feature_queue2, feature2], dim=0)
        contrast_feature = torch.cat([batch1, batch2], dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = feature1
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask # copy mask and expend
        mask = mask.repeat(anchor_count, contrast_count)

        logits_self_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        logits_mask = torch.square(mask - 1)

        mask = mask * logits_self_mask

        # compute log_prob, use exp to range negative logits to [0,1]
        # loss 1
        exp_logits = torch.exp(logits) * logits_mask  # remove all positive pairs

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # torch.maximum(torch.zeros_like(a), a)
        #neg_relu_prob = torch.minimum(torch.zeros_like(log_prob), log_prob)
        # compute mean of log-likelihood over positive
        #mean_log_prob_pos = (mask * neg_relu_prob).sum(1) / mask.sum(1)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        '''
        # TRIPLET LOSS ADVANCE
        # tmp two channel mask
        sim_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        sim_mask = sim_mask.repeat(anchor_count, contrast_count)
        sim_mask = sim_mask * logits_self_mask
        trip_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_neg = torch.exp(logits) * trip_mask
        log_self_diff = logits - torch.log(logits_neg.sum(1, keepdim=True))
        triplet_loss = (sim_mask * log_self_diff).sum(1) / sim_mask.sum(1)
        
        # loss
        loss = - (self.temperature / self.base_temperature) * (
                    (1 - self.self_weight) * triplet_loss + self.self_weight * mean_log_prob_pos)
        loss = loss.view(anchor_count, batch_size).mean()
        self_loss = - (self.temperature / self.base_temperature) * triplet_loss'''
        class_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        with torch.no_grad():
            batch_size_current = feature1.size(0)
            # Determine positions in the queue for the new features
            ptr = self.feature_queue_ptr
            end_ptr = ptr + batch_size_current

            # If end_ptr exceeds queue_size, wrap around
            if end_ptr > self.queue_size:
                self.feature_queue[ptr:] = feature2[:self.queue_size - ptr]
                self.feature_queue[:end_ptr % self.queue_size] = feature2[self.queue_size - ptr:]
                self.feature_queue2[ptr:] = feature1[:self.queue_size - ptr]
                self.feature_queue2[:end_ptr % self.queue_size] = feature1[self.queue_size - ptr:]
                self.label_queue[ptr:] = label[:self.queue_size - ptr]
                self.label_queue[:end_ptr % self.queue_size] = label[self.queue_size - ptr:]
            else:
                self.feature_queue[ptr:end_ptr] = feature2
                self.feature_queue2[ptr:end_ptr] = feature1
                self.label_queue[ptr:end_ptr] = label

            # Update the queue pointer
            self.feature_queue_ptr = (self.feature_queue_ptr + batch_size_current) % self.queue_size

        assert logits.requires_grad, "logits does not require gradients."
        assert log_prob.requires_grad, "log_prob does not require gradients."
        assert class_loss.requires_grad, "Loss tensor does not require gradients."
        self.synchronize_queues()

        return class_loss.mean(), class_loss.mean(), class_loss.mean()


class SupConLossWithMemoryBank(nn.Module):
    def __init__(self, temperature=0.07, bank_size=8192, feature_dim=128, base_temperature=0.07, self_weight=0.5):
        super(SupConLossWithMemoryBank, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.self_weight = self_weight
        self.queue_size = bank_size

        # Initialize three memory banks: for feature_1, feature_2, and labels
        self.memory_bank_feature_1 = MemoryBankModule(size=bank_size)
        self.memory_bank_feature_2 = MemoryBankModule(size=bank_size)
        self.memory_bank_labels = MemoryBankModule(size=bank_size)

    def forward(self, features, label=None, mask=None, weight=0.7):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        features_reshaped = features.view(-1, features.shape[-1])
        feature1, feature2 = features_reshaped.chunk(2, 0)

        # Update and retrieve from memory banks
        feature1, stored_feature_1 = self.memory_bank_feature_1.forward(feature1, update=True)
        feature2, stored_feature_2 = self.memory_bank_feature_2.forward(feature2, update=True)
        labels, stored_labels = self.memory_bank_labels.forward(label.unsqueeze(1), update=True)

        batch_size = self.queue_size + label.shape[0]
        labels = torch.cat([stored_labels.contiguous().view(-1, 1), labels.contiguous().view(-1, 1)], dim=0)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compare feature1 with stored_feature_2 and feature2 with stored_feature_1
        combined_feature_1 = torch.cat([stored_feature_2.T, feature1], dim=0)
        combined_feature_2 = torch.cat([stored_feature_1.T, feature2], dim=0)

        def calculate_loss1(contrast_feature, mask):
            contrast_count = features.shape[1]
            anchor_feature = contrast_feature
            anchor_count = contrast_count

            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)

            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # tile mask
            mask = mask.repeat(anchor_count, contrast_count)
            logits_self_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                0
            )
            logits_mask = torch.square(mask - 1)
            mask = mask * logits_self_mask
            exp_logits = torch.exp(logits) * logits_mask  # remove all positive pairs
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            class_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            return class_loss

        # # TRIPLET LOSS ADVANCE
        # sim_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        # sim_mask = sim_mask.repeat(anchor_count, contrast_count)
        # sim_mask = sim_mask * logits_self_mask
        # trip_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        #     0
        # )
        # logits_neg = torch.exp(logits) * trip_mask
        # log_self_diff = logits - torch.log(logits_neg.sum(1, keepdim=True))
        # triplet_loss = (sim_mask * log_self_diff).sum(1) / sim_mask.sum(1)
        #
        # # loss
        # loss = - (self.temperature / self.base_temperature) * (
        #             (1 - self.self_weight) * triplet_loss + self.self_weight * mean_log_prob_pos)
        # loss = loss.view(anchor_count, batch_size).mean()
        # self_loss = - (self.temperature / self.base_temperature) * triplet_loss
        # class_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        class_loss = (calculate_loss1(combined_feature_1, mask) + calculate_loss1(combined_feature_2, mask))/2
        return (class_loss.view(2, batch_size).mean(), class_loss.view(2, batch_size).mean(),
                class_loss.view(2, batch_size).mean())
