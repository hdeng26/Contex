"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn

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

        # Initialize the feature queue
        self.register_buffer("feature_queue", torch.randn(queue_size, 128))
        self.feature_queue = nn.functional.normalize(self.feature_queue, dim=0)
        self.feature_queue_ptr = 0

        self.register_buffer("feature_queue2", torch.randn(queue_size, 128))
        self.feature_queue2 = nn.functional.normalize(self.feature_queue2, dim=0)
        self.feature_queue2_ptr = 0

        # Initialize the label queue
        self.register_buffer("label_queue", torch.empty(queue_size, dtype=torch.long))
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

        # Update feature and label queues
        with torch.no_grad():
            batch_size_current = feature1.size(0)
            # TODO: label need to be qs + feature label
            if self.feature_queue_ptr + batch_size_current <= self.queue_size:
                self.feature_queue[self.feature_queue_ptr:self.feature_queue_ptr + batch_size_current] = feature2
                self.label_queue[self.feature_queue_ptr:self.feature_queue_ptr + batch_size_current] = label
                self.feature_queue_ptr += batch_size_current
            else:
                remaining_space = self.queue_size - self.feature_queue_ptr
                self.feature_queue[self.feature_queue_ptr:self.feature_queue_ptr + remaining_space] = feature2[:remaining_space]
                self.feature_queue[:batch_size_current - remaining_space] = feature2[remaining_space:]
                self.label_queue[self.feature_queue_ptr:self.feature_queue_ptr + remaining_space] = label[:remaining_space]
                self.label_queue[:batch_size_current - remaining_space] = label[remaining_space:]
                self.feature_queue_ptr = (self.feature_queue_ptr + batch_size_current) % self.queue_size

            if self.feature_queue2_ptr + batch_size_current <= self.queue_size:
                self.feature_queue2[
                self.feature_queue2_ptr:self.feature_queue2_ptr + batch_size_current] = feature1
                self.feature_queue2_ptr += batch_size_current
            else:
                remaining_space = self.queue_size - self.feature_queue2_ptr
                self.feature_queue2[
                self.feature_queue2_ptr:self.feature_queue2_ptr + remaining_space] = feature1[
                                                                                     :remaining_space]
                self.feature_queue2[:batch_size_current - remaining_space] = feature1[remaining_space:]
                self.feature_queue2_ptr = (self.feature_queue2_ptr + batch_size_current) % self.queue_size

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
        # mask-out self-contrast cases (mid ones)
        # loss 1
        pos_1d_label = torch.cat((torch.ones(self.queue_size - label.shape[0]), torch.zeros(label.shape[0]*2)))
        pos_1d_label_src = pos_1d_label.repeat(anchor_count)
        logits_self_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            pos_1d_label_src.view(-1, 1).to(device),
        )

        logits_mask = torch.square(mask - 1)

        mask = mask * logits_self_mask

        #TODO: still not convergence
        # remove not usefull columns
        mask = torch.cat((mask[:0], mask[self.queue_size - label.shape[0] + 1:]), dim=0)
        mask = torch.cat((mask[:batch_size], mask[batch_size + self.queue_size - label.shape[0] + 1:]), dim=0)

        logits = torch.cat((logits[:0], logits[self.queue_size - label.shape[0] + 1:]), dim=0)
        logits = torch.cat((logits[:batch_size], logits[batch_size + self.queue_size - label.shape[0] + 1:]), dim=0)

        logits_mask = torch.cat((logits_mask[:0], logits_mask[self.queue_size - label.shape[0] + 1:]), dim=0)
        logits_mask = torch.cat((logits_mask[:batch_size], logits_mask[batch_size + self.queue_size - label.shape[0] + 1:]), dim=0)

        # compute log_prob, use exp to range negative logits to [0,1]
        # loss 1
        exp_logits = torch.exp(logits) * logits_mask  # remove all positive pairs

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # torch.maximum(torch.zeros_like(a), a)
        neg_relu_prob = torch.minimum(torch.zeros_like(log_prob), log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * neg_relu_prob).sum(1) / mask.sum(1)
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

        assert anchor_dot_contrast.requires_grad, "anchor_dot_contrast does not require gradients."
        #assert logits.requires_grad, "logits does not require gradients."
        #assert log_prob.requires_grad, "log_prob does not require gradients."
        #assert loss.requires_grad, "Loss tensor does not require gradients."

        self.synchronize_queues()
        return class_loss.mean(), class_loss.mean(), class_loss.mean()

class SupConLossWithQueue2(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, queue_size=65536, head_dim=128, views=2):
        super(SupConLossWithQueue2, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.queue_size = queue_size

        if head_dim != 128:
            raise NotImplementedError(
                'head not supported: {}'.format(head_dim))

        if views != 2:
            raise NotImplementedError(
                'views not supported yet: {}'.format(views))
        # buffer queues
        self.register_buffer("queue", torch.randn(head_dim, queue_size))
        #self.register_buffer("queue_2", torch.randn(head_dim, queue_size))
        self.register_buffer("queue_l", torch.arange(-queue_size, 0).T)
        self.queue_1 = nn.functional.normalize(self.queue, dim=0)
        #self.queue_2 = nn.functional.normalize(self.queue_2, dim=0)

    def enqueue_dequeue(self, features, labels):
        # Enqueue new features and dequeue oldest
        feature_1 = torch.unbind(features, dim=1)[0]
        feature_2 = torch.unbind(features, dim=1)[1]
        self.queue_1 = torch.cat([feature_1, self.queue[:-features.size(0)]], dim=0)
        #self.queue_2 = torch.cat([feature_2, self.queue[:-features.size(0)]], dim=0)
        self.queue_label = torch.cat([labels, self.queue[:-labels.size(0)]], dim=0)

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
        # Enqueue the current features
        self.enqueue_dequeue(features, labels)
        # TODO: add self.queue into contrast
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
            labels = self.queue_label.contiguous().view(-1, 1)
            if labels.shape[0] != self.queue_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]

        # add memory tricks
        self.enqueue_dequeue(features, labels)

        #contrast_feature = torch.cat([self.queue, torch.unbind(features, dim=1)[1]], dim=0)
        '''
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        '''
        # compute logits
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(
            torch.matmul(torch.unbind(features, dim=1)[1], self.queue.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask # copy mask and expend
        #mask = mask.repeat(anchor_count, contrast_count)
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

