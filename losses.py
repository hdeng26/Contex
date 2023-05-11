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
        logits_mask = torch.square(mask - 1)

        mask = mask * logits_self_mask # remove mid ones from mask

        # compute log_prob, use exp to range negative logits to [0,1]
        exp_logits = torch.exp(logits) * logits_self_mask # not add all postive pairs

        # logit - log(exp(logit).sum) it is MSE in MLE???
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss_1 = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss_1.view(anchor_count, batch_size).mean()

        return loss

class SupConLoss1(nn.Module):
    # remove the part positive samples from negative calculation
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
        # TODO: support DDP calculation
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # remove self

        # tile mask # copy mask and expend
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases (mid ones) negative samples
        logits_self_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_mask = torch.square(mask - 1) # positive to 0 negative to 1

        mask = mask * logits_self_mask # remove mid ones from mask

        # compute log_prob, use exp to range negative logits to [0,1]
        exp_logits = torch.exp(logits) * logits_mask # not add all postive pairs

        # logit - log(exp(logit).sum)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        #################################use relu to prevent negative loss
        relu = torch.nn.ReLU()
        neg_relu_prob = -relu(-log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * neg_relu_prob).sum(1) / mask.sum(1)


        # compute mean of log-likelihood over positive
        #mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss_1 = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss_1.view(anchor_count, batch_size).mean()

        return loss

class SupConLoss2(nn.Module):
    # update by removing the negative relu (not stable in loss -1)
        # remove the part positive samples from negative calculation
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
        # TODO: support DDP calculation
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # remove self

        # tile mask # copy mask and expend
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases (mid ones) negative samples
        logits_self_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_mask = torch.square(mask - 1) # positive to 0 negative to 1

        mask = mask * logits_self_mask # remove mid ones from mask
        '''
        # compute log_prob, use exp to range negative logits to [0,1]
        exp_logits = torch.exp(logits) * logits_mask # not add all postive pairs

        # logit - log(exp(logit).sum)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        #################################use relu to prevent negative loss
        relu = torch.nn.ReLU()
        neg_relu_prob = -relu(-log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * neg_relu_prob).sum(1) / mask.sum(1)


        # compute mean of log-likelihood over positive
        #mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        '''
        # loss 1
        exp_logits = torch.exp(logits) * logits_mask # add all positive pairs

        log_prob = torch.log(1+(exp_logits.sum(1, keepdim=True)).expand_as(exp_logits)-torch.exp(logits))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss_1 = (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss_1.view(anchor_count, batch_size).mean()

        return loss

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
        exp_logits = torch.exp(logits) * logits_mask # add all positive pairs

        # exp_logits = torch.exp(logits) * logits_self_mask  # not add positive pairs

        # logit - log(exp(logit).sum) it is MSE in MLE???
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        relu = torch.nn.ReLU()
        neg_relu_prob = -relu(-log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * neg_relu_prob).sum(1) / mask.sum(1)

        # TRIPLET LOSS ADVANCE
        #tmp two channel mask
        sim_mask = torch.scatter(torch.zeros_like(mask),
                                 0,
                                 torch.arange(batch_size,batch_size*2).view(1,-1).to(device)
                                 , 1)
        sim_mask = sim_mask + sim_mask.T
        trip_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_pos = (logits*sim_mask).sum(1, keepdim=True)
        logits_neg = logits * trip_mask
        exp_logits_self_diff = torch.exp(logits_neg - logits_pos)
        triplet_loss = torch.log(1 + exp_logits_self_diff.sum(1, keepdim=False))

        # loss
        loss = (self.temperature / self.base_temperature) * (triplet_loss - mean_log_prob_pos)/2
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
class SupConTupletLoss2(nn.Module):
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

        mask = mask * logits_self_mask

        # compute log_prob, use exp to range negative logits to [0,1]
        # loss 1

        exp_logits = torch.exp(logits) * logits_mask # add all positive pairs

        log_prob = torch.log(1+(exp_logits.sum(1, keepdim=True)).expand_as(exp_logits)-torch.exp(logits))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # TRIPLET LOSS ADVANCE
        #tmp two channel mask
        sim_mask = torch.scatter(torch.zeros_like(mask),
                                 0,
                                 torch.arange(batch_size,batch_size*2).view(1,-1).to(device)
                                 , 1)
        sim_mask = sim_mask + sim_mask.T
        trip_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_pos = (logits*sim_mask).sum(1, keepdim=True)
        logits_neg = logits * trip_mask
        exp_logits_self_diff = torch.exp(logits_neg - logits_pos)
        triplet_loss = torch.log(1 + exp_logits_self_diff.sum(1, keepdim=False))

        # loss
        loss_1 = (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss_1.view(anchor_count, batch_size).mean()*0.6 + triplet_loss.mean()*0.4
        # don't know why loss 3 will not go down, but loss 2 will
        loss2 = torch.log(torch.exp(triplet_loss)*torch.exp(mean_log_prob_pos)).mean() * (self.temperature / self.base_temperature)
        loss3 = torch.log(torch.exp(triplet_loss+mean_log_prob_pos)).mean() * (self.temperature / self.base_temperature)
        loss4 = (triplet_loss + mean_log_prob_pos).mean() * (self.temperature / self.base_temperature)
        loss5 = torch.log((torch.exp(triplet_loss)*torch.exp(mean_log_prob_pos)).sum()).mean() * (self.temperature / self.base_temperature)
        return loss5
