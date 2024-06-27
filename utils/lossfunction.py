

from re import S
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.autograd import Variable
from numpy import *
import torch.cuda.amp as amp
from typing import Optional

"""
    label smoothing
    paper : https://arxiv.org/abs/1906.02629
    code: https://github.com/seominseok0429/label-smoothing-visualization-pytorch
"""


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (self.num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLoss:
    def __init__(self, alpha=None, gamma=0):
        """
        :param alpha: A list of weights for each class
        :param gamma:
        github:https://github.com/yxdr/pytorch-multi-class-focal-loss/blob/master/FocalLoss.py
        """
        self.alpha = torch.tensor(alpha) if alpha else None
        self.gamma = gamma

    def __call__(self, outputs, targets):
        if self.alpha is None and self.gamma == 0:
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets)

        elif self.alpha is not None and self.gamma == 0:
            if self.alpha.device != outputs.device:
                self.alpha = self.alpha.to(outputs)
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                           weight=self.alpha)

        elif self.alpha is None and self.gamma != 0:
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()

        elif self.alpha is not None and self.gamma != 0:
            if self.alpha.device != outputs.device:
                self.alpha = self.alpha.to(outputs)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                        weight=self.alpha, reduction='none')
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch

        return focal_loss


"""
paper: Auxiliary tasks in multi-task learning
github: https://github.com/Mikoto10032/AutomaticWeightedLoss
"""


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

"""
paper:End-to-End Multi-Task Learning with Attention
code:https://github.com/lorenmt/mtan/blob/268c5c1796bf972d1e0850dcf1e2f2e6598cc589/im2im_pred/utils.py
"""
class DynamicWeightAverage(nn.Module):
    """
    """
    def __init__(self,num):
        super(DynamicWeightAverage,self).__init__()
        lambd = torch.one(num,requires_grad=True)
        self.params = torch.nn.Parameter(lambd)
    
    def forward(self,*x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class InterflowLoss(nn.Module):
    def __init__(self,num,device):
        super(InterflowLoss,self).__init__()
        self.device = device
        self.num = num
        self.conv1 = nn.Conv2d(num, 1, 1,bias=False)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(1,num,1,bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, *x):
        l = torch.tensor(x).view(1,self.num,1,1).to(self.device)
        l = self.conv2(self.gelu(self.conv1(l)))
        l = self.sigmoid(l) * l
        l = l.view(1,self.num)
        l = l.tolist()
        return torch.tensor(sum(l),requires_grad=True)


class RecallLoss(nn.Module):
    """ An unofficial implementation of
        <Recall Loss for Imbalanced Image Classification and Semantic Segmentation>
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        recall = TP / (TP + FN)
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None):
        super(RecallLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, input, target):
        N, C = input.size()[:2]
        _, predict = torch.max(input, 1)# # (N, C, *) ==> (N, 1, *)

        predict = predict.view(N, 1, -1) # (N, 1, *)
        target = target.view(N, 1, -1) # (N, 1, *)
        last_size = target.size(-1)

        ## convert predict & target (N, 1, *) into one hot vector (N, C, *)
        predict_onehot = torch.zeros((N, C, last_size)).cuda() # (N, 1, *) ==> (N, C, *)
        predict_onehot.scatter_(1, predict, 1) # (N, C, *)
        target_onehot = torch.zeros((N, C, last_size)).cuda() # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1) # (N, C, *)

        true_positive = torch.sum(predict_onehot * target_onehot, dim=2)  # (N, C)
        total_target = torch.sum(target_onehot, dim=2)  # (N, C)
        ## Recall = TP / (TP + FN)
        recall = (true_positive + self.smooth) / (total_target + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != input.type():
                self.weight = self.weight.type_as(input)
                recall = recall * self.weight * C  # (N, C)
        recall_loss = 1 - torch.mean(recall)  # 1

        return recall_loss


class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss



##
# version 2: user derived grad computation
class LSRCrossEntropyFunctionV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, lb_smooth, lb_ignore):
        # prepare label
        num_classes = logits.size(1)
        lb_pos, lb_neg = 1. - lb_smooth, lb_smooth / num_classes
        label = label.clone().detach()
        ignore = label.eq(lb_ignore)
        n_valid = ignore.eq(0).sum()
        label[ignore] = 0
        lb_one_hot = torch.empty_like(logits).fill_(
            lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        ignore = ignore.nonzero(as_tuple=False)
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(logits.size(1)), *b]
        lb_one_hot[mask] = 0
        coeff = (num_classes - 1) * lb_neg + lb_pos

        ctx.variables = coeff, mask, logits, lb_one_hot

        loss = torch.log_softmax(logits, dim=1).neg_().mul_(lb_one_hot).sum(dim=1)
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        coeff, mask, logits, lb_one_hot = ctx.variables

        scores = torch.softmax(logits, dim=1).mul_(coeff)
        grad = scores.sub_(lb_one_hot).mul_(grad_output.unsqueeze(1))
        grad[mask] = 0
        return grad, None, None, None


class LabelSmoothSoftmaxCEV2(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV2, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, labels):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV2()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        losses = LSRCrossEntropyFunctionV2.apply(
                logits, labels, self.lb_smooth, self.lb_ignore)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            n_valid = (labels != self.lb_ignore).sum()
            losses = losses.sum() / n_valid
        return losses


class BinaryCrossEntropy(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """
    def __init__(
            self, smoothing=0.1, target_threshold: Optional[float] = None, weight: Optional[torch.Tensor] = None,
            reduction: str = 'mean', pos_weight: Optional[torch.Tensor] = None):
        super(BinaryCrossEntropy, self).__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing = smoothing
        self.target_threshold = target_threshold
        self.reduction = reduction
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
            num_classes = x.shape[-1]
            # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
            off_value = self.smoothing / num_classes
            on_value = 1. - self.smoothing + off_value
            target = target.long().view(-1, 1)
            target = torch.full(
                (target.size()[0], num_classes),
                off_value,
                device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
        if self.target_threshold is not None:
            # Make target 0, or 1 if threshold set
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        return F.binary_cross_entropy_with_logits(
            x, target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction)


class WeightedSumLoss(nn.Module):
    """
    Overall multi-task loss, consisting of a weighted sum of individual task losses. With
    optional resource loss.
    """

    def __init__(self, tasks, model=None):
        super().__init__()
        self.tasks = tasks
        self.loss_dict = nn.ModuleDict()
        for task in self.tasks:
            if task == 'shape':
                self.loss_dict[task] = FocalLoss(gamma=2)
            elif task == 'margin':
                self.loss_dict[task] = FocalLoss(gamma=2)
            elif task == 'bm':
                self.loss_dict[task] = FocalLoss(gamma=2)
            elif task == 'echo':
                self.loss_dict[task] = FocalLoss(gamma=2)
            elif task == 'composition':
                self.loss_dict[task] = FocalLoss(gamma=2)
            elif task == 'foci':
                self.loss_dict[task] = FocalLoss(gamma=2)
            else:
                raise ValueError

    def forward(self, out, lab, task=None):
        losses = []
        if task:
            losses.append(
                self.loss_dict[task](out[task], lab[task].t().long()-1))
        else:
            for t in self.tasks:
                losses.append(
                    self.loss_dict[t](out[t], lab[t].t().long()-1))

        tot_loss = sum(losses)
        return tot_loss



def kl_divergence_class(outC, outStrong):
    p = F.softmax(outC, dim=1)
    log_p = F.log_softmax(outC, dim=1)
    log_q = F.log_softmax(outStrong, dim=1)
    kl = p * (log_p - log_q)

    return kl.mean()


def calc_loss(outC, labels, outWeak, outStrong,ssl_weight=0.25,threshold=0.7):

    focal_label = FocalLoss(gamma=2)
    focal_unlabel = FocalLoss(gamma=2)

    lossClassifier = focal_label(outC, labels)

    probsWeak = torch.softmax(outWeak, dim=1)
    max_probs, psuedoLabels = torch.max(probsWeak, dim=1)
    mask = max_probs.ge(threshold).float()

    lossUnLabeled = (focal_unlabel(outStrong, psuedoLabels) * mask).mean()

    # kl_class = kl_divergence_class(outC, outStrong)

    loss = lossClassifier + (lossUnLabeled * ssl_weight)

    return loss

def calc_loss2(outWeak, outStrong,ssl_weight=0.25,threshold=0.7):


    focal_unlabel = FocalLoss(gamma=2)

    probsWeak = torch.softmax(outWeak, dim=1)
    max_probs, psuedoLabels = torch.max(probsWeak, dim=1)
    mask = max_probs.ge(threshold).float()

    lossUnLabeled = (focal_unlabel(outStrong, psuedoLabels) * mask).mean()

    # kl_class = kl_divergence_class(outC, outStrong)

    loss = lossUnLabeled * ssl_weight

    return loss


# a = torch.rand(4,2)
# b = torch.rand(4,2)
# c = torch.rand(4,2)
# d = torch.rand(4,2)
#
# e = calc_loss(a,b,c,d)
# print(e)