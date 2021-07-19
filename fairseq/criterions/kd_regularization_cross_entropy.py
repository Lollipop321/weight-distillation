# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('kd_regularization_cross_entropy')
class KDRegularizationCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.alpha = args.alpha
        self.temperature = args.temperature

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: offs
        parser.add_argument('--alpha', default=0., type=float, metavar='D',
                            help='params.reg_alpha')
        parser.add_argument('--temperature', default=0., type=float, metavar='D',
                            help='params.reg_temperature')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        # smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        """
            loss function for mannually-designed regularization: Tf-KD_{reg}
        """
        alpha = self.alpha
        T = self.temperature
        correct_prob = 0.99  # the probability for correct class in u(k)
        # loss_CE = F.cross_entropy(net_output, target)
        output = net_output[0]
        K = output.size(1)
        multiplier = 100

        teacher_soft = torch.ones_like(output).cuda()
        teacher_soft = teacher_soft * (1 - correct_prob) / (K - 1)  # p^d(k)
        for i in range(output.shape[0]):
            teacher_soft[i, target[i]] = correct_prob
        loss_soft_regu = torch.nn.KLDivLoss()(F.log_softmax(output, dim=1), F.softmax(teacher_soft / T, dim=1)) * multiplier

        if reduce:
            nll_loss = nll_loss.sum()
            # smooth_loss = smooth_loss.sum()
            loss_soft_regu = loss_soft_regu.sum()
        # eps_i = self.eps / lprobs.size(-1)
        # loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        loss = (1. - alpha) * nll_loss + alpha * loss_soft_regu
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
