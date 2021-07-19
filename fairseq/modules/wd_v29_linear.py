# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init


class WDV29Linear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, wd_require_gradient=False, bias=True):
        super(WDV29Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=wd_require_gradient)
        self.tanh_weight_weight = Parameter(torch.Tensor(out_features, in_features))
        self.tanh_bias_weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features), requires_grad=wd_require_gradient)
            self.tanh_weight_bias = Parameter(torch.Tensor(out_features))
            self.tanh_bias_bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.tanh_weight_weight, 1.)
        torch.nn.init.constant_(self.tanh_weight_bias, 1.)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            torch.nn.init.constant_(self.tanh_bias_weight, 0.)
            torch.nn.init.constant_(self.tanh_bias_bias, 0.)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = self.weight
        weight = self.tanh_weight_weight * torch.tanh(weight) + self.tanh_bias_weight
        bias = self.bias
        bias = self.tanh_weight_bias * torch.tanh(bias) + self.tanh_bias_bias
        return F.linear(input, weight, bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )