import math
import numbers

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init


class WDV52LayerNormalization(torch.nn.Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization`_ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = nn.LayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = nn.LayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
    """
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, ts, normalized_shape, wd_normalized_shape, wd_decoder_layers, wd_require_gradient=False, eps=1e-5, elementwise_affine=True):
        super(WDV52LayerNormalization, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.ts = ts
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.wd_decoder_layers = wd_decoder_layers
        if self.elementwise_affine:
            self.out_wd_weight = Parameter(torch.Tensor(wd_normalized_shape, *normalized_shape))
            self.ly_wd_weight = Parameter(torch.Tensor(ts, ts))
            self.wd_weight = Parameter(torch.Tensor(ts, 1))
            self.weight = Parameter(torch.Tensor(wd_normalized_shape, ts), requires_grad=wd_require_gradient)
            self.tanh_weight_weight = Parameter(torch.Tensor(*normalized_shape))
            self.tanh_weight_bias = Parameter(torch.Tensor(*normalized_shape))
            self.out_wd_bias = Parameter(torch.Tensor(wd_normalized_shape, *normalized_shape))
            self.ly_wd_bias = Parameter(torch.Tensor(ts, ts))
            self.wd_bias = Parameter(torch.Tensor(ts, 1))
            self.bias = Parameter(torch.Tensor(wd_normalized_shape, ts), requires_grad=wd_require_gradient)
            self.tanh_bias_weight = Parameter(torch.Tensor(*normalized_shape))
            self.tanh_bias_bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            torch.nn.init.xavier_normal_(self.out_wd_weight)
            torch.nn.init.xavier_normal_(self.ly_wd_weight)
            torch.nn.init.xavier_normal_(self.out_wd_bias)
            torch.nn.init.xavier_normal_(self.ly_wd_bias)
            torch.nn.init.constant_(self.wd_weight, 1 / self.ts)
            torch.nn.init.constant_(self.wd_bias, 1 / self.ts)
            torch.nn.init.constant_(self.tanh_weight_weight, 1.)
            torch.nn.init.constant_(self.tanh_weight_bias, 1.)
            torch.nn.init.constant_(self.tanh_bias_weight, 0.)
            torch.nn.init.constant_(self.tanh_bias_bias, 0.)
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        weight = self.weight
        weight_wd = torch.transpose(weight, 0, 1)
        weight_wd = torch.matmul(weight_wd, self.out_wd_weight)
        weight_wd = torch.transpose(weight_wd, 0, 1)
        weight_wd = torch.matmul(weight_wd, self.ly_wd_weight)
        weight_wd = torch.matmul(weight_wd, self.wd_weight)
        weight_wd = weight_wd.squeeze(-1)
        weight_wd = self.tanh_weight_weight * torch.tanh(weight_wd) + self.tanh_bias_weight
        bias = self.bias
        bias_wd = torch.transpose(bias, 0, 1)
        bias_wd = torch.matmul(bias_wd, self.out_wd_bias)
        bias_wd = torch.transpose(bias_wd, 0, 1)
        bias_wd = torch.matmul(bias_wd, self.ly_wd_bias)
        bias_wd = torch.matmul(bias_wd, self.wd_bias)
        bias_wd = bias_wd.squeeze(-1)
        bias_wd = self.tanh_weight_bias * torch.tanh(bias_wd) + self.tanh_bias_bias
        return F.layer_norm(
            input, self.normalized_shape, weight_wd, bias_wd, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
