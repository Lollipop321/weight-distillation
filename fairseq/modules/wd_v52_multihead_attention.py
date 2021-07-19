# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import fairseq

from fairseq import utils


class WDV52MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, ts, args, embed_dim, num_heads, wd_embed_dim, wd_decoder_layers, dropout=0., is_self=True, wd_require_gradient=False, bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.wd_decoder_layers = wd_decoder_layers
        self.is_self = is_self
        self.ts = ts

        if is_self==True:
            in_embed_dim = args.decoder_embed_dim
        else:
            in_embed_dim = args.encoder_embed_dim

        self.out_wd_q_weight = Parameter(torch.Tensor(wd_embed_dim, embed_dim))
        self.out_wd_k_weight = Parameter(torch.Tensor(wd_embed_dim, embed_dim))
        self.out_wd_v_weight = Parameter(torch.Tensor(wd_embed_dim, embed_dim))
        self.in_wd_q_weight = Parameter(torch.Tensor(wd_embed_dim, embed_dim))
        self.in_wd_k_weight = Parameter(torch.Tensor(wd_embed_dim, in_embed_dim))
        self.in_wd_v_weight = Parameter(torch.Tensor(wd_embed_dim, in_embed_dim))
        self.ly_wd_q_weight = Parameter(torch.Tensor(ts, ts))
        self.ly_wd_k_weight = Parameter(torch.Tensor(ts, ts))
        self.ly_wd_v_weight = Parameter(torch.Tensor(ts, ts))
        self.wd_q_weight = Parameter(torch.Tensor(ts, 1))
        self.wd_k_weight = Parameter(torch.Tensor(ts, 1))
        self.wd_v_weight = Parameter(torch.Tensor(ts, 1))
        self.q_weight = Parameter(torch.Tensor(wd_embed_dim, wd_embed_dim, ts), requires_grad=wd_require_gradient)
        self.k_weight = Parameter(torch.Tensor(wd_embed_dim, wd_embed_dim, ts), requires_grad=wd_require_gradient)
        self.v_weight = Parameter(torch.Tensor(wd_embed_dim, wd_embed_dim, ts), requires_grad=wd_require_gradient)
        self.tanh_weight_q_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.tanh_weight_k_weight = Parameter(torch.Tensor(embed_dim, in_embed_dim))
        self.tanh_weight_v_weight = Parameter(torch.Tensor(embed_dim, in_embed_dim))
        self.tanh_bias_q_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.tanh_bias_k_weight = Parameter(torch.Tensor(embed_dim, in_embed_dim))
        self.tanh_bias_v_weight = Parameter(torch.Tensor(embed_dim, in_embed_dim))
        if bias:
            self.out_wd_q_bias = Parameter(torch.Tensor(wd_embed_dim, embed_dim))
            self.out_wd_k_bias = Parameter(torch.Tensor(wd_embed_dim, embed_dim))
            self.out_wd_v_bias = Parameter(torch.Tensor(wd_embed_dim, embed_dim))
            self.ly_wd_q_bias = Parameter(torch.Tensor(ts, ts))
            self.ly_wd_k_bias = Parameter(torch.Tensor(ts, ts))
            self.ly_wd_v_bias = Parameter(torch.Tensor(ts, ts))
            self.wd_q_bias = Parameter(torch.Tensor(ts, 1))
            self.wd_k_bias = Parameter(torch.Tensor(ts, 1))
            self.wd_v_bias = Parameter(torch.Tensor(ts, 1))
            self.q_bias = Parameter(torch.Tensor(wd_embed_dim, ts), requires_grad=wd_require_gradient)
            self.k_bias = Parameter(torch.Tensor(wd_embed_dim, ts), requires_grad=wd_require_gradient)
            self.v_bias = Parameter(torch.Tensor(wd_embed_dim, ts), requires_grad=wd_require_gradient)
            self.tanh_weight_q_bias = Parameter(torch.Tensor(embed_dim))
            self.tanh_weight_k_bias = Parameter(torch.Tensor(embed_dim))
            self.tanh_weight_v_bias = Parameter(torch.Tensor(embed_dim))
            self.tanh_bias_q_bias = Parameter(torch.Tensor(embed_dim))
            self.tanh_bias_k_bias = Parameter(torch.Tensor(embed_dim))
            self.tanh_bias_v_bias = Parameter(torch.Tensor(embed_dim))
        else:
            self.register_parameter('q_bias', None)
            self.register_parameter('k_bias', None)
            self.register_parameter('v_bias', None)
        self.out_proj = fairseq.modules.WDV52Linear(ts, embed_dim, embed_dim, wd_embed_dim, wd_embed_dim, wd_decoder_layers, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.out_wd_q_weight)
        torch.nn.init.xavier_normal_(self.out_wd_k_weight)
        torch.nn.init.xavier_normal_(self.out_wd_v_weight)
        torch.nn.init.xavier_normal_(self.in_wd_q_weight)
        torch.nn.init.xavier_normal_(self.in_wd_k_weight)
        torch.nn.init.xavier_normal_(self.in_wd_v_weight)
        torch.nn.init.xavier_normal_(self.ly_wd_q_weight)
        torch.nn.init.xavier_normal_(self.ly_wd_k_weight)
        torch.nn.init.xavier_normal_(self.ly_wd_v_weight)
        nn.init.xavier_uniform_(self.q_weight)
        nn.init.xavier_uniform_(self.k_weight)
        nn.init.xavier_uniform_(self.v_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        torch.nn.init.constant_(self.wd_q_weight, 1 / self.ts)
        torch.nn.init.constant_(self.wd_k_weight, 1 / self.ts)
        torch.nn.init.constant_(self.wd_v_weight, 1 / self.ts)
        torch.nn.init.constant_(self.tanh_weight_q_weight, 1.)
        torch.nn.init.constant_(self.tanh_weight_q_bias, 1.)
        torch.nn.init.constant_(self.tanh_weight_k_weight, 1.)
        torch.nn.init.constant_(self.tanh_weight_k_bias, 1.)
        torch.nn.init.constant_(self.tanh_weight_v_weight, 1.)
        torch.nn.init.constant_(self.tanh_weight_v_bias, 1.)
        if self.wd_q_bias is not None:
            torch.nn.init.xavier_normal_(self.out_wd_q_bias)
            torch.nn.init.xavier_normal_(self.out_wd_k_bias)
            torch.nn.init.xavier_normal_(self.out_wd_v_bias)
            torch.nn.init.xavier_normal_(self.ly_wd_q_bias)
            torch.nn.init.xavier_normal_(self.ly_wd_k_bias)
            torch.nn.init.xavier_normal_(self.ly_wd_v_bias)
            torch.nn.init.constant_(self.wd_q_bias, 1 / self.ts)
            torch.nn.init.constant_(self.wd_k_bias, 1 / self.ts)
            torch.nn.init.constant_(self.wd_v_bias, 1 / self.ts)
            torch.nn.init.constant_(self.tanh_bias_q_weight, 0.)
            torch.nn.init.constant_(self.tanh_bias_q_bias, 0.)
            torch.nn.init.constant_(self.tanh_bias_k_weight, 0.)
            torch.nn.init.constant_(self.tanh_bias_k_bias, 0.)
            torch.nn.init.constant_(self.tanh_bias_v_weight, 0.)
            torch.nn.init.constant_(self.tanh_bias_v_bias, 0.)
            nn.init.constant_(self.q_bias, 0.)
            nn.init.constant_(self.k_bias, 0.)
            nn.init.constant_(self.v_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        q = self.in_proj_q(query)
        k = self.in_proj_k(key) if key is not None else None
        v = self.in_proj_v(value) if value is not None else None
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            # print(attn_weights.size())
            # print(attn_mask.size())
            # print('hello')
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self.in_proj_wd(query, self.tanh_weight_q_weight, self.tanh_bias_q_weight, self.tanh_weight_q_bias, self.tanh_bias_q_bias,
                               self.out_wd_q_weight, self.in_wd_q_weight, self.ly_wd_q_weight, self.q_weight, self.wd_q_weight, self.out_wd_q_bias, self.ly_wd_q_bias, self.q_bias, self.wd_q_bias)

    def in_proj_k(self, key):
        return self.in_proj_wd(key, self.tanh_weight_k_weight, self.tanh_bias_k_weight, self.tanh_weight_k_bias, self.tanh_bias_k_bias,
                               self.out_wd_k_weight, self.in_wd_k_weight, self.ly_wd_k_weight, self.k_weight, self.wd_k_weight, self.out_wd_k_bias, self.ly_wd_k_bias, self.k_bias, self.wd_k_bias)

    def in_proj_v(self, value):
        return self.in_proj_wd(value, self.tanh_weight_v_weight, self.tanh_bias_v_weight, self.tanh_weight_v_bias, self.tanh_bias_v_bias,
                               self.out_wd_v_weight, self.in_wd_v_weight, self.ly_wd_v_weight, self.v_weight, self.wd_v_weight, self.out_wd_v_bias, self.ly_wd_v_bias, self.v_bias, self.wd_v_bias)

    def _in_proj(self, input, start=0, end=None):
        weight_wd = torch.matmul(self.in_proj_weight, self.wd_weight)
        weight_wd = weight_wd.squeeze(-1)
        bias_wd = torch.matmul(self.in_proj_bias, self.wd_bias)
        bias_wd = bias_wd.squeeze(-1)
        weight_wd = weight_wd[start:end, :]
        if bias_wd is not None:
            bias_wd = bias_wd[start:end]
        return F.linear(input, weight_wd, bias_wd)

    def in_proj_wd(self, input, tanh_weight_weight, tanh_bias_weight, tanh_weight_bias, tanh_bias_bias, weight_wd_out, weight_wd_in, weight_wd_ly, weight, weight_wd_tmp, bias_wd_out, bias_wd_ly, bias, bias_wd_tmp):
        weight_wd = torch.transpose(weight, 0, 2)
        weight_wd = torch.matmul(weight_wd, weight_wd_out)
        weight_wd = torch.transpose(weight_wd, 0, 2)
        weight_wd = torch.matmul(weight_wd, weight_wd_ly)
        weight_wd = torch.transpose(weight_wd, 1, 2)
        weight_wd = torch.matmul(weight_wd, weight_wd_in)
        weight_wd = torch.transpose(weight_wd, 1, 2)
        weight_wd = torch.matmul(weight_wd, weight_wd_tmp)
        weight_wd = weight_wd.squeeze(-1)
        weight_wd = tanh_weight_weight * torch.tanh(weight_wd) + tanh_bias_weight
        if bias_wd_tmp is not None:
            bias_wd = torch.transpose(bias, 0, 1)
            bias_wd = torch.matmul(bias_wd, bias_wd_out)
            bias_wd = torch.transpose(bias_wd, 0, 1)
            bias_wd = torch.matmul(bias_wd, bias_wd_ly)
            bias_wd = torch.matmul(bias_wd, bias_wd_tmp)
            bias_wd = bias_wd.squeeze(-1)
            bias_wd = tanh_weight_bias * torch.tanh(bias_wd) + tanh_bias_bias
        return F.linear(input, weight_wd, bias_wd)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )