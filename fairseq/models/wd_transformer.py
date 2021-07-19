# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq

from torch.serialization import default_restore_location

from fairseq import options, utils
from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LayerNorm, WDV52LayerNorm, WDV29LayerNorm,
    LearnedPositionalEmbedding, WDV52MultiheadAttention, WDV29MultiheadAttention, MultiheadAttention, SinusoidalPositionalEmbedding,
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel,
    FairseqModel, register_model, register_model_architecture,
)

@register_model('wd_transformer')
class WDTransformerModel(FairseqModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--teacher-model', metavar='DIR', default='',
                            help='path to load teacher model, should match --teacher-model')
        # parser.add_argument('--wd-require-gradient', action='store_true',
        #                     help='wd require gradient')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        args.wd_require_gradient = False

        filename = args.teacher_model
        state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        model = state['model']
        wd_arg = state['args']
        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens, model=model)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens, model=model, wd_arg=wd_arg)

        return WDTransformerModel(encoder, decoder)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded
            (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=True, model=None):
        super().__init__(dictionary)
        self.wd_require_gradient = args.wd_require_gradient

        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        num_embeddings = embed_tokens.num_embeddings
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_tokens_weight = nn.Embedding(num_embeddings, embed_dim, self.padding_idx)
        self.embed_tokens_bias = nn.Embedding(num_embeddings, embed_dim, self.padding_idx)
        torch.nn.init.constant_(self.embed_tokens_weight.weight, 1.)
        torch.nn.init.constant_(self.embed_tokens_bias.weight, 0.)
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
            # self.layer_norm = WDV29LayerNorm(embed_dim, args.wd_require_gradient)

        if args.wd_require_gradient == False:
            self.embed_tokens.weight.data = model['encoder.embed_tokens.weight']
            if self.normalize:
                self.layer_norm.weight.data = model['encoder.layer_norm.weight']
                self.layer_norm.bias.data = model['encoder.layer_norm.bias']
            for i in range(args.encoder_layers):
                selfattn_inproj_weight = model['encoder.layers.' + str(i) + '.self_attn.in_proj_weight']
                self.layers[i].self_attn.q_weight.data, self.layers[i].self_attn.k_weight.data, \
                self.layers[i].self_attn.v_weight.data = selfattn_inproj_weight.chunk(3, dim=0)
                selfattn_inproj_bias = model['encoder.layers.' + str(i) + '.self_attn.in_proj_bias']
                self.layers[i].self_attn.q_bias.data, self.layers[i].self_attn.k_bias.data, \
                self.layers[i].self_attn.v_bias.data = selfattn_inproj_bias.chunk(3, dim=0)
                self.layers[i].self_attn.out_proj.weight.data = model['encoder.layers.'+str(i)+'.self_attn.out_proj.weight']
                self.layers[i].self_attn.out_proj.bias.data = model['encoder.layers.'+str(i)+'.self_attn.out_proj.bias']
                self.layers[i].fc1.weight.data = model['encoder.layers.'+str(i)+'.fc1.weight']
                self.layers[i].fc1.bias.data = model['encoder.layers.'+str(i)+'.fc1.bias']
                self.layers[i].fc2.weight.data = model['encoder.layers.'+str(i)+'.fc2.weight']
                self.layers[i].fc2.bias.data = model['encoder.layers.'+str(i)+'.fc2.bias']
                self.layers[i].layer_norms[0].weight.data = model['encoder.layers.'+str(i)+'.layer_norms.0.weight']
                self.layers[i].layer_norms[0].bias.data = model['encoder.layers.'+str(i)+'.layer_norms.0.bias']
                self.layers[i].layer_norms[1].weight.data = model['encoder.layers.'+str(i)+'.layer_norms.1.weight']
                self.layers[i].layer_norms[1].bias.data = model['encoder.layers.'+str(i)+'.layer_norms.1.bias']

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        if self.wd_require_gradient == False:
            embed_tokens = self.embed_tokens_weight(src_tokens) * torch.tanh(self.embed_tokens(src_tokens).detach()) \
                       + self.embed_tokens_bias(src_tokens)
        else:
            embed_tokens = self.embed_tokens_weight(src_tokens) * torch.tanh(self.embed_tokens(src_tokens)) \
                       + self.embed_tokens_bias(src_tokens)
        x = self.embed_scale * embed_tokens
        # x = self.embed_scale * self.embed_tokens(src_tokens).detach()
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            # mark
            # 'encoder_out': x.detach(),  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
        left_pad (bool, optional): whether the input is left-padded
            (default: False).
        final_norm (bool, optional): apply layer norm to the output of the
            final decoder layer (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True, model=None, wd_arg=None):
        super().__init__(dictionary)

        self.wd_require_gradient = args.wd_require_gradient
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        wd_embed_dim = wd_arg.decoder_embed_dim
        self.wd_embed_dim = wd_embed_dim
        num_embeddings = embed_tokens.num_embeddings
        output_embed_dim = args.decoder_output_dim
        self.ts = int(wd_arg.decoder_layers / args.decoder_layers)

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_tokens_weight = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_tokens_bias = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        torch.nn.init.constant_(self.embed_tokens_weight.weight, 1.)
        torch.nn.init.constant_(self.embed_tokens_bias.weight, 0.)
        if wd_embed_dim != embed_dim:
            self.wd_embed_tokens_weight = nn.Parameter(torch.Tensor(wd_embed_dim, embed_dim))
            self.wd_embed_out_weight = nn.Parameter(torch.Tensor(wd_embed_dim, embed_dim))
            torch.nn.init.xavier_normal_(self.wd_embed_tokens_weight)
            torch.nn.init.xavier_normal_(self.wd_embed_out_weight)

        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(self.ts, args, no_encoder_attn, wd_arg)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            # self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            # mark
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim), requires_grad=False)
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
            # self.layer_norm = WDV29LayerNorm(embed_dim, args.wd_require_gradient)

        if args.wd_require_gradient == False:
            list_selfattn_inproj_weight_q = []
            list_selfattn_inproj_weight_k = []
            list_selfattn_inproj_weight_v = []
            list_selfattn_inproj_bias_q = []
            list_selfattn_inproj_bias_k = []
            list_selfattn_inproj_bias_v = []
            list_selfattn_outproj_weight = []
            list_selfattn_outproj_bias = []
            list_selfattn_layernorm_weight = []
            list_selfattn_layernorm_bias = []
            list_encoderattn_layernorm_weight = []
            list_encoderattn_layernorm_bias = []
            list_fc1_weight = []
            list_fc1_bias = []
            list_fc2_weight = []
            list_fc2_bias = []
            list_final_layernorm_weight = []
            list_final_layernorm_bias = []

            list_encoderattn_inproj_weight_q = []
            list_encoderattn_inproj_weight_k = []
            list_encoderattn_inproj_weight_v = []
            list_encoderattn_inproj_bias_q = []
            list_encoderattn_inproj_bias_k = []
            list_encoderattn_inproj_bias_v = []
            list_encoderattn_outproj_weight = []
            list_encoderattn_outproj_bias = []

            self.embed_tokens.weight.data = model['decoder.embed_tokens.weight']
            if hasattr(self, 'embed_out'):
                self.embed_out.data = model['decoder.embed_out']
            if self.normalize:
                self.layer_norm.weight.data = model['decoder.layer_norm.weight']
                self.layer_norm.bias.data = model['decoder.layer_norm.bias']
            for i in range(wd_arg.decoder_layers):
                selfattn_inproj_weight = model['decoder.layers.'+str(i)+'.self_attn.in_proj_weight']
                selfattn_inproj_weight_q, selfattn_inproj_weight_k, selfattn_inproj_weight_v = selfattn_inproj_weight.chunk(3, dim=0)
                list_selfattn_inproj_weight_q.append(selfattn_inproj_weight_q)
                list_selfattn_inproj_weight_k.append(selfattn_inproj_weight_k)
                list_selfattn_inproj_weight_v.append(selfattn_inproj_weight_v)
                selfattn_inproj_bias = model['decoder.layers.'+str(i)+'.self_attn.in_proj_bias']
                selfattn_inproj_bias_q, selfattn_inproj_bias_k, selfattn_inproj_bias_v = selfattn_inproj_bias.chunk(3, dim=0)
                list_selfattn_inproj_bias_q.append(selfattn_inproj_bias_q)
                list_selfattn_inproj_bias_k.append(selfattn_inproj_bias_k)
                list_selfattn_inproj_bias_v.append(selfattn_inproj_bias_v)
                selfattn_outproj_weight = model['decoder.layers.'+str(i)+'.self_attn.out_proj.weight']
                list_selfattn_outproj_weight.append(selfattn_outproj_weight)
                selfattn_outproj_bias = model['decoder.layers.'+str(i)+'.self_attn.out_proj.bias']
                list_selfattn_outproj_bias.append(selfattn_outproj_bias)
                selfattn_layernorm_weight = model['decoder.layers.'+str(i)+'.self_attn_layer_norm.weight']
                list_selfattn_layernorm_weight.append(selfattn_layernorm_weight)
                selfattn_layernorm_bias = model['decoder.layers.'+str(i)+'.self_attn_layer_norm.bias']
                list_selfattn_layernorm_bias.append(selfattn_layernorm_bias)
                encoderattn_layernorm_weight = model['decoder.layers.'+str(i)+'.encoder_attn_layer_norm.weight']
                list_encoderattn_layernorm_weight.append(encoderattn_layernorm_weight)
                encoderattn_layernorm_bias = model['decoder.layers.'+str(i)+'.encoder_attn_layer_norm.bias']
                list_encoderattn_layernorm_bias.append(encoderattn_layernorm_bias)
                fc1_weight = model['decoder.layers.'+str(i)+'.fc1.weight']
                list_fc1_weight.append(fc1_weight)
                fc1_bias = model['decoder.layers.'+str(i)+'.fc1.bias']
                list_fc1_bias.append(fc1_bias)
                fc2_weight = model['decoder.layers.'+str(i)+'.fc2.weight']
                list_fc2_weight.append(fc2_weight)
                fc2_bias = model['decoder.layers.'+str(i)+'.fc2.bias']
                list_fc2_bias.append(fc2_bias)
                final_layernorm_weight = model['decoder.layers.'+str(i)+'.final_layer_norm.weight']
                list_final_layernorm_weight.append(final_layernorm_weight)
                final_layernorm_bias = model['decoder.layers.'+str(i)+'.final_layer_norm.bias']
                list_final_layernorm_bias.append(final_layernorm_bias)

                encoderattn_inproj_weight = model['decoder.layers.' + str(i) + '.encoder_attn.in_proj_weight']
                encoderattn_inproj_weight_q, encoderattn_inproj_weight_k, encoderattn_inproj_weight_v = encoderattn_inproj_weight.chunk(3, dim=0)
                list_encoderattn_inproj_weight_q.append(encoderattn_inproj_weight_q)
                list_encoderattn_inproj_weight_k.append(encoderattn_inproj_weight_k)
                list_encoderattn_inproj_weight_v.append(encoderattn_inproj_weight_v)
                encoderattn_inproj_bias = model['decoder.layers.' + str(i) + '.encoder_attn.in_proj_bias']
                encoderattn_inproj_bias_q, encoderattn_inproj_bias_k, encoderattn_inproj_bias_v = encoderattn_inproj_bias.chunk(3, dim=0)
                list_encoderattn_inproj_bias_q.append(encoderattn_inproj_bias_q)
                list_encoderattn_inproj_bias_k.append(encoderattn_inproj_bias_k)
                list_encoderattn_inproj_bias_v.append(encoderattn_inproj_bias_v)
                encoderattn_outproj_weight = model['decoder.layers.' + str(i) + '.encoder_attn.out_proj.weight']
                list_encoderattn_outproj_weight.append(encoderattn_outproj_weight)
                encoderattn_outproj_bias = model['decoder.layers.' + str(i) + '.encoder_attn.out_proj.bias']
                list_encoderattn_outproj_bias.append(encoderattn_outproj_bias)

            for i in range(args.decoder_layers):
                j = self.ts * i
                k = self.ts * (i+1)
                self.layers[i].self_attn.q_weight.data = torch.stack(list_selfattn_inproj_weight_q[j:k], -1)
                self.layers[i].self_attn.k_weight.data = torch.stack(list_selfattn_inproj_weight_k[j:k], -1)
                self.layers[i].self_attn.v_weight.data = torch.stack(list_selfattn_inproj_weight_v[j:k], -1)
                self.layers[i].self_attn.q_bias.data = torch.stack(list_selfattn_inproj_bias_q[j:k], -1)
                self.layers[i].self_attn.k_bias.data = torch.stack(list_selfattn_inproj_bias_k[j:k], -1)
                self.layers[i].self_attn.v_bias.data = torch.stack(list_selfattn_inproj_bias_v[j:k], -1)
                self.layers[i].self_attn.out_proj.weight.data = torch.stack(list_selfattn_outproj_weight[j:k], -1)
                self.layers[i].self_attn.out_proj.bias.data = torch.stack(list_selfattn_outproj_bias[j:k], -1)
                self.layers[i].self_attn_layer_norm.weight.data = torch.stack(list_selfattn_layernorm_weight[j:k], -1)
                self.layers[i].self_attn_layer_norm.bias.data = torch.stack(list_selfattn_layernorm_bias[j:k], -1)
                self.layers[i].encoder_attn_layer_norm.weight.data = torch.stack(list_encoderattn_layernorm_weight[j:k], -1)
                self.layers[i].encoder_attn_layer_norm.bias.data = torch.stack(list_encoderattn_layernorm_bias[j:k], -1)
                self.layers[i].fc1.weight.data = torch.stack(list_fc1_weight[j:k], -1)
                self.layers[i].fc1.bias.data = torch.stack(list_fc1_bias[j:k], -1)
                self.layers[i].fc2.weight.data = torch.stack(list_fc2_weight[j:k], -1)
                self.layers[i].fc2.bias.data = torch.stack(list_fc2_bias[j:k], -1)
                self.layers[i].final_layer_norm.weight.data = torch.stack(list_final_layernorm_weight[j:k], -1)
                self.layers[i].final_layer_norm.bias.data = torch.stack(list_final_layernorm_bias[j:k], -1)

                self.layers[i].encoder_attn.q_weight.data = torch.stack(list_encoderattn_inproj_weight_q[j:k], -1)
                self.layers[i].encoder_attn.k_weight.data = torch.stack(list_encoderattn_inproj_weight_k[j:k], -1)
                self.layers[i].encoder_attn.v_weight.data = torch.stack(list_encoderattn_inproj_weight_v[j:k], -1)
                self.layers[i].encoder_attn.q_bias.data = torch.stack(list_encoderattn_inproj_bias_q[j:k], -1)
                self.layers[i].encoder_attn.k_bias.data = torch.stack(list_encoderattn_inproj_bias_k[j:k], -1)
                self.layers[i].encoder_attn.v_bias.data = torch.stack(list_encoderattn_inproj_bias_v[j:k], -1)
                self.layers[i].encoder_attn.out_proj.weight.data = torch.stack(list_encoderattn_outproj_weight[j:k], -1)
                self.layers[i].encoder_attn.out_proj.bias.data = torch.stack(list_encoderattn_outproj_bias[j:k], -1)


    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        if self.wd_require_gradient == False:
            embed_tokens = self.embed_tokens(prev_output_tokens).detach()
        else:
            embed_tokens = self.embed_tokens(prev_output_tokens)
        if self.embed_dim != self.wd_embed_dim:
            embed_tokens = torch.matmul(embed_tokens, self.wd_embed_tokens_weight)
        embed_tokens = self.embed_tokens_weight(prev_output_tokens) * torch.tanh(embed_tokens) \
                       + self.embed_tokens_bias(prev_output_tokens)
        x = self.embed_scale * embed_tokens
        # x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        # mark
        # x = self.embed_scale * self.embed_tokens(prev_output_tokens).detach()

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)


        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                embed_out = self.embed_out
                if self.embed_dim != self.wd_embed_dim:
                    embed_out = torch.matmul(embed_out, self.wd_embed_out_weight)
                x = F.linear(x, embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = WDV29MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
            wd_require_gradient=args.wd_require_gradient,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = WDV29Linear(self.embed_dim, args.encoder_ffn_embed_dim, args.wd_require_gradient)
        self.fc2 = WDV29Linear(args.encoder_ffn_embed_dim, self.embed_dim, args.wd_require_gradient)
        self.layer_norms = nn.ModuleList([WDV29LayerNorm(self.embed_dim, args.wd_require_gradient) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, ts, args, no_encoder_attn=False, wd_arg=None):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = WDV52MultiheadAttention(
            ts, args,
            self.embed_dim, args.decoder_attention_heads,
            wd_arg.decoder_embed_dim, wd_arg.decoder_layers,
            dropout=args.attention_dropout,
            is_self=True,
            wd_require_gradient=args.wd_require_gradient,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = WDV52LayerNorm(ts, self.embed_dim, wd_arg.decoder_embed_dim, wd_arg.decoder_layers, args.wd_require_gradient)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = WDV52MultiheadAttention(
                ts, args,
                self.embed_dim, args.decoder_attention_heads,
                wd_arg.decoder_embed_dim, wd_arg.decoder_layers,
                dropout=args.attention_dropout,
                is_self=False,
                wd_require_gradient=args.wd_require_gradient,
            )
            self.encoder_attn_layer_norm = WDV52LayerNorm(ts, self.embed_dim, wd_arg.decoder_embed_dim, wd_arg.decoder_layers, args.wd_require_gradient)

        self.fc1 = WDV52Linear(ts, self.embed_dim, args.decoder_ffn_embed_dim, wd_arg.decoder_embed_dim, wd_arg.decoder_ffn_embed_dim, wd_arg.decoder_layers, args.wd_require_gradient)
        self.fc2 = WDV52Linear(ts, args.decoder_ffn_embed_dim, self.embed_dim, wd_arg.decoder_ffn_embed_dim, wd_arg.decoder_embed_dim, wd_arg.decoder_layers, args.wd_require_gradient)

        self.final_layer_norm = WDV52LayerNorm(ts, self.embed_dim, wd_arg.decoder_embed_dim, wd_arg.decoder_layers, args.wd_require_gradient)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state,
                prev_self_attn_state=None, prev_attn_state=None, self_attn_mask=None,
                self_attn_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def WDV52Linear(ts, in_features, out_features, wd_in_features, wd_out_features, wd_decoder_layers, wd_require_gradient=False, bias=True):
    # m = nn.Linear(in_features, out_features, bias)
    m = fairseq.modules.WDV52Linear(ts, in_features, out_features, wd_in_features, wd_out_features, wd_decoder_layers, wd_require_gradient, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def WDV29Linear(in_features, out_features, wd_require_gradient=False, bias=True):
    # m = nn.Linear(in_features, out_features, bias)
    m = fairseq.modules.WDV29Linear(in_features, out_features, wd_require_gradient, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m


@register_model_architecture('wd_transformer', 'wd_transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

@register_model_architecture('wd_transformer', 'wd_transformer_wmt_en_de')
def transformer_wmt_en_de(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    base_architecture(args)

@register_model_architecture('wd_transformer', 'wd_transformer_wmt_en_de_stu1')
def transformer_wmt_en_de_stu1(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    transformer_wmt_en_de(args)

@register_model_architecture('wd_transformer', 'wd_transformer_wmt_en_de_stu2')
def transformer_wmt_en_de_stu2(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    transformer_wmt_en_de(args)