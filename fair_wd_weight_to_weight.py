import torch
import math
import time
import struct
import argparse
import numpy as np
from collections import OrderedDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True, help="trained model prefix, also include dir, e.g. ../data/model-100")

    args = parser.parse_args()

    model_path = args.model

    checkpoint = torch.load(model_path, map_location='cpu')
    assert 'args' in checkpoint
    assert 'model' in checkpoint
    args = checkpoint['args']
    model = checkpoint['model']

    checkpoint_new = {}
    model_new = {}

    for key in model.keys():
        if not key.startswith('decoder.layers'):
            if not key.startswith('encoder.layers'):
                if key.startswith('encoder.embed_tokens'):
                    embed_tokens_weight_en = model['encoder.embed_tokens.weight'].float()
                    embed_tokens_weight_weight_en = model['encoder.embed_tokens_weight.weight'].float()
                    embed_tokens_weight_bias_en = model['encoder.embed_tokens_bias.weight'].float()
                    embed_tokens_weight_en = embed_tokens_weight_weight_en * torch.tanh(embed_tokens_weight_en) \
                                             + embed_tokens_weight_bias_en
                    model_new['encoder.embed_tokens.weight'] = embed_tokens_weight_en
                elif key.startswith('decoder.embed_tokens'):
                    if 'decoder.wd_embed_tokens_weight' in model.keys():
                        wd_embed_tokens_weight_de = model['decoder.wd_embed_tokens_weight'].float()
                        embed_tokens_weight_de = model['decoder.embed_tokens.weight'].float()
                        embed_tokens_weight_weight_de = model['decoder.embed_tokens_weight.weight'].float()
                        embed_tokens_weight_bias_de = model['decoder.embed_tokens_bias.weight'].float()
                        embed_tokens_weight_de = torch.matmul(embed_tokens_weight_de, wd_embed_tokens_weight_de)
                        embed_tokens_weight_de = embed_tokens_weight_weight_de * torch.tanh(embed_tokens_weight_de) \
                                                 + embed_tokens_weight_bias_de
                        model_new['decoder.embed_tokens.weight'] = embed_tokens_weight_de
                    else:
                        embed_tokens_weight_de = model['decoder.embed_tokens.weight'].float()
                        embed_tokens_weight_weight_de = model['decoder.embed_tokens_weight.weight'].float()
                        embed_tokens_weight_bias_de = model['decoder.embed_tokens_bias.weight'].float()
                        embed_tokens_weight_de = embed_tokens_weight_weight_de * torch.tanh(embed_tokens_weight_de) \
                                                 + embed_tokens_weight_bias_de
                        model_new['decoder.embed_tokens.weight'] = embed_tokens_weight_de
                elif key.startswith('decoder.embed_out'):
                    if 'decoder.wd_embed_out_weight' in model.keys():
                        wd_embed_out_weight_de = model['decoder.wd_embed_out_weight'].float()
                        embed_out_de = model['decoder.embed_out'].float()
                        embed_out_de = torch.matmul(embed_out_de, wd_embed_out_weight_de)
                        model_new['decoder.embed_out'] = embed_out_de
                    else:
                        model_new['decoder.embed_out'] = model['decoder.embed_out']
                else:
                    if key.startswith('decoder.wd'):
                        continue
                    elif key.startswith('encoder.wd'):
                        continue
                    else:
                        model_new[key] = model[key]

    for i in range(args.encoder_layers):
        # encoder
        selfattn_q_weight_en = model['encoder.layers.' + str(i) + '.self_attn.q_weight'].float()
        selfattn_k_weight_en = model['encoder.layers.' + str(i) + '.self_attn.k_weight'].float()
        selfattn_v_weight_en = model['encoder.layers.' + str(i) + '.self_attn.v_weight'].float()
        selfattn_tanh_weight_q_weight_en = model['encoder.layers.' + str(i) + '.self_attn.tanh_weight_q_weight'].float()
        selfattn_tanh_weight_k_weight_en = model['encoder.layers.' + str(i) + '.self_attn.tanh_weight_k_weight'].float()
        selfattn_tanh_weight_v_weight_en = model['encoder.layers.' + str(i) + '.self_attn.tanh_weight_v_weight'].float()
        selfattn_tanh_bias_q_weight_en = model['encoder.layers.' + str(i) + '.self_attn.tanh_bias_q_weight'].float()
        selfattn_tanh_bias_k_weight_en = model['encoder.layers.' + str(i) + '.self_attn.tanh_bias_k_weight'].float()
        selfattn_tanh_bias_v_weight_en = model['encoder.layers.' + str(i) + '.self_attn.tanh_bias_v_weight'].float()
        selfattn_q_weight_en = selfattn_tanh_weight_q_weight_en * torch.tanh(selfattn_q_weight_en) + selfattn_tanh_bias_q_weight_en
        selfattn_k_weight_en = selfattn_tanh_weight_k_weight_en * torch.tanh(selfattn_k_weight_en) + selfattn_tanh_bias_k_weight_en
        selfattn_v_weight_en = selfattn_tanh_weight_v_weight_en * torch.tanh(selfattn_v_weight_en) + selfattn_tanh_bias_v_weight_en
        selfattn_inproj_weight_en = torch.cat((selfattn_q_weight_en, selfattn_k_weight_en, selfattn_v_weight_en), dim=0)
        model_new['encoder.layers.' + str(i) + '.self_attn.in_proj_weight'] = selfattn_inproj_weight_en

        selfattn_q_bias_en = model['encoder.layers.' + str(i) + '.self_attn.q_bias'].float()
        selfattn_k_bias_en = model['encoder.layers.' + str(i) + '.self_attn.k_bias'].float()
        selfattn_v_bias_en = model['encoder.layers.' + str(i) + '.self_attn.v_bias'].float()
        selfattn_tanh_weight_q_bias_en = model['encoder.layers.' + str(i) + '.self_attn.tanh_weight_q_bias'].float()
        selfattn_tanh_weight_k_bias_en = model['encoder.layers.' + str(i) + '.self_attn.tanh_weight_k_bias'].float()
        selfattn_tanh_weight_v_bias_en = model['encoder.layers.' + str(i) + '.self_attn.tanh_weight_v_bias'].float()
        selfattn_tanh_bias_q_bias_en = model['encoder.layers.' + str(i) + '.self_attn.tanh_bias_q_bias'].float()
        selfattn_tanh_bias_k_bias_en = model['encoder.layers.' + str(i) + '.self_attn.tanh_bias_k_bias'].float()
        selfattn_tanh_bias_v_bias_en = model['encoder.layers.' + str(i) + '.self_attn.tanh_bias_v_bias'].float()
        selfattn_q_bias_en = selfattn_tanh_weight_q_bias_en * torch.tanh(selfattn_q_bias_en) + selfattn_tanh_bias_q_bias_en
        selfattn_k_bias_en = selfattn_tanh_weight_k_bias_en * torch.tanh(selfattn_k_bias_en) + selfattn_tanh_bias_k_bias_en
        selfattn_v_bias_en = selfattn_tanh_weight_v_bias_en * torch.tanh(selfattn_v_bias_en) + selfattn_tanh_bias_v_bias_en
        selfattn_inproj_bias_en = torch.cat((selfattn_q_bias_en, selfattn_k_bias_en, selfattn_v_bias_en), dim=0)
        model_new['encoder.layers.' + str(i) + '.self_attn.in_proj_bias'] = selfattn_inproj_bias_en

        selfattn_outproj_weight_en = model['encoder.layers.' + str(i) + '.self_attn.out_proj.weight'].float()
        selfattn_outproj_tanh_weight_weight_en = model[
            'encoder.layers.' + str(i) + '.self_attn.out_proj.tanh_weight_weight'].float()
        selfattn_outproj_tanh_bias_weight_en = model[
            'encoder.layers.' + str(i) + '.self_attn.out_proj.tanh_bias_weight'].float()
        selfattn_outproj_weight_en = selfattn_outproj_tanh_weight_weight_en * torch.tanh(selfattn_outproj_weight_en) + selfattn_outproj_tanh_bias_weight_en
        model_new['encoder.layers.' + str(i) + '.self_attn.out_proj.weight'] = selfattn_outproj_weight_en

        selfattn_outproj_bias_en = model['encoder.layers.' + str(i) + '.self_attn.out_proj.bias'].float()
        selfattn_outproj_tanh_weight_bias_en = model[
            'encoder.layers.' + str(i) + '.self_attn.out_proj.tanh_weight_bias'].float()
        selfattn_outproj_tanh_bias_bias_en = model[
            'encoder.layers.' + str(i) + '.self_attn.out_proj.tanh_bias_bias'].float()
        selfattn_outproj_bias_en = selfattn_outproj_tanh_weight_bias_en * torch.tanh(selfattn_outproj_bias_en) + selfattn_outproj_tanh_bias_bias_en
        model_new['encoder.layers.' + str(i) + '.self_attn.out_proj.bias'] = selfattn_outproj_bias_en

        fc1_weight_en = model['encoder.layers.' + str(i) + '.fc1.weight'].float()
        fc1_tanh_weight_weight_en = model['encoder.layers.' + str(i) + '.fc1.tanh_weight_weight'].float()
        fc1_tanh_bias_weight_en = model['encoder.layers.' + str(i) + '.fc1.tanh_bias_weight'].float()
        fc1_weight_en = fc1_tanh_weight_weight_en * torch.tanh(fc1_weight_en) + fc1_tanh_bias_weight_en
        model_new['encoder.layers.' + str(i) + '.fc1.weight'] = fc1_weight_en

        fc1_bias_en = model['encoder.layers.' + str(i) + '.fc1.bias'].float()
        fc1_tanh_weight_bias_en = model['encoder.layers.' + str(i) + '.fc1.tanh_weight_bias'].float()
        fc1_tanh_bias_bias_en = model['encoder.layers.' + str(i) + '.fc1.tanh_bias_bias'].float()
        fc1_bias_en = fc1_tanh_weight_bias_en * torch.tanh(fc1_bias_en) + fc1_tanh_bias_bias_en
        model_new['encoder.layers.' + str(i) + '.fc1.bias'] = fc1_bias_en

        fc2_weight_en = model['encoder.layers.' + str(i) + '.fc2.weight'].float()
        fc2_tanh_weight_weight_en = model['encoder.layers.' + str(i) + '.fc2.tanh_weight_weight'].float()
        fc2_tanh_bias_weight_en = model['encoder.layers.' + str(i) + '.fc2.tanh_bias_weight'].float()
        fc2_weight_en = fc2_tanh_weight_weight_en * torch.tanh(fc2_weight_en) + fc2_tanh_bias_weight_en
        model_new['encoder.layers.' + str(i) + '.fc2.weight'] = fc2_weight_en

        fc2_bias_en = model['encoder.layers.' + str(i) + '.fc2.bias'].float()
        fc2_tanh_weight_bias_en = model['encoder.layers.' + str(i) + '.fc2.tanh_weight_bias'].float()
        fc2_tanh_bias_bias_en = model['encoder.layers.' + str(i) + '.fc2.tanh_bias_bias'].float()
        fc2_bias_en = fc2_tanh_weight_bias_en * torch.tanh(fc2_bias_en) + fc2_tanh_bias_bias_en
        model_new['encoder.layers.' + str(i) + '.fc2.bias'] = fc2_bias_en

        layernorm_0_weight = model['encoder.layers.' + str(i) + '.layer_norms.0.weight'].float()
        layernorm_0_tanh_weight_weight = model['encoder.layers.' + str(i) + '.layer_norms.0.tanh_weight_weight'].float()
        layernorm_0_tanh_bias_weight = model['encoder.layers.' + str(i) + '.layer_norms.0.tanh_bias_weight'].float()
        layernorm_0_weight = layernorm_0_tanh_weight_weight * torch.tanh(layernorm_0_weight) + layernorm_0_tanh_bias_weight
        model_new['encoder.layers.' + str(i) + '.layer_norms.0.weight'] = layernorm_0_weight

        layernorm_0_bias = model['encoder.layers.' + str(i) + '.layer_norms.0.bias'].float()
        layernorm_0_tanh_weight_bias = model['encoder.layers.' + str(i) + '.layer_norms.0.tanh_weight_bias'].float()
        layernorm_0_tanh_bias_bias = model['encoder.layers.' + str(i) + '.layer_norms.0.tanh_bias_bias'].float()
        layernorm_0_bias = layernorm_0_tanh_weight_bias * torch.tanh(layernorm_0_bias) + layernorm_0_tanh_bias_bias
        model_new['encoder.layers.' + str(i) + '.layer_norms.0.bias'] = layernorm_0_bias

        layernorm_1_weight = model['encoder.layers.' + str(i) + '.layer_norms.1.weight'].float()
        layernorm_1_tanh_weight_weight = model['encoder.layers.' + str(i) + '.layer_norms.1.tanh_weight_weight'].float()
        layernorm_1_tanh_bias_weight = model['encoder.layers.' + str(i) + '.layer_norms.1.tanh_bias_weight'].float()
        layernorm_1_weight = layernorm_1_tanh_weight_weight * torch.tanh(layernorm_1_weight) + layernorm_1_tanh_bias_weight
        model_new['encoder.layers.' + str(i) + '.layer_norms.1.weight'] = layernorm_1_weight

        layernorm_1_bias = model['encoder.layers.' + str(i) + '.layer_norms.1.bias'].float()
        layernorm_1_tanh_weight_bias = model['encoder.layers.' + str(i) + '.layer_norms.1.tanh_weight_bias'].float()
        layernorm_1_tanh_bias_bias = model['encoder.layers.' + str(i) + '.layer_norms.1.tanh_bias_bias'].float()
        layernorm_1_bias = layernorm_1_tanh_weight_bias * torch.tanh(layernorm_1_bias) + layernorm_1_tanh_bias_bias
        model_new['encoder.layers.' + str(i) + '.layer_norms.1.bias'] = layernorm_1_bias

    for i in range(args.decoder_layers):
        # decoder
        selfattn_q_weight = model['decoder.layers.' + str(i) + '.self_attn.q_weight'].float()
        selfattn_k_weight = model['decoder.layers.' + str(i) + '.self_attn.k_weight'].float()
        selfattn_v_weight = model['decoder.layers.' + str(i) + '.self_attn.v_weight'].float()
        selfattn_wd_q_weight = model['decoder.layers.' + str(i) + '.self_attn.wd_q_weight'].float()
        selfattn_wd_k_weight = model['decoder.layers.' + str(i) + '.self_attn.wd_k_weight'].float()
        selfattn_wd_v_weight = model['decoder.layers.' + str(i) + '.self_attn.wd_v_weight'].float()
        selfattn_out_wd_q_weight = model['decoder.layers.' + str(i) + '.self_attn.out_wd_q_weight'].float()
        selfattn_out_wd_k_weight = model['decoder.layers.' + str(i) + '.self_attn.out_wd_k_weight'].float()
        selfattn_out_wd_v_weight = model['decoder.layers.' + str(i) + '.self_attn.out_wd_v_weight'].float()
        selfattn_in_wd_q_weight = model['decoder.layers.' + str(i) + '.self_attn.in_wd_q_weight'].float()
        selfattn_in_wd_k_weight = model['decoder.layers.' + str(i) + '.self_attn.in_wd_k_weight'].float()
        selfattn_in_wd_v_weight = model['decoder.layers.' + str(i) + '.self_attn.in_wd_v_weight'].float()
        selfattn_ly_wd_q_weight = model['decoder.layers.' + str(i) + '.self_attn.ly_wd_q_weight'].float()
        selfattn_ly_wd_k_weight = model['decoder.layers.' + str(i) + '.self_attn.ly_wd_k_weight'].float()
        selfattn_ly_wd_v_weight = model['decoder.layers.' + str(i) + '.self_attn.ly_wd_v_weight'].float()
        selfattn_tanh_weight_q_weight = model['decoder.layers.' + str(i) + '.self_attn.tanh_weight_q_weight'].float()
        selfattn_tanh_weight_k_weight = model['decoder.layers.' + str(i) + '.self_attn.tanh_weight_k_weight'].float()
        selfattn_tanh_weight_v_weight = model['decoder.layers.' + str(i) + '.self_attn.tanh_weight_v_weight'].float()
        selfattn_tanh_bias_q_weight = model['decoder.layers.' + str(i) + '.self_attn.tanh_bias_q_weight'].float()
        selfattn_tanh_bias_k_weight = model['decoder.layers.' + str(i) + '.self_attn.tanh_bias_k_weight'].float()
        selfattn_tanh_bias_v_weight = model['decoder.layers.' + str(i) + '.self_attn.tanh_bias_v_weight'].float()
        selfattn_q_weight = torch.transpose(selfattn_q_weight, 0, 2)
        selfattn_k_weight = torch.transpose(selfattn_k_weight, 0, 2)
        selfattn_v_weight = torch.transpose(selfattn_v_weight, 0, 2)
        selfattn_q_weight = torch.matmul(selfattn_q_weight, selfattn_out_wd_q_weight)
        selfattn_k_weight = torch.matmul(selfattn_k_weight, selfattn_out_wd_k_weight)
        selfattn_v_weight = torch.matmul(selfattn_v_weight, selfattn_out_wd_v_weight)
        selfattn_q_weight = torch.transpose(selfattn_q_weight, 0, 2)
        selfattn_k_weight = torch.transpose(selfattn_k_weight, 0, 2)
        selfattn_v_weight = torch.transpose(selfattn_v_weight, 0, 2)
        selfattn_q_weight = torch.matmul(selfattn_q_weight, selfattn_ly_wd_q_weight)
        selfattn_k_weight = torch.matmul(selfattn_k_weight, selfattn_ly_wd_k_weight)
        selfattn_v_weight = torch.matmul(selfattn_v_weight, selfattn_ly_wd_v_weight)
        selfattn_q_weight = torch.transpose(selfattn_q_weight, 1, 2)
        selfattn_k_weight = torch.transpose(selfattn_k_weight, 1, 2)
        selfattn_v_weight = torch.transpose(selfattn_v_weight, 1, 2)
        selfattn_q_weight = torch.matmul(selfattn_q_weight, selfattn_in_wd_q_weight)
        selfattn_k_weight = torch.matmul(selfattn_k_weight, selfattn_in_wd_k_weight)
        selfattn_v_weight = torch.matmul(selfattn_v_weight, selfattn_in_wd_v_weight)
        selfattn_q_weight = torch.transpose(selfattn_q_weight, 1, 2)
        selfattn_k_weight = torch.transpose(selfattn_k_weight, 1, 2)
        selfattn_v_weight = torch.transpose(selfattn_v_weight, 1, 2)
        selfattn_q_weight = torch.matmul(selfattn_q_weight, selfattn_wd_q_weight)
        selfattn_k_weight = torch.matmul(selfattn_k_weight, selfattn_wd_k_weight)
        selfattn_v_weight = torch.matmul(selfattn_v_weight, selfattn_wd_v_weight)
        selfattn_q_weight = selfattn_q_weight.squeeze(-1)
        selfattn_k_weight = selfattn_k_weight.squeeze(-1)
        selfattn_v_weight = selfattn_v_weight.squeeze(-1)
        selfattn_q_weight = selfattn_tanh_weight_q_weight * torch.tanh(selfattn_q_weight) + selfattn_tanh_bias_q_weight
        selfattn_k_weight = selfattn_tanh_weight_k_weight * torch.tanh(selfattn_k_weight) + selfattn_tanh_bias_k_weight
        selfattn_v_weight = selfattn_tanh_weight_v_weight * torch.tanh(selfattn_v_weight) + selfattn_tanh_bias_v_weight
        selfattn_inproj_weight = torch.cat((selfattn_q_weight, selfattn_k_weight, selfattn_v_weight), dim=0)
        model_new['decoder.layers.' + str(i) + '.self_attn.in_proj_weight'] = selfattn_inproj_weight

        selfattn_q_bias = model['decoder.layers.' + str(i) + '.self_attn.q_bias'].float()
        selfattn_k_bias = model['decoder.layers.' + str(i) + '.self_attn.k_bias'].float()
        selfattn_v_bias = model['decoder.layers.' + str(i) + '.self_attn.v_bias'].float()
        selfattn_wd_q_bias = model['decoder.layers.' + str(i) + '.self_attn.wd_q_bias'].float()
        selfattn_wd_k_bias = model['decoder.layers.' + str(i) + '.self_attn.wd_k_bias'].float()
        selfattn_wd_v_bias = model['decoder.layers.' + str(i) + '.self_attn.wd_v_bias'].float()
        selfattn_out_wd_q_bias = model['decoder.layers.' + str(i) + '.self_attn.out_wd_q_bias'].float()
        selfattn_out_wd_k_bias = model['decoder.layers.' + str(i) + '.self_attn.out_wd_k_bias'].float()
        selfattn_out_wd_v_bias = model['decoder.layers.' + str(i) + '.self_attn.out_wd_v_bias'].float()
        selfattn_ly_wd_q_bias = model['decoder.layers.' + str(i) + '.self_attn.ly_wd_q_bias'].float()
        selfattn_ly_wd_k_bias = model['decoder.layers.' + str(i) + '.self_attn.ly_wd_k_bias'].float()
        selfattn_ly_wd_v_bias = model['decoder.layers.' + str(i) + '.self_attn.ly_wd_v_bias'].float()
        selfattn_tanh_weight_q_bias = model['decoder.layers.' + str(i) + '.self_attn.tanh_weight_q_bias'].float()
        selfattn_tanh_weight_k_bias = model['decoder.layers.' + str(i) + '.self_attn.tanh_weight_k_bias'].float()
        selfattn_tanh_weight_v_bias = model['decoder.layers.' + str(i) + '.self_attn.tanh_weight_v_bias'].float()
        selfattn_tanh_bias_q_bias = model['decoder.layers.' + str(i) + '.self_attn.tanh_bias_q_bias'].float()
        selfattn_tanh_bias_k_bias = model['decoder.layers.' + str(i) + '.self_attn.tanh_bias_k_bias'].float()
        selfattn_tanh_bias_v_bias = model['decoder.layers.' + str(i) + '.self_attn.tanh_bias_v_bias'].float()
        selfattn_q_bias = torch.transpose(selfattn_q_bias, 0, 1)
        selfattn_k_bias = torch.transpose(selfattn_k_bias, 0, 1)
        selfattn_v_bias = torch.transpose(selfattn_v_bias, 0, 1)
        selfattn_q_bias = torch.matmul(selfattn_q_bias, selfattn_out_wd_q_bias)
        selfattn_k_bias = torch.matmul(selfattn_k_bias, selfattn_out_wd_k_bias)
        selfattn_v_bias = torch.matmul(selfattn_v_bias, selfattn_out_wd_v_bias)
        selfattn_q_bias = torch.transpose(selfattn_q_bias, 0, 1)
        selfattn_k_bias = torch.transpose(selfattn_k_bias, 0, 1)
        selfattn_v_bias = torch.transpose(selfattn_v_bias, 0, 1)
        selfattn_q_bias = torch.matmul(selfattn_q_bias, selfattn_ly_wd_q_bias)
        selfattn_k_bias = torch.matmul(selfattn_k_bias, selfattn_ly_wd_k_bias)
        selfattn_v_bias = torch.matmul(selfattn_v_bias, selfattn_ly_wd_v_bias)
        selfattn_q_bias = torch.matmul(selfattn_q_bias, selfattn_wd_q_bias)
        selfattn_k_bias = torch.matmul(selfattn_k_bias, selfattn_wd_k_bias)
        selfattn_v_bias = torch.matmul(selfattn_v_bias, selfattn_wd_v_bias)
        selfattn_q_bias = selfattn_q_bias.squeeze(-1)
        selfattn_k_bias = selfattn_k_bias.squeeze(-1)
        selfattn_v_bias = selfattn_v_bias.squeeze(-1)
        selfattn_q_bias = selfattn_tanh_weight_q_bias * torch.tanh(selfattn_q_bias) + selfattn_tanh_bias_q_bias
        selfattn_k_bias = selfattn_tanh_weight_k_bias * torch.tanh(selfattn_k_bias) + selfattn_tanh_bias_k_bias
        selfattn_v_bias = selfattn_tanh_weight_v_bias * torch.tanh(selfattn_v_bias) + selfattn_tanh_bias_v_bias
        selfattn_inproj_bias = torch.cat((selfattn_q_bias, selfattn_k_bias, selfattn_v_bias), dim=0)
        model_new['decoder.layers.' + str(i) + '.self_attn.in_proj_bias'] = selfattn_inproj_bias

        selfattn_outproj_weight = model['decoder.layers.' + str(i) + '.self_attn.out_proj.weight'].float()
        selfattn_outproj_wd_weight = model['decoder.layers.' + str(i) + '.self_attn.out_proj.wd_weight'].float()
        selfattn_outproj_out_wd_weight = model['decoder.layers.' + str(i) + '.self_attn.out_proj.out_wd_weight'].float()
        selfattn_outproj_in_wd_weight = model['decoder.layers.' + str(i) + '.self_attn.out_proj.in_wd_weight'].float()
        selfattn_outproj_ly_wd_weight = model['decoder.layers.' + str(i) + '.self_attn.out_proj.ly_wd_weight'].float()
        selfattn_outproj_tanh_weight_weight = model[
            'decoder.layers.' + str(i) + '.self_attn.out_proj.tanh_weight_weight'].float()
        selfattn_outproj_tanh_bias_weight = model[
            'decoder.layers.' + str(i) + '.self_attn.out_proj.tanh_bias_weight'].float()
        selfattn_outproj_weight = torch.transpose(selfattn_outproj_weight, 0, 2)
        selfattn_outproj_weight = torch.matmul(selfattn_outproj_weight, selfattn_outproj_out_wd_weight)
        selfattn_outproj_weight = torch.transpose(selfattn_outproj_weight, 0, 2)
        selfattn_outproj_weight = torch.matmul(selfattn_outproj_weight, selfattn_outproj_ly_wd_weight)
        selfattn_outproj_weight = torch.transpose(selfattn_outproj_weight, 1, 2)
        selfattn_outproj_weight = torch.matmul(selfattn_outproj_weight, selfattn_outproj_in_wd_weight)
        selfattn_outproj_weight = torch.transpose(selfattn_outproj_weight, 1, 2)
        selfattn_outproj_weight = torch.matmul(selfattn_outproj_weight, selfattn_outproj_wd_weight)
        selfattn_outproj_weight = selfattn_outproj_weight.squeeze(-1)
        selfattn_outproj_weight = selfattn_outproj_tanh_weight_weight * torch.tanh(
            selfattn_outproj_weight) + selfattn_outproj_tanh_bias_weight
        model_new['decoder.layers.' + str(i) + '.self_attn.out_proj.weight'] = selfattn_outproj_weight

        selfattn_outproj_bias = model['decoder.layers.' + str(i) + '.self_attn.out_proj.bias'].float()
        selfattn_outproj_wd_bias = model['decoder.layers.' + str(i) + '.self_attn.out_proj.wd_bias'].float()
        selfattn_outproj_out_wd_bias = model['decoder.layers.' + str(i) + '.self_attn.out_proj.out_wd_bias'].float()
        selfattn_outproj_ly_wd_bias = model['decoder.layers.' + str(i) + '.self_attn.out_proj.ly_wd_bias'].float()
        selfattn_outproj_tanh_weight_bias = model[
            'decoder.layers.' + str(i) + '.self_attn.out_proj.tanh_weight_bias'].float()
        selfattn_outproj_tanh_bias_bias = model[
            'decoder.layers.' + str(i) + '.self_attn.out_proj.tanh_bias_bias'].float()
        selfattn_outproj_bias = torch.transpose(selfattn_outproj_bias, 0, 1)
        selfattn_outproj_bias = torch.matmul(selfattn_outproj_bias, selfattn_outproj_out_wd_bias)
        selfattn_outproj_bias = torch.transpose(selfattn_outproj_bias, 0, 1)
        selfattn_outproj_bias = torch.matmul(selfattn_outproj_bias, selfattn_outproj_ly_wd_bias)
        selfattn_outproj_bias = torch.matmul(selfattn_outproj_bias, selfattn_outproj_wd_bias)
        selfattn_outproj_bias = selfattn_outproj_bias.squeeze(-1)
        selfattn_outproj_bias = selfattn_outproj_tanh_weight_bias * torch.tanh(
            selfattn_outproj_bias) + selfattn_outproj_tanh_bias_bias
        model_new['decoder.layers.' + str(i) + '.self_attn.out_proj.bias'] = selfattn_outproj_bias

        selfattn_layernorm_weight = model['decoder.layers.' + str(i) + '.self_attn_layer_norm.weight'].float()
        selfattn_layernorm_wd_weight = model['decoder.layers.' + str(i) + '.self_attn_layer_norm.wd_weight'].float()
        selfattn_layernorm_out_wd_weight = model[
            'decoder.layers.' + str(i) + '.self_attn_layer_norm.out_wd_weight'].float()
        selfattn_layernorm_ly_wd_weight = model[
            'decoder.layers.' + str(i) + '.self_attn_layer_norm.ly_wd_weight'].float()
        selfattn_layernorm_tanh_weight_weight = model[
            'decoder.layers.' + str(i) + '.self_attn_layer_norm.tanh_weight_weight'].float()
        selfattn_layernorm_tanh_bias_weight = model[
            'decoder.layers.' + str(i) + '.self_attn_layer_norm.tanh_bias_weight'].float()
        selfattn_layernorm_weight = torch.transpose(selfattn_layernorm_weight, 0, 1)
        selfattn_layernorm_weight = torch.matmul(selfattn_layernorm_weight, selfattn_layernorm_out_wd_weight)
        selfattn_layernorm_weight = torch.transpose(selfattn_layernorm_weight, 0, 1)
        selfattn_layernorm_weight = torch.matmul(selfattn_layernorm_weight, selfattn_layernorm_ly_wd_weight)
        selfattn_layernorm_weight = torch.matmul(selfattn_layernorm_weight, selfattn_layernorm_wd_weight)
        selfattn_layernorm_weight = selfattn_layernorm_weight.squeeze(-1)
        selfattn_layernorm_weight = selfattn_layernorm_tanh_weight_weight * torch.tanh(
            selfattn_layernorm_weight) + selfattn_layernorm_tanh_bias_weight
        model_new['decoder.layers.' + str(i) + '.self_attn_layer_norm.weight'] = selfattn_layernorm_weight

        selfattn_layernorm_bias = model['decoder.layers.' + str(i) + '.self_attn_layer_norm.bias'].float()
        selfattn_layernorm_wd_bias = model['decoder.layers.' + str(i) + '.self_attn_layer_norm.wd_bias'].float()
        selfattn_layernorm_out_wd_bias = model['decoder.layers.' + str(i) + '.self_attn_layer_norm.out_wd_bias'].float()
        selfattn_layernorm_ly_wd_bias = model['decoder.layers.' + str(i) + '.self_attn_layer_norm.ly_wd_bias'].float()
        selfattn_layernorm_tanh_weight_bias = model[
            'decoder.layers.' + str(i) + '.self_attn_layer_norm.tanh_weight_bias'].float()
        selfattn_layernorm_tanh_bias_bias = model[
            'decoder.layers.' + str(i) + '.self_attn_layer_norm.tanh_bias_bias'].float()
        selfattn_layernorm_bias = torch.transpose(selfattn_layernorm_bias, 0, 1)
        selfattn_layernorm_bias = torch.matmul(selfattn_layernorm_bias, selfattn_layernorm_out_wd_bias)
        selfattn_layernorm_bias = torch.transpose(selfattn_layernorm_bias, 0, 1)
        selfattn_layernorm_bias = torch.matmul(selfattn_layernorm_bias, selfattn_layernorm_ly_wd_bias)
        selfattn_layernorm_bias = torch.matmul(selfattn_layernorm_bias, selfattn_layernorm_wd_bias)
        selfattn_layernorm_bias = selfattn_layernorm_bias.squeeze(-1)
        selfattn_layernorm_bias = selfattn_layernorm_tanh_weight_bias * torch.tanh(
            selfattn_layernorm_bias) + selfattn_layernorm_tanh_bias_bias
        model_new['decoder.layers.' + str(i) + '.self_attn_layer_norm.bias'] = selfattn_layernorm_bias

        encoderattn_layernorm_weight = model['decoder.layers.' + str(i) + '.encoder_attn_layer_norm.weight'].float()
        encoderattn_layernorm_wd_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn_layer_norm.wd_weight'].float()
        encoderattn_layernorm_out_wd_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn_layer_norm.out_wd_weight'].float()
        encoderattn_layernorm_ly_wd_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn_layer_norm.ly_wd_weight'].float()
        encoderattn_layernorm_tanh_weight_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn_layer_norm.tanh_weight_weight'].float()
        encoderattn_layernorm_tanh_bias_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn_layer_norm.tanh_bias_weight'].float()
        encoderattn_layernorm_weight = torch.transpose(encoderattn_layernorm_weight, 0, 1)
        encoderattn_layernorm_weight = torch.matmul(encoderattn_layernorm_weight, encoderattn_layernorm_out_wd_weight)
        encoderattn_layernorm_weight = torch.transpose(encoderattn_layernorm_weight, 0, 1)
        encoderattn_layernorm_weight = torch.matmul(encoderattn_layernorm_weight, encoderattn_layernorm_ly_wd_weight)
        encoderattn_layernorm_weight = torch.matmul(encoderattn_layernorm_weight, encoderattn_layernorm_wd_weight)
        encoderattn_layernorm_weight = encoderattn_layernorm_weight.squeeze(-1)
        encoderattn_layernorm_weight = encoderattn_layernorm_tanh_weight_weight * torch.tanh(
            encoderattn_layernorm_weight) + encoderattn_layernorm_tanh_bias_weight
        model_new['decoder.layers.' + str(i) + '.encoder_attn_layer_norm.weight'] = encoderattn_layernorm_weight

        encoderattn_layernorm_bias = model['decoder.layers.' + str(i) + '.encoder_attn_layer_norm.bias'].float()
        encoderattn_layernorm_wd_bias = model['decoder.layers.' + str(i) + '.encoder_attn_layer_norm.wd_bias'].float()
        encoderattn_layernorm_out_wd_bias = model[
            'decoder.layers.' + str(i) + '.encoder_attn_layer_norm.out_wd_bias'].float()
        encoderattn_layernorm_ly_wd_bias = model[
            'decoder.layers.' + str(i) + '.encoder_attn_layer_norm.ly_wd_bias'].float()
        encoderattn_layernorm_tanh_weight_bias = model[
            'decoder.layers.' + str(i) + '.encoder_attn_layer_norm.tanh_weight_bias'].float()
        encoderattn_layernorm_tanh_bias_bias = model[
            'decoder.layers.' + str(i) + '.encoder_attn_layer_norm.tanh_bias_bias'].float()
        encoderattn_layernorm_bias = torch.transpose(encoderattn_layernorm_bias, 0, 1)
        encoderattn_layernorm_bias = torch.matmul(encoderattn_layernorm_bias, encoderattn_layernorm_out_wd_bias)
        encoderattn_layernorm_bias = torch.transpose(encoderattn_layernorm_bias, 0, 1)
        encoderattn_layernorm_bias = torch.matmul(encoderattn_layernorm_bias, encoderattn_layernorm_ly_wd_bias)
        encoderattn_layernorm_bias = torch.matmul(encoderattn_layernorm_bias, encoderattn_layernorm_wd_bias)
        encoderattn_layernorm_bias = encoderattn_layernorm_bias.squeeze(-1)
        encoderattn_layernorm_bias = encoderattn_layernorm_tanh_weight_bias * torch.tanh(
            encoderattn_layernorm_bias) + encoderattn_layernorm_tanh_bias_bias
        model_new['decoder.layers.' + str(i) + '.encoder_attn_layer_norm.bias'] = encoderattn_layernorm_bias

        fc1_weight = model['decoder.layers.' + str(i) + '.fc1.weight'].float()
        fc1_wd_weight = model['decoder.layers.' + str(i) + '.fc1.wd_weight'].float()
        fc1_out_wd_weight = model['decoder.layers.' + str(i) + '.fc1.out_wd_weight'].float()
        fc1_in_wd_weight = model['decoder.layers.' + str(i) + '.fc1.in_wd_weight'].float()
        fc1_ly_wd_weight = model['decoder.layers.' + str(i) + '.fc1.ly_wd_weight'].float()
        fc1_tanh_weight_weight = model['decoder.layers.' + str(i) + '.fc1.tanh_weight_weight'].float()
        fc1_tanh_bias_weight = model['decoder.layers.' + str(i) + '.fc1.tanh_bias_weight'].float()
        fc1_weight = torch.transpose(fc1_weight, 0, 2)
        fc1_weight = torch.matmul(fc1_weight, fc1_out_wd_weight)
        fc1_weight = torch.transpose(fc1_weight, 0, 2)
        fc1_weight = torch.matmul(fc1_weight, fc1_ly_wd_weight)
        fc1_weight = torch.transpose(fc1_weight, 1, 2)
        fc1_weight = torch.matmul(fc1_weight, fc1_in_wd_weight)
        fc1_weight = torch.transpose(fc1_weight, 1, 2)
        fc1_weight = torch.matmul(fc1_weight, fc1_wd_weight)
        fc1_weight = fc1_weight.squeeze(-1)
        fc1_weight = fc1_tanh_weight_weight * torch.tanh(fc1_weight) + fc1_tanh_bias_weight
        model_new['decoder.layers.' + str(i) + '.fc1.weight'] = fc1_weight

        fc1_bias = model['decoder.layers.' + str(i) + '.fc1.bias'].float()
        fc1_wd_bias = model['decoder.layers.' + str(i) + '.fc1.wd_bias'].float()
        fc1_out_wd_bias = model['decoder.layers.' + str(i) + '.fc1.out_wd_bias'].float()
        fc1_ly_wd_bias = model['decoder.layers.' + str(i) + '.fc1.ly_wd_bias'].float()
        fc1_tanh_weight_bias = model['decoder.layers.' + str(i) + '.fc1.tanh_weight_bias'].float()
        fc1_tanh_bias_bias = model['decoder.layers.' + str(i) + '.fc1.tanh_bias_bias'].float()
        fc1_bias = torch.transpose(fc1_bias, 0, 1)
        fc1_bias = torch.matmul(fc1_bias, fc1_out_wd_bias)
        fc1_bias = torch.transpose(fc1_bias, 0, 1)
        fc1_bias = torch.matmul(fc1_bias, fc1_ly_wd_bias)
        fc1_bias = torch.matmul(fc1_bias, fc1_wd_bias)
        fc1_bias = fc1_bias.squeeze(-1)
        fc1_bias = fc1_tanh_weight_bias * torch.tanh(fc1_bias) + fc1_tanh_bias_bias
        model_new['decoder.layers.' + str(i) + '.fc1.bias'] = fc1_bias

        fc2_weight = model['decoder.layers.' + str(i) + '.fc2.weight'].float()
        fc2_wd_weight = model['decoder.layers.' + str(i) + '.fc2.wd_weight'].float()
        fc2_out_wd_weight = model['decoder.layers.' + str(i) + '.fc2.out_wd_weight'].float()
        fc2_in_wd_weight = model['decoder.layers.' + str(i) + '.fc2.in_wd_weight'].float()
        fc2_ly_wd_weight = model['decoder.layers.' + str(i) + '.fc2.ly_wd_weight'].float()
        fc2_tanh_weight_weight = model['decoder.layers.' + str(i) + '.fc2.tanh_weight_weight'].float()
        fc2_tanh_bias_weight = model['decoder.layers.' + str(i) + '.fc2.tanh_bias_weight'].float()
        fc2_weight = torch.transpose(fc2_weight, 0, 2)
        fc2_weight = torch.matmul(fc2_weight, fc2_out_wd_weight)
        fc2_weight = torch.transpose(fc2_weight, 0, 2)
        fc2_weight = torch.matmul(fc2_weight, fc2_ly_wd_weight)
        fc2_weight = torch.transpose(fc2_weight, 1, 2)
        fc2_weight = torch.matmul(fc2_weight, fc2_in_wd_weight)
        fc2_weight = torch.transpose(fc2_weight, 1, 2)
        fc2_weight = torch.matmul(fc2_weight, fc2_wd_weight)
        fc2_weight = fc2_weight.squeeze(-1)
        fc2_weight = fc2_tanh_weight_weight * torch.tanh(fc2_weight) + fc2_tanh_bias_weight
        model_new['decoder.layers.' + str(i) + '.fc2.weight'] = fc2_weight

        fc2_bias = model['decoder.layers.' + str(i) + '.fc2.bias'].float()
        fc2_wd_bias = model['decoder.layers.' + str(i) + '.fc2.wd_bias'].float()
        fc2_out_wd_bias = model['decoder.layers.' + str(i) + '.fc2.out_wd_bias'].float()
        fc2_ly_wd_bias = model['decoder.layers.' + str(i) + '.fc2.ly_wd_bias'].float()
        fc2_tanh_weight_bias = model['decoder.layers.' + str(i) + '.fc2.tanh_weight_bias'].float()
        fc2_tanh_bias_bias = model['decoder.layers.' + str(i) + '.fc2.tanh_bias_bias'].float()
        fc2_bias = torch.transpose(fc2_bias, 0, 1)
        fc2_bias = torch.matmul(fc2_bias, fc2_out_wd_bias)
        fc2_bias = torch.transpose(fc2_bias, 0, 1)
        fc2_bias = torch.matmul(fc2_bias, fc2_ly_wd_bias)
        fc2_bias = torch.matmul(fc2_bias, fc2_wd_bias)
        fc2_bias = fc2_bias.squeeze(-1)
        fc2_bias = fc2_tanh_weight_bias * torch.tanh(fc2_bias) + fc2_tanh_bias_bias
        model_new['decoder.layers.' + str(i) + '.fc2.bias'] = fc2_bias

        final_layernorm_weight = model['decoder.layers.' + str(i) + '.final_layer_norm.weight'].float()
        final_layernorm_wd_weight = model['decoder.layers.' + str(i) + '.final_layer_norm.wd_weight'].float()
        final_layernorm_out_wd_weight = model['decoder.layers.' + str(i) + '.final_layer_norm.out_wd_weight'].float()
        final_layernorm_ly_wd_weight = model['decoder.layers.' + str(i) + '.final_layer_norm.ly_wd_weight'].float()
        final_layernorm_tanh_weight_weight = model[
            'decoder.layers.' + str(i) + '.final_layer_norm.tanh_weight_weight'].float()
        final_layernorm_tanh_bias_weight = model[
            'decoder.layers.' + str(i) + '.final_layer_norm.tanh_bias_weight'].float()
        final_layernorm_weight = torch.transpose(final_layernorm_weight, 0, 1)
        final_layernorm_weight = torch.matmul(final_layernorm_weight, final_layernorm_out_wd_weight)
        final_layernorm_weight = torch.transpose(final_layernorm_weight, 0, 1)
        final_layernorm_weight = torch.matmul(final_layernorm_weight, final_layernorm_ly_wd_weight)
        final_layernorm_weight = torch.matmul(final_layernorm_weight, final_layernorm_wd_weight)
        final_layernorm_weight = final_layernorm_weight.squeeze(-1)
        final_layernorm_weight = final_layernorm_tanh_weight_weight * torch.tanh(
            final_layernorm_weight) + final_layernorm_tanh_bias_weight
        model_new['decoder.layers.' + str(i) + '.final_layer_norm.weight'] = final_layernorm_weight

        final_layernorm_bias = model['decoder.layers.' + str(i) + '.final_layer_norm.bias'].float()
        final_layernorm_wd_bias = model['decoder.layers.' + str(i) + '.final_layer_norm.wd_bias'].float()
        final_layernorm_out_wd_bias = model['decoder.layers.' + str(i) + '.final_layer_norm.out_wd_bias'].float()
        final_layernorm_ly_wd_bias = model['decoder.layers.' + str(i) + '.final_layer_norm.ly_wd_bias'].float()
        final_layernorm_tanh_weight_bias = model[
            'decoder.layers.' + str(i) + '.final_layer_norm.tanh_weight_bias'].float()
        final_layernorm_tanh_bias_bias = model['decoder.layers.' + str(i) + '.final_layer_norm.tanh_bias_bias'].float()
        final_layernorm_bias = torch.transpose(final_layernorm_bias, 0, 1)
        final_layernorm_bias = torch.matmul(final_layernorm_bias, final_layernorm_out_wd_bias)
        final_layernorm_bias = torch.transpose(final_layernorm_bias, 0, 1)
        final_layernorm_bias = torch.matmul(final_layernorm_bias, final_layernorm_ly_wd_bias)
        final_layernorm_bias = torch.matmul(final_layernorm_bias, final_layernorm_wd_bias)
        final_layernorm_bias = final_layernorm_bias.squeeze(-1)
        final_layernorm_bias = final_layernorm_tanh_weight_bias * torch.tanh(
            final_layernorm_bias) + final_layernorm_tanh_bias_bias
        model_new['decoder.layers.' + str(i) + '.final_layer_norm.bias'] = final_layernorm_bias

        encoderattn_q_weight = model['decoder.layers.' + str(i) + '.encoder_attn.q_weight'].float()
        encoderattn_k_weight = model['decoder.layers.' + str(i) + '.encoder_attn.k_weight'].float()
        encoderattn_v_weight = model['decoder.layers.' + str(i) + '.encoder_attn.v_weight'].float()
        encoderattn_wd_q_weight = model['decoder.layers.' + str(i) + '.encoder_attn.wd_q_weight'].float()
        encoderattn_wd_k_weight = model['decoder.layers.' + str(i) + '.encoder_attn.wd_k_weight'].float()
        encoderattn_wd_v_weight = model['decoder.layers.' + str(i) + '.encoder_attn.wd_v_weight'].float()
        encoderattn_out_wd_q_weight = model['decoder.layers.' + str(i) + '.encoder_attn.out_wd_q_weight'].float()
        encoderattn_out_wd_k_weight = model['decoder.layers.' + str(i) + '.encoder_attn.out_wd_k_weight'].float()
        encoderattn_out_wd_v_weight = model['decoder.layers.' + str(i) + '.encoder_attn.out_wd_v_weight'].float()
        encoderattn_in_wd_q_weight = model['decoder.layers.' + str(i) + '.encoder_attn.in_wd_q_weight'].float()
        encoderattn_in_wd_k_weight = model['decoder.layers.' + str(i) + '.encoder_attn.in_wd_k_weight'].float()
        encoderattn_in_wd_v_weight = model['decoder.layers.' + str(i) + '.encoder_attn.in_wd_v_weight'].float()
        encoderattn_ly_wd_q_weight = model['decoder.layers.' + str(i) + '.encoder_attn.ly_wd_q_weight'].float()
        encoderattn_ly_wd_k_weight = model['decoder.layers.' + str(i) + '.encoder_attn.ly_wd_k_weight'].float()
        encoderattn_ly_wd_v_weight = model['decoder.layers.' + str(i) + '.encoder_attn.ly_wd_v_weight'].float()
        encoderattn_tanh_weight_q_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn.tanh_weight_q_weight'].float()
        encoderattn_tanh_weight_k_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn.tanh_weight_k_weight'].float()
        encoderattn_tanh_weight_v_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn.tanh_weight_v_weight'].float()
        encoderattn_tanh_bias_q_weight = model['decoder.layers.' + str(i) + '.encoder_attn.tanh_bias_q_weight'].float()
        encoderattn_tanh_bias_k_weight = model['decoder.layers.' + str(i) + '.encoder_attn.tanh_bias_k_weight'].float()
        encoderattn_tanh_bias_v_weight = model['decoder.layers.' + str(i) + '.encoder_attn.tanh_bias_v_weight'].float()
        encoderattn_q_weight = torch.transpose(encoderattn_q_weight, 0, 2)
        encoderattn_k_weight = torch.transpose(encoderattn_k_weight, 0, 2)
        encoderattn_v_weight = torch.transpose(encoderattn_v_weight, 0, 2)
        encoderattn_q_weight = torch.matmul(encoderattn_q_weight, encoderattn_out_wd_q_weight)
        encoderattn_k_weight = torch.matmul(encoderattn_k_weight, encoderattn_out_wd_k_weight)
        encoderattn_v_weight = torch.matmul(encoderattn_v_weight, encoderattn_out_wd_v_weight)
        encoderattn_q_weight = torch.transpose(encoderattn_q_weight, 0, 2)
        encoderattn_k_weight = torch.transpose(encoderattn_k_weight, 0, 2)
        encoderattn_v_weight = torch.transpose(encoderattn_v_weight, 0, 2)
        encoderattn_q_weight = torch.matmul(encoderattn_q_weight, encoderattn_ly_wd_q_weight)
        encoderattn_k_weight = torch.matmul(encoderattn_k_weight, encoderattn_ly_wd_k_weight)
        encoderattn_v_weight = torch.matmul(encoderattn_v_weight, encoderattn_ly_wd_v_weight)
        encoderattn_q_weight = torch.transpose(encoderattn_q_weight, 1, 2)
        encoderattn_k_weight = torch.transpose(encoderattn_k_weight, 1, 2)
        encoderattn_v_weight = torch.transpose(encoderattn_v_weight, 1, 2)
        encoderattn_q_weight = torch.matmul(encoderattn_q_weight, encoderattn_in_wd_q_weight)
        encoderattn_k_weight = torch.matmul(encoderattn_k_weight, encoderattn_in_wd_k_weight)
        encoderattn_v_weight = torch.matmul(encoderattn_v_weight, encoderattn_in_wd_v_weight)
        encoderattn_q_weight = torch.transpose(encoderattn_q_weight, 1, 2)
        encoderattn_k_weight = torch.transpose(encoderattn_k_weight, 1, 2)
        encoderattn_v_weight = torch.transpose(encoderattn_v_weight, 1, 2)
        encoderattn_q_weight = torch.matmul(encoderattn_q_weight, encoderattn_wd_q_weight)
        encoderattn_k_weight = torch.matmul(encoderattn_k_weight, encoderattn_wd_k_weight)
        encoderattn_v_weight = torch.matmul(encoderattn_v_weight, encoderattn_wd_v_weight)
        encoderattn_q_weight = encoderattn_q_weight.squeeze(-1)
        encoderattn_k_weight = encoderattn_k_weight.squeeze(-1)
        encoderattn_v_weight = encoderattn_v_weight.squeeze(-1)
        encoderattn_q_weight = encoderattn_tanh_weight_q_weight * torch.tanh(
            encoderattn_q_weight) + encoderattn_tanh_bias_q_weight
        encoderattn_k_weight = encoderattn_tanh_weight_k_weight * torch.tanh(
            encoderattn_k_weight) + encoderattn_tanh_bias_k_weight
        encoderattn_v_weight = encoderattn_tanh_weight_v_weight * torch.tanh(
            encoderattn_v_weight) + encoderattn_tanh_bias_v_weight
        # encoderattn_inproj_weight = torch.cat((encoderattn_q_weight, encoderattn_k_weight, encoderattn_v_weight), dim=0)
        # model_new['decoder.layers.' + str(i) + '.encoder_attn.in_proj_weight'] = encoderattn_inproj_weight
        model_new['decoder.layers.' + str(i) + '.encoder_attn.q_weight'] = encoderattn_q_weight
        model_new['decoder.layers.' + str(i) + '.encoder_attn.k_weight'] = encoderattn_k_weight
        model_new['decoder.layers.' + str(i) + '.encoder_attn.v_weight'] = encoderattn_v_weight

        encoderattn_q_bias = model['decoder.layers.' + str(i) + '.encoder_attn.q_bias'].float()
        encoderattn_k_bias = model['decoder.layers.' + str(i) + '.encoder_attn.k_bias'].float()
        encoderattn_v_bias = model['decoder.layers.' + str(i) + '.encoder_attn.v_bias'].float()
        encoderattn_wd_q_bias = model['decoder.layers.' + str(i) + '.encoder_attn.wd_q_bias'].float()
        encoderattn_wd_k_bias = model['decoder.layers.' + str(i) + '.encoder_attn.wd_k_bias'].float()
        encoderattn_wd_v_bias = model['decoder.layers.' + str(i) + '.encoder_attn.wd_v_bias'].float()
        encoderattn_out_wd_q_bias = model['decoder.layers.' + str(i) + '.encoder_attn.out_wd_q_bias'].float()
        encoderattn_out_wd_k_bias = model['decoder.layers.' + str(i) + '.encoder_attn.out_wd_k_bias'].float()
        encoderattn_out_wd_v_bias = model['decoder.layers.' + str(i) + '.encoder_attn.out_wd_v_bias'].float()
        encoderattn_ly_wd_q_bias = model['decoder.layers.' + str(i) + '.encoder_attn.ly_wd_q_bias'].float()
        encoderattn_ly_wd_k_bias = model['decoder.layers.' + str(i) + '.encoder_attn.ly_wd_k_bias'].float()
        encoderattn_ly_wd_v_bias = model['decoder.layers.' + str(i) + '.encoder_attn.ly_wd_v_bias'].float()
        encoderattn_tanh_weight_q_bias = model['decoder.layers.' + str(i) + '.encoder_attn.tanh_weight_q_bias'].float()
        encoderattn_tanh_weight_k_bias = model['decoder.layers.' + str(i) + '.encoder_attn.tanh_weight_k_bias'].float()
        encoderattn_tanh_weight_v_bias = model['decoder.layers.' + str(i) + '.encoder_attn.tanh_weight_v_bias'].float()
        encoderattn_tanh_bias_q_bias = model['decoder.layers.' + str(i) + '.encoder_attn.tanh_bias_q_bias'].float()
        encoderattn_tanh_bias_k_bias = model['decoder.layers.' + str(i) + '.encoder_attn.tanh_bias_k_bias'].float()
        encoderattn_tanh_bias_v_bias = model['decoder.layers.' + str(i) + '.encoder_attn.tanh_bias_v_bias'].float()
        encoderattn_q_bias = torch.transpose(encoderattn_q_bias, 0, 1)
        encoderattn_k_bias = torch.transpose(encoderattn_k_bias, 0, 1)
        encoderattn_v_bias = torch.transpose(encoderattn_v_bias, 0, 1)
        encoderattn_q_bias = torch.matmul(encoderattn_q_bias, encoderattn_out_wd_q_bias)
        encoderattn_k_bias = torch.matmul(encoderattn_k_bias, encoderattn_out_wd_k_bias)
        encoderattn_v_bias = torch.matmul(encoderattn_v_bias, encoderattn_out_wd_v_bias)
        encoderattn_q_bias = torch.transpose(encoderattn_q_bias, 0, 1)
        encoderattn_k_bias = torch.transpose(encoderattn_k_bias, 0, 1)
        encoderattn_v_bias = torch.transpose(encoderattn_v_bias, 0, 1)
        encoderattn_q_bias = torch.matmul(encoderattn_q_bias, encoderattn_ly_wd_q_bias)
        encoderattn_k_bias = torch.matmul(encoderattn_k_bias, encoderattn_ly_wd_k_bias)
        encoderattn_v_bias = torch.matmul(encoderattn_v_bias, encoderattn_ly_wd_v_bias)
        encoderattn_q_bias = torch.matmul(encoderattn_q_bias, encoderattn_wd_q_bias)
        encoderattn_k_bias = torch.matmul(encoderattn_k_bias, encoderattn_wd_k_bias)
        encoderattn_v_bias = torch.matmul(encoderattn_v_bias, encoderattn_wd_v_bias)
        encoderattn_q_bias = encoderattn_q_bias.squeeze(-1)
        encoderattn_k_bias = encoderattn_k_bias.squeeze(-1)
        encoderattn_v_bias = encoderattn_v_bias.squeeze(-1)
        encoderattn_q_bias = encoderattn_tanh_weight_q_bias * torch.tanh(
            encoderattn_q_bias) + encoderattn_tanh_bias_q_bias
        encoderattn_k_bias = encoderattn_tanh_weight_k_bias * torch.tanh(
            encoderattn_k_bias) + encoderattn_tanh_bias_k_bias
        encoderattn_v_bias = encoderattn_tanh_weight_v_bias * torch.tanh(
            encoderattn_v_bias) + encoderattn_tanh_bias_v_bias

        model_new['decoder.layers.' + str(i) + '.encoder_attn.q_bias'] = encoderattn_q_bias
        model_new['decoder.layers.' + str(i) + '.encoder_attn.k_bias'] = encoderattn_k_bias
        model_new['decoder.layers.' + str(i) + '.encoder_attn.v_bias'] = encoderattn_v_bias

        encoderattn_outproj_weight = model['decoder.layers.' + str(i) + '.encoder_attn.out_proj.weight'].float()
        encoderattn_outproj_wd_weight = model['decoder.layers.' + str(i) + '.encoder_attn.out_proj.wd_weight'].float()
        encoderattn_outproj_out_wd_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn.out_proj.out_wd_weight'].float()
        encoderattn_outproj_in_wd_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn.out_proj.in_wd_weight'].float()
        encoderattn_outproj_ly_wd_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn.out_proj.ly_wd_weight'].float()
        encoderattn_outproj_tanh_weight_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn.out_proj.tanh_weight_weight'].float()
        encoderattn_outproj_tanh_bias_weight = model[
            'decoder.layers.' + str(i) + '.encoder_attn.out_proj.tanh_bias_weight'].float()
        encoderattn_outproj_weight = torch.transpose(encoderattn_outproj_weight, 0, 2)
        encoderattn_outproj_weight = torch.matmul(encoderattn_outproj_weight, encoderattn_outproj_out_wd_weight)
        encoderattn_outproj_weight = torch.transpose(encoderattn_outproj_weight, 0, 2)
        encoderattn_outproj_weight = torch.matmul(encoderattn_outproj_weight, encoderattn_outproj_ly_wd_weight)
        encoderattn_outproj_weight = torch.transpose(encoderattn_outproj_weight, 1, 2)
        encoderattn_outproj_weight = torch.matmul(encoderattn_outproj_weight, encoderattn_outproj_in_wd_weight)
        encoderattn_outproj_weight = torch.transpose(encoderattn_outproj_weight, 1, 2)
        encoderattn_outproj_weight = torch.matmul(encoderattn_outproj_weight, encoderattn_outproj_wd_weight)
        encoderattn_outproj_weight = encoderattn_outproj_weight.squeeze(-1)
        encoderattn_outproj_weight = encoderattn_outproj_tanh_weight_weight * torch.tanh(
            encoderattn_outproj_weight) + encoderattn_outproj_tanh_bias_weight
        model_new['decoder.layers.' + str(i) + '.encoder_attn.out_proj.weight'] = encoderattn_outproj_weight

        encoderattn_outproj_bias = model['decoder.layers.' + str(i) + '.encoder_attn.out_proj.bias'].float()
        encoderattn_outproj_wd_bias = model['decoder.layers.' + str(i) + '.encoder_attn.out_proj.wd_bias'].float()
        encoderattn_outproj_out_wd_bias = model[
            'decoder.layers.' + str(i) + '.encoder_attn.out_proj.out_wd_bias'].float()
        encoderattn_outproj_ly_wd_bias = model['decoder.layers.' + str(i) + '.encoder_attn.out_proj.ly_wd_bias'].float()
        encoderattn_outproj_tanh_weight_bias = model[
            'decoder.layers.' + str(i) + '.encoder_attn.out_proj.tanh_weight_bias'].float()
        encoderattn_outproj_tanh_bias_bias = model[
            'decoder.layers.' + str(i) + '.encoder_attn.out_proj.tanh_bias_bias'].float()
        encoderattn_outproj_bias = torch.transpose(encoderattn_outproj_bias, 0, 1)
        encoderattn_outproj_bias = torch.matmul(encoderattn_outproj_bias, encoderattn_outproj_out_wd_bias)
        encoderattn_outproj_bias = torch.transpose(encoderattn_outproj_bias, 0, 1)
        encoderattn_outproj_bias = torch.matmul(encoderattn_outproj_bias, encoderattn_outproj_ly_wd_bias)
        encoderattn_outproj_bias = torch.matmul(encoderattn_outproj_bias, encoderattn_outproj_wd_bias)
        encoderattn_outproj_bias = encoderattn_outproj_bias.squeeze(-1)
        encoderattn_outproj_bias = encoderattn_outproj_tanh_weight_bias * torch.tanh(
            encoderattn_outproj_bias) + encoderattn_outproj_tanh_bias_bias
        model_new['decoder.layers.' + str(i) + '.encoder_attn.out_proj.bias'] = encoderattn_outproj_bias

    args.arch = args.arch.replace('wd_v52', 'stu')
    checkpoint_new['args'] = args
    checkpoint_new['model'] = model_new

    torch.save(checkpoint_new, 'checkpoint_new.pt')

    print("finished!")