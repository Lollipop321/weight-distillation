# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import fairseq


def WDV29LayerNorm(normalized_shape, wd_require_gradient=False, eps=1e-5, elementwise_affine=True):
    return fairseq.modules.WDV29LayerNormalization(normalized_shape, wd_require_gradient, eps, elementwise_affine)