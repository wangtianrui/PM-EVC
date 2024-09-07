# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True, causal=True, dropout=0.1):
        super().__init__()
        self.skip = skip
        if not causal:
            hidden_dim = dimension // 2
        else:
            hidden_dim = dimension
        self.lstm = nn.LSTM(dimension, hidden_dim, num_layers, bidirectional=not causal, dropout=dropout)
        

    def forward(self, x, feature_len):
        x = x.permute(2, 0, 1)
        x_in = pack_padded_sequence(x, feature_len, enforce_sorted=False)
        y, _ = self.lstm(x_in)
        y, feature_len = pad_packed_sequence(y)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
