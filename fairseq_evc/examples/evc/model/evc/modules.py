from ..conv import (
    SConv1d,
)
import math
from ..lstm import SLSTM
from ..activations import *
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)
        return utter_rep

class CNNSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        padding,
        pooling,
        dropout,
        output_class_num,
        **kwargs
    ):
        super(CNNSelfAttention, self).__init__()
        self.pooling = pooling
        self.model_seq = nn.Sequential(
            nn.AvgPool1d(kernel_size, pooling, padding),
            nn.Dropout(p=dropout),
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
        )
        # self.lstm 
        self.pooling_model = SelfAttentionPooling(hidden_dim)
        self.out_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )

    def forward(self, features, features_len):
        # features_len_new = []
        attention_mask = []
        for l in features_len:
            temp = math.ceil((l / self.pooling))
            attention_mask.append(torch.ones(temp))
            # features_len_new.append(temp)
            
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        attention_mask = (1.0 - attention_mask) * -60000.0
        attention_mask = attention_mask.to(features.dtype).to(features.device)
        
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out_pooled = self.pooling_model(out, attention_mask).squeeze(-1)
        predicted = self.out_layer(out_pooled)
        return predicted, out_pooled

class RNNLayer(nn.Module):
    def __init__(self, input_dim, bidirection=True, dim=1024, num_layers=1, dropout=0.1):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2 * dim if bidirection else dim
        self.out_dim = rnn_out_dim
        # Recurrent layer
        self.layer = nn.LSTM(
            input_dim, dim, bidirectional=bidirection, num_layers=num_layers, batch_first=True, dropout=dropout
        )

    def forward(self, input_x, x_len):
        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()
        input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output, h = self.layer(input_x)
        output, x_len = pad_packed_sequence(output, batch_first=True)
        return output, x_len

class RNNs(nn.Module):
    def __init__(self,
        input_size=1024,
        bidirection=True,
        dim=1024,
        lstm_layer=2,
        vocab_size=10000,
    ):
        super(RNNs, self).__init__()
        latest_size = input_size
        self.rnns = RNNLayer(
            latest_size,
            bidirection,
            dim,
            num_layers=lstm_layer
        )
        latest_size = self.rnns.out_dim
        self.linear = nn.Linear(latest_size, dim)
        self.ctc_linear = nn.Linear(dim, vocab_size)
    
    def forward(self, x, x_len):
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, input_length, num_features).
            x_len (torch.IntTensor): Tensor of dimension (batch_size).
        Returns:
            Tensor: Predictor tensor of dimension (batch_size, input_length, number_of_classes).
        """
        x, x_len = self.rnns(x, x_len)
        x = self.linear(x)
        logits = self.ctc_linear(x)
        return logits, x
    
# class RNNs(nn.Module):
#     def __init__(self,
#         input_size=1024,
#         bidirection=True,
#         dim=1024,
#         vocab_size=10000,
#         lstm_layer=1,
#     ):
#         super(RNNs, self).__init__()
#         latest_size = input_size
        
#         self.projector = nn.Linear(input_size, input_size)
#         self.rnns = nn.ModuleList()
#         for i in range(2):
#             rnn_layer = RNNLayer(
#                 latest_size,
#                 bidirection,
#                 dim,
#             )
#             self.rnns.append(rnn_layer)
#             latest_size = rnn_layer.out_dim
#         self.linear = nn.Linear(latest_size, input_size)
#         self.ctc_linear = nn.Linear(input_size, vocab_size)
    
#     def forward(self, x, x_len):
#         r"""
#         Args:
#             x (torch.Tensor): Tensor of dimension (batch_size, input_length, num_features).
#             x_len (torch.IntTensor): Tensor of dimension (batch_size).
#         Returns:
#             Tensor: Predictor tensor of dimension (batch_size, input_length, number_of_classes).
#         """
#         for rnn in self.rnns:
#             x, x_len = rnn(x, x_len)
#         x = self.linear(x)
#         logits = self.ctc_linear(x)
#         return logits, x

class Compensator(nn.Module):
    def __init__(self, spk_dim, emo_dim, conv_dim, hidden_dim, 
                 lstm_layer, dropout, 
                 conv_ker, n_mel,
                 causal=True):
        super(Compensator, self).__init__()
        if not causal:
            out_dim = hidden_dim * 2
        else:
            out_dim = hidden_dim
        self.map = RNNLayer(
            input_dim=conv_dim+spk_dim+emo_dim, bidirection=(not causal),
            dim=hidden_dim, num_layers=lstm_layer, dropout=dropout   
        )
        self.encoder = RNNLayer(
            input_dim=out_dim+spk_dim+emo_dim, bidirection=(not causal),
            dim=hidden_dim, num_layers=lstm_layer, dropout=dropout   
        )
        self.out = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            SConv1d(in_channels=out_dim, out_channels=out_dim, kernel_size=conv_ker, causal=causal),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            SConv1d(in_channels=out_dim, out_channels=n_mel, kernel_size=conv_ker, causal=causal)
        )
        
    def forward(self, conv_out, spk, emo, feature_len):
        """compensator forward # output: B,T,n_mel
        Args:
            conv_out (_type_): B,T,D
            spk (_type_): B,T,D (repeated T times)
            emo (_type_): B,T,D (repeated T times)
            feature_len (_type_): B
        """
        fusion_inp, _ = self.map(
            torch.cat([conv_out.to(spk.device), spk, emo], dim=-1), 
            feature_len
        )
        hidden_feature, _ = self.encoder(
            torch.cat([fusion_inp, spk, emo], dim=-1), 
            feature_len
        )
        compensate = self.out(hidden_feature.permute(0, 2, 1))
        return compensate

class SEANetResnetBlock(nn.Module):
    def __init__(self, dim: int, kernel_sizes, dilations,
                 causal: bool = True, dropout=0.1, compress: int = 2):
        super().__init__()
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                nn.ELU(alpha=1.0),
                nn.Dropout(p=dropout),
                SConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation, causal=causal),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut = SConv1d(dim, dim, kernel_size=1, causal=causal)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class ProgressiveDecoder(nn.Module):
    def __init__(self, spk_dim, emo_dim, content_dim, hidden_dim, 
                 lstm_layer, dropout, 
                 resnet_conv_ker=3, resnet_dilation=2,
                 resnet_layers=4, n_filters=32, in_kernel_size = 7, resnet_compress=2,
                 n_mel=80, causal=True):
        super().__init__()
        mult = int(2 ** resnet_layers) 
        self.conv_in = nn.Sequential(
            SConv1d(in_channels=content_dim, out_channels=hidden_dim, kernel_size=in_kernel_size),
            SConv1d(hidden_dim, mult * n_filters, in_kernel_size, causal=causal),
        )
        self.conv_lstm = SLSTM(mult * n_filters, num_layers=lstm_layer, causal=causal, dropout=dropout)
        self.models = nn.ModuleList()
        
        for i in range(resnet_layers):
            if i == 1:
                inp_ch = mult * n_filters + emo_dim
            elif i == 2:
                inp_ch = mult * n_filters + spk_dim
            else:
                inp_ch = mult * n_filters
            model = []
            model += [
                nn.ELU(alpha=1.0),
                nn.Dropout(p=dropout),
                SConv1d(
                    inp_ch, mult * n_filters // 2, kernel_size=resnet_conv_ker, causal=causal
                ),
            ]
            model += [
                SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[resnet_conv_ker, 1], dropout=dropout,
                                    dilations=[resnet_dilation, 1], causal=causal, compress=resnet_compress)]

            mult //= 2
            self.models.append(nn.Sequential(*model))
        
        self.final_layer = nn.Sequential(
            nn.ELU(alpha=1.0),
            SConv1d(n_filters, n_mel, in_kernel_size, causal=causal)
        )
    
    def forward(self, content, speaker, emo, feature_len):
        x = self.conv_in(content)
        x = self.conv_lstm(x, feature_len)
        for i in range(len(self.models)):
            # print(x.size(), speaker.size())
            if i == 1:
                x = self.models[i](torch.cat([x, emo], dim=1))
            elif i == 2:
                x = self.models[i](torch.cat([x, speaker], dim=1))
            else:
                x = self.models[i](x)
        return self.final_layer(x)