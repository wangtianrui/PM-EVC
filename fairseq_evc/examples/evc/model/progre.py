# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import typing as tp
from omegaconf import II
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel,
)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fairseq.modules import GradMultiply, LayerNorm
from .wav2vec2_model import (
    EXTRACTOR_MODE_CHOICES,
    LAYER_TYPE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    ChoiceEnum,
    ConvFeatureExtractionModel,
    GradMultiply,
    get_activation_fn,
    MultiheadAttention,
    SamePad,
    TransposeLast,
    TransposeLast,
    make_conv_pos,
    index_put
)

logger = logging.getLogger(__name__)

def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d

@dataclass
class ProgREConfig:
    label_rate: float = II("task.label_rate")

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )
    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    conv_pos_batch_norm: bool = field(
        default=False,
        metadata={
            "help": "use batch norm instead of weight norm in conv_pos (for bf16 models)"
        },
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )

    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})
    encoder_inter_loss_layer: str = field(default="3;999", metadata={"help": "If fp16 is being used"})
    spk_add: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    _name: Optional[str] = field(default="")

class PitchEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=256,
                kernel_size=5,
                bias=True,
                padding="same"
            ), 
            nn.BatchNorm1d(256, momentum=0.1), 
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                bias=True,
                padding="same"
            ), 
            nn.BatchNorm1d(256, momentum=0.1), 
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                bias=True,
                padding="same"
            ), 
            nn.BatchNorm1d(256, momentum=0.1), 
            nn.ReLU()
        )
        self.lstm = nn.GRU(input_size=256, hidden_size=32, 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.out = nn.Linear(in_features=32*2, out_features=out_dim)
    
    def forward(self, x, padding_mask, x_len):
        T = x.size(1)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.swapaxes(1, 2)
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, x_len = pad_packed_sequence(x, batch_first=True)
        if padding_mask is not None:
            return self.out(x) * (1.0 - padding_mask.to(x.dtype)).unsqueeze(-1), x_len
        else:
            return self.out(x), x_len

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        fp32 = False
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            # print(x.dtype)
            # for param_tensor in self.state_dict():
            #     print(param_tensor, self.state_dict()[param_tensor].dtype)
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x
            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            layer_result = x

            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)
    

class TransformerEncoder(nn.Module):
    def build_encoder_layer(self, args):
        if args.layer_type == "transformer":
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
            )
        return layer

    def __init__(self, args):
        super().__init__()
        self.addorsub = args.spk_add   # add is true
        self.encoder_inter_loss_layer = list(map(int, args.encoder_inter_loss_layer.split(";")))
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple

        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)

            def make_conv_block(e, k, g, l):
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                e,
                                e,
                                kernel_size=k,
                                padding=k // 2,
                                groups=g,
                            ),
                            SamePad(k),
                            TransposeLast(),
                            LayerNorm(e, elementwise_affine=False),
                            TransposeLast(),
                            nn.GELU(),
                        )
                        for _ in range(l)
                    ]
                )

            self.pos_conv = make_conv_block(
                self.embedding_dim, k, args.conv_pos_groups, num_layers
            )

        else:
            self.pos_conv = make_conv_pos(
                self.embedding_dim,
                args.conv_pos,
                args.conv_pos_groups,
            )

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(args) for _ in range(args.encoder_layers)]
        )
    
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop
        
        self.astp = ASTP(in_dim=self.embedding_dim, bottleneck_dim=256)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        min_layer=0,
    ):  
        if self.layers[-1].self_attn.k_proj.weight.dtype == torch.float16:
            self.layers[-1] = self.layers[-1].float()

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # inter_out = []
        
        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                origin_type = x.dtype
                x = x.float()
            x, (z, lr) = layer(
                x, self_attn_padding_mask=padding_mask, need_weights=False
            )
            if i in self.encoder_inter_loss_layer:
                if i == self.encoder_inter_loss_layer[0]:
                    spk_emb = self.astp(x.permute(1, 2, 0).contiguous(), padding_mask)
                    spk_emb = spk_emb.permute(2, 0, 1).contiguous()
                    layer_results.append(spk_emb.permute(1, 0, 2))
                # x_in = last_block_in - x.detach()
                # last_block_in = x_in
                if self.addorsub:
                    x = x + spk_emb.detach()
                else:
                    x = x - spk_emb.detach()

            if i == len(self.layers) - 1:
                x = x.to(origin_type)
            
            if i >= min_layer:
                layer_results.append(x.permute(1, 0, 2))
            
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r
        # T x B x C -> B x T x C
        x = x.permute(1, 0, 2).contiguous()
        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

class ProgRE(nn.Module):
    def __init__(
        self,
        cfg: ProgREConfig,
    ) -> None:
        super().__init__()
        logger.info(f"DHuBERT Config: {cfg}")

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp

        self.pitch_encoder = PitchEncoder(out_dim=cfg.encoder_embed_dim)
        self.sub_norm = LayerNorm(cfg.encoder_embed_dim)
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
        

    # def upgrade_state_dict_named(self, state_dict, name):
    #     """Upgrade a (possibly old) state dict for new versions of fairseq."""

    #     super().upgrade_state_dict_named(state_dict, name)
    #     return state_dict

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features
    
    def forward(
        self,
        source: torch.Tensor,
        pitch,
        padding_mask: Optional[torch.Tensor] = None,
        x_len = None
    ) -> Dict[str, torch.Tensor]:
        features = self.forward_features(source)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
            
        # print(source.size(), features.size(), pitch.size())
        if features.size(1) < max(x_len):
            x_len[x_len > features.size(1)] = features.size(1)
            
        pitch_emb, x_len = self.pitch_encoder(pitch[:, :features.size(1)], padding_mask=padding_mask, x_len=x_len)
        features_wo_pitch = self.sub_norm(features - pitch_emb)
        
        features_wo_pitch = self.dropout_input(features_wo_pitch)
        x = features_wo_pitch
        
        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None,
        )
        layer_results.insert(0, pitch_emb)
        layer_results.insert(0, features_wo_pitch)
        layer_results.insert(0, features)
        return x, layer_results, padding_mask, x_len

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None


class ASTP(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False, **kwargs):
        super(ASTP, self).__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1, bias=False)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1, bias=False)  # equals V and k in the paper
        self.out = nn.Linear(in_dim * 2, in_dim)
        self.layer_norm = LayerNorm(in_dim)

    def forward(self, x, padding_mask):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        # print(x.size(), padding_mask.size())
        if padding_mask is not None:
            mask = (1-padding_mask.to(x.dtype)).unsqueeze(1)
            x = x * mask
        x_in = x
        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        if padding_mask is not None:
            alpha = torch.softmax(self.linear2(alpha)*mask+(1-mask)*(-1000.0), dim=2)
        else:
            alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = alpha * x
        var = alpha * (x**2) - mean**2
        std = torch.sqrt(var + 1e-5)
        cated = torch.cat([mean, std], dim=1).permute(0, 2, 1).contiguous()
        return self.layer_norm(self.out(cated)).permute(0, 2, 1).contiguous()

