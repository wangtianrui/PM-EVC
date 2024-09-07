# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field, fields
from typing import Callable, Dict, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import typing as tp
from omegaconf import II
import torch.nn.functional as F
from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import LayerNorm
from .modules import RNNs, Compensator, ProgressiveDecoder, CNNSelfAttention
from ..progre import ProgREConfig, ProgRE
import s3prl.hub as hub
from fairseq.modules import GradMultiply
from copy import deepcopy
import random
import os
from ..whisper import load_model as whisper_load_model, pad_or_trim, mel_filters, N_FFT, N_SAMPLES, HOP_LENGTH, N_SAMPLES_PER_TOKEN

logger = logging.getLogger(__name__)

def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d

def make_pad_mask(lengths: List[int], max_len: int = 0):
    """Make mask containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (List[int]): Batch of lengths (B,).
    Returns:
        np.ndarray: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    # lengths:(B,)
    batch_size = int(len(lengths))
    max_len = max_len if max_len > 0 else max(lengths)
    # np.arange(0, max_len): [0,1,2,...,max_len-1]
    # seq_range (1, max_len)
    seq_range = np.expand_dims(np.arange(0, max_len), 0)
    # seq_range_expand (B,max_len)
    seq_range_expand = np.tile(seq_range, (batch_size, 1))
    # (B,1)
    seq_length_expand = np.expand_dims(lengths, -1)
    # (B,max_len)
    mask = seq_range_expand >= seq_length_expand
    return torch.BoolTensor(mask)

def load_progre_param(model, pretrained_ckpt):
    model_state = model.state_dict()
    torch_names = deepcopy(list(model_state.keys()))
    pretrained_dict = {}
    for k, v in pretrained_ckpt.items():
        # print(k, v.size())
        if k.find("fbank_encoder") != -1 and k.find("fc") != -1:
            k = k.replace("fc1", "norm1")
            k = k.replace("fc2", "norm2")
        if k.find("spk_astp.out.dense") != -1:
            k = k.replace("dense.", "")
        if k.find("sub_spk_norm") != -1:
            k = k.replace("beta", "bias")
        if k in model_state and k.find("label_embs_concat") == -1 and k.find("final_proj") == -1:
            if v.shape != model_state[k].shape:
                if len(v.shape) != len(model_state[k].shape):
                    print("%s convert shape from %s to %s"%(k, str(v.shape), str(model_state[k].shape))) 
                    v = v.reshape(model_state[k].shape)
                else:
                    print("transpose shape from %s to %s"%(str(v.shape), str(v.T.shape))) 
                    v = v.T
                
            pretrained_dict[k] = v
            # print("%s loaded"%k)
            torch_names.remove(k)
        else:
            print("%s not in model"%k)
    print("not loaded params: "+str(torch_names))
    model_state.update(pretrained_dict)
    model.load_state_dict(model_state)
    return model

def spec_augment(spec, mask_T=40, mask_F=50, num_T=2, num_F=4, p=1.0):
    # spec: B, T, F
    def _start_to_intervals(starts, consecutive):
        tiled = starts.expand(consecutive, starts.size(0)).permute(1, 0)
        offset = torch.arange(consecutive).expand_as(tiled)
        intervals = tiled + offset
        return intervals.view(-1)

    with torch.no_grad():
        upper_bound = (
            spec.shape[1] * p
        )  # upper bound on the time mask so that a time mask cannot be wider than p times the number of time steps
        for idx in range(spec.shape[0]):
            # time masking
            if mask_T > 0 and mask_T < upper_bound:
                for _ in range(num_T):
                    rand_consecutive = random.randint(0, mask_T)
                    chosen_start = torch.randperm(spec.shape[1] - rand_consecutive)[:1]
                    chosen_intervals = _start_to_intervals(
                        chosen_start, rand_consecutive
                    )
                    spec[idx, chosen_intervals, :] = 0
            # frequency masking
            if mask_F > 0:
                for _ in range(num_F):
                    rand_bandwidth = random.randint(0, mask_F)
                    chosen_start = torch.randperm(spec.shape[2] - rand_bandwidth)[:1]
                    chosen_intervals = _start_to_intervals(chosen_start, rand_bandwidth)
                    spec[idx, :, chosen_intervals] = 0
        return spec

@dataclass
class PMECVCConfig(FairseqDataclass):
    spk_hidden_dim: int = field(default=256)
    spk_ker_size: int = field(default=5)
    spk_pool_size: int = field(default=5)
    spk_dropout: float = field(default=0.4)
    
    emo_hidden_dim: int = field(default=256)
    emo_ker_size: int = field(default=5)
    emo_pool_size: int = field(default=5)
    emo_dropout: float = field(default=0.4)
    
    asr_hidden_dim: int = field(default=1024)
    asr_bidirection: bool = field(default=False)
    asr_lstm_layer: int = field(default=2)
    
    compensator_hidden_dim: int = field(default=256)
    compensator_lstm_layer: int = field(default=2)
    compensator_lstm_dropout: float = field(default=0.1)
    compensator_conv_ker: int = field(default=3)
    
    decoder_resnet_conv_ker: int = field(default=3)
    decoder_resnet_dilation: int = field(default=2)
    decoder_resnet_layers: int = field(default=4)
    decoder_n_filters: int = field(default=32)
    decoder_in_kernel_size: int = field(default=4)
    decoder_resnet_compress: int = field(default=4)
    decoder_hidden_dim: int = field(default=512)
    decoder_lstm_layer: int = field(default=2)
    decoder_dropout: float = field(default=0.1)
    decoder_causal: bool = field(default=True)
    
    n_mel: int = field(default=80)
    
    emo_num: int = field(default=8)
    spk_num: int = field(default=2716)
    upstream: str = field(default="hubert")
    upstream_ckpt: str = field(default="hubert")
    upstream_num_hidden: int = field(default=-1)
    upstream_hidden_dim: int = field(default=512)
    
    spec_aug: bool = field(default=False)
    mask_T: int = field(default=40)
    mask_F: int = field(default=50)
    num_T: int = field(default=2)
    num_F: int = field(default=4)

@register_model("pm_ecvc", dataclass=PMECVCConfig)
class PMECVC(BaseFairseqModel):
    def __init__(self, cfg: PMECVCConfig, asr_dict):
        super().__init__()
        self.spec_aug = cfg.spec_aug
        self.cfg = cfg
        self.asr_dict = asr_dict
        if cfg.upstream == "progre":
            pm_state = torch.load(cfg.upstream_ckpt, map_location="cpu")
            # print("\n\n", pm_state.keys(), "\n\n")
            if "cfg" not in pm_state.keys():
                dataclass_fields = {field.name: getattr(pm_state["args"], field.name) for field in fields(ProgREConfig) if hasattr(pm_state["args"], field.name)}
                pm_state["cfg"] = {"model": dataclass_fields}
            config = ProgREConfig(**pm_state["cfg"]["model"])
            self.pm = ProgRE(config)
            self.pm = load_progre_param(self.pm, pretrained_ckpt=pm_state["model"])
        elif cfg.upstream == "whisper":
            pm = whisper_load_model(cfg.upstream_ckpt.split("/")[-1], download_root=os.path.dirname(cfg.upstream_ckpt))
            self.register_buffer("mel_filter", mel_filters("cpu", pm.dims.n_mels))
            self.register_buffer("window", torch.hann_window(N_FFT))
            self.pm = pm.encoder
        elif cfg.upstream == "fbank":
            import librosa
            self.win_len = 1280
            self.hop_len = 320
            self.register_buffer("mel_filter", torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=self.win_len, n_mels=128)))
            self.register_buffer("window", torch.hann_window(self.win_len))
            self.pm = None
        else:
            self.pm = getattr(hub, cfg.upstream)(cfg.upstream_ckpt)
            
        if self.pm is not None:
            self.pm = self.pm.half().eval()
        
        self.spk_weights = nn.Parameter(torch.ones(cfg.upstream_num_hidden, 1))
        self.asr_weights = nn.Parameter(torch.ones(cfg.upstream_num_hidden, 1))
        self.emo_weights = nn.Parameter(torch.ones(cfg.upstream_num_hidden, 1))
        
        self.spk_utt_model = CNNSelfAttention(
            input_dim=cfg.upstream_hidden_dim, hidden_dim=cfg.spk_hidden_dim, 
            padding=cfg.spk_ker_size//2, pooling=cfg.spk_pool_size, 
            kernel_size=cfg.spk_ker_size, dropout=cfg.spk_dropout, output_class_num=cfg.spk_num
        )
        self.asr_model = RNNs(
            input_size=cfg.upstream_hidden_dim, 
            bidirection=cfg.asr_bidirection, dim=cfg.asr_hidden_dim, lstm_layer=cfg.asr_lstm_layer,
            vocab_size=len(asr_dict)
        )
        self.emo_utt_model = CNNSelfAttention(
            input_dim=cfg.upstream_hidden_dim, hidden_dim=cfg.emo_hidden_dim, 
            padding=cfg.emo_ker_size//2, pooling=cfg.emo_pool_size, 
            kernel_size=cfg.emo_ker_size, dropout=cfg.emo_dropout, output_class_num=cfg.emo_num
        )
        self.decoder = ProgressiveDecoder(
            spk_dim=cfg.spk_hidden_dim, emo_dim=cfg.emo_hidden_dim, content_dim=cfg.asr_hidden_dim,
            hidden_dim=cfg.decoder_hidden_dim,
            lstm_layer=cfg.decoder_lstm_layer, dropout=cfg.decoder_dropout, 
            resnet_conv_ker=cfg.decoder_resnet_conv_ker, resnet_dilation=cfg.decoder_resnet_dilation,
            resnet_layers=cfg.decoder_resnet_layers, n_filters=cfg.decoder_n_filters,
            in_kernel_size=cfg.decoder_in_kernel_size, resnet_compress=cfg.decoder_resnet_compress,
            n_mel=cfg.n_mel, causal=cfg.decoder_causal
        )
        # self.content_vq = VectorQuantization(
        #     dim=cfg.asr_hidden_dim, codebook_size=cfg.bins, codebook_dim=cfg.codebook_dim
        # )
        # self.content_vq = VectorQuantization(
        #     dim=1024, codebook_size=cfg.bins, codebook_dim=cfg.codebook_dim
        # )
        self.compensator = Compensator(
            spk_dim=cfg.spk_hidden_dim, emo_dim=cfg.emo_hidden_dim, conv_dim=cfg.upstream_hidden_dim,
            hidden_dim=cfg.compensator_hidden_dim, lstm_layer=cfg.compensator_lstm_layer,
            dropout=cfg.compensator_lstm_dropout,
            conv_ker=cfg.compensator_conv_ker, n_mel=cfg.n_mel, causal=cfg.decoder_causal
        )

        self.l1_loss = nn.L1Loss(reduction="sum")
        self.ctc = nn.CTCLoss(blank=self.asr_dict.bos(), zero_infinity=True)
        self.ce = nn.CrossEntropyLoss()
        self.emotion_loss = nn.CrossEntropyLoss()
        
    @classmethod
    def build_model(cls, cfg: PMECVCConfig, task):
        """Build a new model instance."""
        model = PMECVC(cfg, task.asr_dict)
        if task.stage == 1:
            for name, param in model.named_parameters():
                if name.startswith("pm."):
                    param.requires_grad = False
                elif name.startswith("decoder."):
                    param.requires_grad = False
                elif name.startswith("compensator."):
                    param.requires_grad = False
                elif name.startswith("fm."):
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if name.startswith("pm."):
                    param.requires_grad = False
                elif name.startswith("asr"):
                    param.requires_grad = False
                elif name.startswith("spk"):
                    param.requires_grad = False
                elif name.startswith("emo"):
                    param.requires_grad = False
        return model
    
    def ctc_loss(self, logits, log_probs_len, labels, labels_len, reverse_g=False):
        if reverse_g:
            logits = GradMultiply.apply(logits, -1.0)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        
        pad_mask = (labels != self.asr_dict.pad()) & (labels != self.asr_dict.eos())
        labels = labels.masked_select(pad_mask)
        target_lengths = pad_mask.sum(-1)
        
        with torch.backends.cudnn.flags(enabled=False):
            loss = self.ctc(
                    log_probs.transpose(0, 1), # (N, T, C) -> (T, N, C)
                    labels,
                    log_probs_len,
                    target_lengths,
                )
        # with torch.no_grad():
        #     pred_token_ids = log_probs.argmax(dim=-1).unique_consecutive()
        #     acc = torch.sum((pred_token_ids == labels) * pad_mask) / torch.sum(target_lengths)
        return loss
    
    def spk_entro_loss(self, logits, labels, reverse_g=False):
        if reverse_g:
            logits = GradMultiply.apply(logits, -1.0)
        return self.ce(logits, labels), self.compute_accuracy(logits, labels)
    
    def emo_entro_loss(self, logits, labels, reverse_g=False):
        if reverse_g:
            logits = GradMultiply.apply(logits, -1.0)
        return self.emotion_loss(logits, labels), self.compute_accuracy(logits, labels)
    
    def compute_accuracy(self, lprobs, target):
        """_summary_
            lprobs (_type_): B,D
            target (_type_): B
        """
        with torch.no_grad():
            n_correct = torch.sum(
                lprobs.argmax(1).eq(target)
            )
            acc = n_correct / lprobs.size(0)
        return acc
    
    def forward_downstream(self, stacked_feature, feature_len, padding_mask):
        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(stacked_feature.size(0), -1)
        
        content_feature = (F.softmax(self.asr_weights, dim=0) * stacked_feature).sum(dim=0)
        content_feature = content_feature.view(*origin_shape)
        
        speaker_feature = (F.softmax(self.spk_weights, dim=0) * stacked_feature).sum(dim=0)
        speaker_feature = speaker_feature.view(*origin_shape)
        
        emotion_feature = (F.softmax(self.emo_weights, dim=0) * stacked_feature).sum(dim=0)
        emotion_feature = emotion_feature.view(*origin_shape) 
        
        if self.spec_aug and self.training:
            content_feature = spec_augment(content_feature, mask_T=self.cfg.mask_T, mask_F=self.cfg.mask_F, num_T=self.cfg.num_T, num_F=self.cfg.num_F)
            speaker_feature = spec_augment(speaker_feature, mask_T=self.cfg.mask_T, mask_F=self.cfg.mask_F, num_T=self.cfg.num_T, num_F=self.cfg.num_F)
            emotion_feature = spec_augment(emotion_feature, mask_T=self.cfg.mask_T, mask_F=self.cfg.mask_F, num_T=self.cfg.num_T, num_F=self.cfg.num_F)
        
        # specific modeling
        con_4_ctc, content_feature = self.asr_model(content_feature, feature_len)
        spk_4_ce, spk_pooled_utt = self.spk_utt_model(speaker_feature, feature_len)
        spk_pooled = spk_pooled_utt.unsqueeze(1).repeat(1, origin_shape[1], 1) # B,D -> B,T,D
        emo_4_ce, emo_pooled_utt = self.emo_utt_model(emotion_feature, feature_len)
        emo_pooled = emo_pooled_utt.unsqueeze(1).repeat(1, origin_shape[1], 1) # B,D -> B,T,D
        
        if padding_mask is not None:
            temp_mask = (1-padding_mask.to(content_feature.dtype)).unsqueeze(-1)
            content_feature = content_feature * temp_mask
            spk_pooled = spk_pooled * temp_mask
            emo_pooled = emo_pooled * temp_mask
            con_4_ctc = con_4_ctc * temp_mask
            
        return content_feature, spk_pooled, emo_pooled, con_4_ctc, spk_4_ce, emo_4_ce, spk_pooled_utt, emo_pooled_utt
    
    def forward(
        self,
        source: torch.Tensor,
        pitch,
        padding_mask: Optional[torch.Tensor] = None,
        fbank = None,
        tgt_content = None,
        tgt_content_len = None,
        tgt_spk = None,
        tgt_emo = None,
        feature_len = None,
        only_downstream = False,
        no_paded_source = None,
        audio_lens = None,
        *args, **kwargs
    ):
        losses = {}
        with torch.no_grad():
            feature_len = feature_len.cpu()
            if self.pm is not None:
                self.pm.eval()
            if self.cfg.upstream == "progre":
                temp, hidden_layers, padding_mask, feature_len = self.pm(source.half(), pitch.half(), padding_mask, x_len=feature_len)
            elif self.cfg.upstream == "whisper":
                source = nn.functional.pad(source, (0, N_SAMPLES - source.size(1), 0, 0), mode="constant", value=0.0)
                stft = torch.stft(source, N_FFT, HOP_LENGTH, window=self.window, return_complex=True)
                magnitudes = stft[..., :-1].abs() ** 2
                mel_spec = self.mel_filter @ magnitudes
                log_spec = torch.clamp(mel_spec, min=1e-10).log10()
                log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
                log_spec = (log_spec + 4.0) / 4.0
                _, hidden_layers_temp = self.pm(log_spec.half(), hidden_return=True)
                hidden_layers = []
                max_len = feature_len.max()
                for i in range(len(hidden_layers_temp)):
                    hidden_layers.append(hidden_layers_temp[i][:, :max_len])
            elif self.cfg.upstream == "fbank":
                stft = torch.stft(source, self.win_len, self.hop_len, window=self.window, return_complex=True)
                magnitudes = stft[..., :-1].abs() ** 2
                mel_spec = (self.mel_filter @ magnitudes).permute(0, 2, 1)
                hidden_layers = [mel_spec, torch.clamp(mel_spec, min=1e-10).log10()] #
            else:
                no_paded_source = [item.half() for item in no_paded_source]
                hidden_layers_temp = self.pm(no_paded_source)["hidden_states"]
                feature_len_s3prl = hidden_layers_temp[0].size(1)
                hidden_layers = []
                max_len = feature_len.max()
                for i in range(len(hidden_layers_temp)):
                    if max_len <= feature_len_s3prl:
                        hidden_layers.append(hidden_layers_temp[i][:, :max_len])
                    else:
                        hidden_layers.append(F.pad(hidden_layers_temp[i], (0, 0, 0, max_len-feature_len_s3prl), mode="constant", value=0.0))
                del hidden_layers_temp
                
            conv_out = hidden_layers[0]
            stacked_feature = torch.stack(hidden_layers[1:], dim=0)
            stacked_feature = F.layer_norm(
                stacked_feature, (stacked_feature.shape[-1],)
            )
        if only_downstream:
            content_feature, spk_pooled, emo_pooled, con_4_ctc, spk_4_ce, emo_4_ce, spk_pooled_utt, emo_pooled_utt = \
                self.forward_downstream(stacked_feature, feature_len, padding_mask)
            ctc_score = self.ctc_loss(con_4_ctc, feature_len, tgt_content, tgt_content_len)
            sid_score, sid_acc = self.spk_entro_loss(spk_4_ce, tgt_spk)
            emo_score, emo_acc = self.emo_entro_loss(emo_4_ce, tgt_emo[:, 0])
        else:
            with torch.no_grad():
                content_feature, spk_pooled, emo_pooled, con_4_ctc, spk_4_ce, emo_4_ce, spk_pooled_utt, emo_pooled_utt = \
                self.forward_downstream(stacked_feature, feature_len, padding_mask)
                ctc_score = self.ctc_loss(con_4_ctc, feature_len, tgt_content, tgt_content_len)
                sid_score, sid_acc = self.spk_entro_loss(spk_4_ce, tgt_spk)
                emo_score, emo_acc = self.emo_entro_loss(emo_4_ce, tgt_emo[:, 0])
            
            decoder_out_first = self.decoder(content_feature.permute(0, 2, 1), spk_pooled.permute(0, 2, 1), emo_pooled.permute(0, 2, 1), feature_len)
            compensate = self.compensator(
                conv_out.to(decoder_out_first.dtype), spk_pooled, emo_pooled, feature_len
            )
            decoder_out = decoder_out_first.detach() + compensate
            
            decoder_out = decoder_out.permute(0, 2, 1)
            decoder_out_first = decoder_out_first.permute(0, 2, 1)
            max_len = min(decoder_out.size(1), fbank.size(1))
            decoder_out = decoder_out[:, :max_len]
            tgt_fbank = fbank[:, :max_len]
            decoder_out_first = decoder_out_first[:, :max_len]
            
            if padding_mask is not None:
                temp_mask = (1-padding_mask.to(decoder_out_first.dtype)).unsqueeze(-1)
                decoder_out = decoder_out * temp_mask
                tgt_fbank = tgt_fbank * temp_mask
                decoder_out_first = decoder_out_first * temp_mask
            # Fbank loss
            div = torch.sum(feature_len) * decoder_out_first.size(-1)
            loss1 = self.l1_loss(input=decoder_out_first, target=tgt_fbank) 
            loss2 = self.l1_loss(input=decoder_out, target=tgt_fbank) 
            losses["fbank_loss_0"] = loss1 / div
            losses["fbank_loss_1"] = loss2 / div
            
        losses["ctc_score"] = ctc_score
        losses["spk_score"] = sid_score
        losses["emo_score"] = emo_score
        losses["sid_acc"] = sid_acc
        losses["emo_acc"] = emo_acc
        return {"losses" : losses}
        
        
    def encode(self, source, pitch, padding_mask=None, feature_len=None):
        feature_len = feature_len.cpu()
        if self.pm is not None:
            self.pm.eval()
        if self.cfg.upstream == "progre":
            temp, hidden_layers, padding_mask, feature_len = self.pm(source, pitch, padding_mask, x_len=feature_len)
        elif self.cfg.upstream == "whisper":
            source = nn.functional.pad(source, (0, N_SAMPLES - source.size(1), 0, 0), mode="constant", value=0.0)
            stft = torch.stft(source, N_FFT, HOP_LENGTH, window=self.window, return_complex=True)
            magnitudes = stft[..., :-1].abs() ** 2
            mel_spec = self.mel_filter @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            _, hidden_layers_temp = self.pm(log_spec, hidden_return=True)
            hidden_layers = []
            max_len = feature_len.max()
            for i in range(len(hidden_layers_temp)):
                hidden_layers.append(hidden_layers_temp[i][:, :max_len])
        elif self.cfg.upstream == "fbank":
            stft = torch.stft(source, self.win_len, self.hop_len, window=self.window, return_complex=True)
            magnitudes = stft[..., :-1].abs() ** 2
            mel_spec = (self.mel_filter @ magnitudes).permute(0, 2, 1)
            hidden_layers = [mel_spec, torch.clamp(mel_spec, min=1e-10).log10()] #
        else:
            no_paded_source = [item for item in source]
            hidden_layers_temp = self.pm(no_paded_source)["hidden_states"]
            feature_len_s3prl = hidden_layers_temp[0].size(1)
            hidden_layers = []
            max_len = feature_len.max()
            for i in range(len(hidden_layers_temp)):
                if max_len <= feature_len_s3prl:
                    hidden_layers.append(hidden_layers_temp[i][:, :max_len])
                else:
                    hidden_layers.append(F.pad(hidden_layers_temp[i], (0, 0, 0, max_len-feature_len_s3prl), mode="constant", value=0.0))
            del hidden_layers_temp
            
        conv_out = hidden_layers[0]
        stacked_feature = torch.stack(hidden_layers[1:], dim=0)
        stacked_feature = F.layer_norm(
            stacked_feature, (stacked_feature.shape[-1],)
        )
        content_feature, spk_pooled, emo_pooled, con_4_ctc, spk_4_ce, emo_4_ce, spk_pooled_utt, emo_pooled_utt = \
                self.forward_downstream(stacked_feature, feature_len, padding_mask)
        feature = [content_feature, spk_pooled_utt, emo_pooled_utt, conv_out]
        return feature
    
    def decode(self, stacked_feature, feature_len=None, only2stage=False):
        content_feature, spk_pooled_utt, emo_pooled_utt, conv_out = stacked_feature
        spk_pooled = spk_pooled_utt.unsqueeze(1).repeat(1, content_feature.size(1), 1) # B,D -> B,T,D
        emo_pooled = emo_pooled_utt.unsqueeze(1).repeat(1, content_feature.size(1), 1) # B,D -> B,T,D
        # decoder
        decoder_out_first = self.decoder(content_feature.permute(0, 2, 1), spk_pooled.permute(0, 2, 1), emo_pooled.permute(0, 2, 1), feature_len)
        compensate = self.compensator(
            conv_out.to(decoder_out_first.dtype), spk_pooled, emo_pooled, feature_len
        )
        decoder_out = decoder_out_first.detach() + compensate
        # decoder_out = decoder_out_first
        if only2stage:
            return decoder_out_first
        return decoder_out
    
    def downstream(self, source, pitch, padding_mask=None, feature_len=None):
        if self.pm is not None:
            self.pm.eval()
        if self.cfg.upstream == "progre":
            temp, hidden_layers, padding_mask, feature_len = self.pm(source, pitch, padding_mask, x_len=feature_len)
        elif self.cfg.upstream == "whisper":
            source = nn.functional.pad(source, (0, N_SAMPLES - source.size(1), 0, 0), mode="constant", value=0.0)
            stft = torch.stft(source, N_FFT, HOP_LENGTH, window=self.window, return_complex=True)
            magnitudes = stft[..., :-1].abs() ** 2
            mel_spec = self.mel_filter @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            _, hidden_layers_temp = self.pm(log_spec, hidden_return=True)
            hidden_layers = []
            max_len = feature_len.max()
            for i in range(len(hidden_layers_temp)):
                hidden_layers.append(hidden_layers_temp[i][:, :max_len])
        elif self.cfg.upstream == "fbank":
            stft = torch.stft(source, self.win_len, self.hop_len, window=self.window, return_complex=True)
            magnitudes = stft[..., :-1].abs() ** 2
            mel_spec = (self.mel_filter @ magnitudes).permute(0, 2, 1)
            hidden_layers = [mel_spec, torch.clamp(mel_spec, min=1e-10).log10()] #
        else:
            no_paded_source = [item for item in source]
            hidden_layers_temp = self.pm(no_paded_source)["hidden_states"]
            feature_len_s3prl = hidden_layers_temp[0].size(1)
            hidden_layers = []
            max_len = feature_len.max()
            for i in range(len(hidden_layers_temp)):
                if max_len <= feature_len_s3prl:
                    hidden_layers.append(hidden_layers_temp[i][:, :max_len])
                else:
                    hidden_layers.append(F.pad(hidden_layers_temp[i], (0, 0, 0, max_len-feature_len_s3prl), mode="constant", value=0.0))
            del hidden_layers_temp
            
        conv_out = hidden_layers[0]
        stacked_feature = torch.stack(hidden_layers[1:], dim=0)
        stacked_feature = F.layer_norm(
            stacked_feature, (stacked_feature.shape[-1],)
        )
        content_feature, spk_pooled, emo_pooled, con_4_ctc, spk_4_ce, emo_4_ce, spk_pooled_utt, emo_pooled_utt = \
                self.forward_downstream(stacked_feature, feature_len, padding_mask)
        
        feature = [con_4_ctc, spk_4_ce, emo_4_ce, spk_pooled_utt]
        return feature