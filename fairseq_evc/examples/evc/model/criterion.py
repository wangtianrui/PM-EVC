# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class DCodecRecConfig(FairseqDataclass):
    n_mel_channels: int = field(
        default=32,
        metadata={"help": "n_mel channels"}
    )
    loss_weights: Optional[Dict[str, float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )


@register_criterion("dcodec_reconstruct", dataclass=DCodecRecConfig)
class DCodecRecoCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        n_mel_channels,
        loss_weights=None,
    ):
        super().__init__(task)
        self.n_mel_channels = n_mel_channels
        self.sample_rate = task.cfg.sample_rate
        self.loss_weights = loss_weights
        
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss = 0.0
        logging_output = {}

        net_output = model(**sample["net_input"])

        losses = net_output["losses"]

        if self.loss_weights is not None:
            for lk, lw in losses.items():
                if lk in self.loss_weights:
                    loss = loss + self.loss_weights[lk] * lw

        logging_output = {
            "loss": loss.item(),
            "sample_size": sample["net_input"]["source"].shape[0],
            "nsentences": sample["net_input"]["source"].size(0),
        }
        
        for loss_key in losses.keys():
            logging_output[loss_key] = losses[loss_key].item()

        return loss, sample["net_input"]["source"].shape[0], logging_output # TODO: check this line

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        total_node = len(logging_outputs)
        if total_node == 0:
            total_node = 1.0

        metrics.log_scalar(
            "loss", loss_sum / total_node, sample_size, round=5
        )
        metrics.log_scalar(
            "fbank_loss_0",
            (sum(log.get("fbank_loss_0", 0) for log in logging_outputs)) / total_node,
            sample_size, round=5
        )
        metrics.log_scalar(
            "fbank_loss_1",
            (sum(log.get("fbank_loss_1", 0) for log in logging_outputs)) / total_node,
            sample_size, round=5
        )
        metrics.log_scalar(
            "ctc_score",
            (sum(log.get("ctc_score", 0) for log in logging_outputs)) / total_node,
            sample_size, round=5
        )
        metrics.log_scalar(
            "spk_score",
            (sum(log.get("spk_score", 0) for log in logging_outputs)) / total_node,
            sample_size, round=5
        )
        metrics.log_scalar(
            "emo_score",
            (sum(log.get("emo_score", 0) for log in logging_outputs)) / total_node,
            sample_size, round=5
        )
        metrics.log_scalar(
            "sid_acc",
            (sum(log.get("sid_acc", 0) for log in logging_outputs)) / total_node,
            sample_size, round=5
        )
        # metrics.log_scalar(
        #     "asr_acc",
        #     (sum(log.get("asr_acc", 0) for log in logging_outputs)) / total_node,
        #     sample_size, round=5
        # )
        # metrics.log_scalar(
        #     "acc_mean",
        #     (sum(log.get("acc_mean", 0) for log in logging_outputs)) / total_node,
        #     sample_size, round=5
        # )
        metrics.log_scalar(
            "emo_acc",
            (sum(log.get("emo_acc", 0) for log in logging_outputs)) / total_node,
            sample_size, round=5
        )
        metrics.log_scalar(
            "prior_loss",
            (sum(log.get("prior_loss", 0) for log in logging_outputs)) / total_node,
            sample_size, round=5
        )
        metrics.log_scalar(
            "diff_loss",
            (sum(log.get("diff_loss", 0) for log in logging_outputs)) / total_node,
            sample_size, round=5
        )
        metrics.log_scalar(
            "vq_loss",
            (sum(log.get("vq_loss", 0) for log in logging_outputs)) / total_node,
            sample_size, round=5
        )
    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
