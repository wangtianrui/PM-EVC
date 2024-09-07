from copy import deepcopy
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
from fairseq.data import Dictionary
from dataclasses import dataclass, field
from .dataset_ec import EVCDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING
import torch
logger = logging.getLogger(__name__)


@dataclass
class EVCConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    label_rate: int = field(
        default=50,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    pad_tgt_length: int = field(
        default=-1,
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    pretrain_checkpoint: Optional[str] = field(
        default=None
    )
    pretrain_checkpoint_sid: Optional[str] = field(
        default=None
    )
    pretrain_checkpoint_asr: Optional[str] = field(
        default=None
    )
    pretrain_checkpoint_esc: Optional[str] = field(
        default=None
    )
    pitch_home: Optional[str] = field(
        default=""
    )
    fbank_home: Optional[str] = field(
        default=""
    )
    speech_fbank_home: Optional[str] = field(
        default=""
    )
    noise_fbank_home: Optional[str] = field(
        default=""
    )
    load_layer_weight_fromckpt: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    vocab_name: Optional[str] = field(
        default=""
    )
    audio_home: Optional[str] = field(
        default=""
    )
    downstream_checkpoint: Optional[str] = field(
        default=""
    )
    hifi_checkpoint: Optional[str] = field(
        default=""
    )
    fm_checkpoint: Optional[str] = field(
        default=""
    )
    pretrained_checkpoint: Optional[str] = field(
        default=""
    )
    stage: Optional[int] = field(
        default=1
    )
    change_content: Optional[str] = field(
        default=""
    )

@register_task("evc", dataclass=EVCConfig)
class ECPretrainingTask(FairseqTask):

    cfg: EVCConfig

    def __init__(
        self,
        cfg: EVCConfig,
    ) -> None:
        super().__init__(cfg)
        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"ECPretrainingTask Config {cfg}")
        self.cfg = cfg
        vocab_path = os.path.join(self.cfg.data, self.cfg.vocab_name)
        assert os.path.exists(vocab_path)
        self.asr_dict = Dictionary.load(vocab_path)
        self.padding = self.asr_dict.pad()
        self.stage = cfg.stage

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return self.asr_dict

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.asr_dict

    @property
    def dictionaries(self) -> List[Dictionary]:
        return [self.asr_dict]

    @classmethod
    def setup_task(
        cls, cfg: EVCConfig, **kwargs
    ):
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        # hubert v1: pad_audio=True, random_crop=False;
        import json
        if self.cfg.change_content != "":
            with open(f"{self.cfg.data}/{self.cfg.change_content}_wer.json", 'r') as file:
                wer_json = json.load(file)
                print(f"Change Content to {self.cfg.data}/{self.cfg.change_content}_wer.json")
        else:
            wer_json = None
            
        self.datasets[split] = EVCDataset(
            manifest,
            wer_json,
            split=split,
            sample_rate=self.cfg.sample_rate,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            random_crop=self.cfg.random_crop,
            pad_token=self.padding,
            pitch_home=self.cfg.pitch_home,
            label_rate=self.cfg.label_rate,
            fbank_home=self.cfg.fbank_home,
            asr_dict=self.asr_dict,
            audio_home=self.cfg.audio_home,
            stage=self.cfg.stage,
            pad_tgt_length=self.cfg.pad_tgt_length
        )
        logger.info(f"{split} dataloader is length of {len(self.datasets[split])}")

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices

    def build_model(self, cfg, from_checkpoint=False):
        from fairseq import models, quantization_utils
        model = models.build_model(cfg, self, from_checkpoint)
        if self.cfg.pretrain_checkpoint is not None:
            state = torch.load(self.cfg.pretrain_checkpoint)["model"]
            model = load_dhubert_param(model, state)
        if self.cfg.hifi_checkpoint != "":
            model = load_hifi_models(model, self.cfg.hifi_checkpoint)
        if self.cfg.downstream_checkpoint != "":
            model = load_downstream_models(model, self.cfg.downstream_checkpoint)
        if self.cfg.fm_checkpoint != "":
            model = load_fm_models(model, self.cfg.fm_checkpoint)
        if self.cfg.pretrained_checkpoint != "":
            model = load_pretrained_decoder_models(model, self.cfg.pretrained_checkpoint)
        return model

def load_dhubert_param(model, pretrained_ckpt):
    model_state = model.dhubert.state_dict()
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
    model.dhubert.load_state_dict(model_state)
    return model

def load_downstream_models(model, downstream_ckpt):
    state = torch.load(downstream_ckpt)["model"]
    model_state = model.state_dict()
    pretrained_dict = {}
    for k, v in state.items():
        if k.find("dhubert") != -1:
            continue
        if (k.find("spk_") != -1 or k.find("asr_") != -1 or k.find("emo_") != -1) and k in model_state.keys():
            if v.size() == model_state[k].size():
                pretrained_dict[k] = v.to(model_state[k].dtype)
            else:
                pretrained_dict[k] = model_state[k].to(model_state[k].dtype)
                print(k, v.size(), model_state[k].size(), "size error")
    print("loaded: "+str(pretrained_dict.keys()))
    model_state.update(pretrained_dict)
    model.load_state_dict(model_state)
    return model

def load_fm_models(model, fm_ckpt):
    state = torch.load(fm_ckpt)
    model_state = model.fm.state_dict()
    pretrained_dict = {}
    for k, v in state.items():
        if k in model_state.keys():
            if v.size() == model_state[k].size():
                pretrained_dict[k] = v.to(model_state[k].dtype)
            else:
                pretrained_dict[k] = model_state[k].to(model_state[k].dtype)
                print(k, v.size(), model_state[k].size(), "size error")
    print("loaded: "+str(pretrained_dict.keys()))
    model_state.update(pretrained_dict)
    model.fm.load_state_dict(model_state)
    return model

def load_hifi_models(model, hifi_ckpt):
    state = torch.load(hifi_ckpt)["model"]
    model_state = model.state_dict()
    pretrained_dict = {}
    for k, v in state.items():
        if k in model_state.keys():
            if v.size() == model_state[k].size():
                pretrained_dict[k] = v.to(model_state[k].dtype)
            else:
                pretrained_dict[k] = model_state[k].to(model_state[k].dtype)
                print(k, v.size(), model_state[k].size(), "size error")
    print("loaded: "+str(pretrained_dict.keys()))
    model_state.update(pretrained_dict)
    model.load_state_dict(model_state)
    return model

def load_pretrained_decoder_models(model, pretrained_checkpoint):
    state = torch.load(pretrained_checkpoint)["model"]
    model_state = model.state_dict()
    cur_model_keys = deepcopy(list(model_state.keys()))
    pretrained_dict = {}
    for k, v in state.items():
        if k in model_state.keys():
            if v.size() == model_state[k].size():
                pretrained_dict[k] = v.to(model_state[k].dtype)
                cur_model_keys.remove(k)
            else:
                pretrained_dict[k] = model_state[k].to(model_state[k].dtype)
                print(k, v.size(), model_state[k].size(), "size error")
    print("loaded: "+str(pretrained_dict.keys()))
    print("not loaded: "+str(cur_model_keys))
    model_state.update(pretrained_dict)
    model.load_state_dict(model_state)
    return model