# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Any, List, Optional, Union
from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
import torchaudio

logger = logging.getLogger(__name__)

def load_audio(manifest_path, max_keep, min_keep, wer_json):
    infos = []
    lengthes = []
    skiped = 0
    no_wer = 0
    with open(manifest_path, "r") as rf:
        for line in rf.readlines():
            temp = line.split("\t")
            if len(temp) != 8:
                (path, sr, length, spk, emo, level), o_trans, emo2 = temp[:6], " ".join(temp[7:-1]), temp[-1]
            else:
                path, sr, length, spk, emo, level, o_trans, emo2 = temp
            length = int(length)
            if not (length >= min_keep and length <= max_keep):
                skiped += 1
                continue
            if wer_json is not None:
                if path not in wer_json.keys():
                    no_wer += 1
                else:
                    if wer_json[path]["wer"] > 20:
                        o_trans = wer_json[path]["rec"]
            # o_trans = o_trans
            infos.append([path, 16000, length, spk, emo, level, o_trans, emo2])
            lengthes.append(int(int(length)/int(sr)*16000))
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(infos)}, skipped {skiped}"
            f"longest-loaded={max(lengthes)}, shortest-loaded={min(lengthes)},"
            f"no-wer={no_wer}"
        )
    )
    return infos, lengthes

class EVCDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        wer_json:str,
        sample_rate: float,
        split: str,
        pad_token: int,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = False,
        pad_audio: bool = False,
        normalize: bool = False,
        random_crop: bool = False,
        pitch_home=None,
        fbank_home="",
        label_rate=50,
        asr_dict=None,
        audio_home="",
        stage=2,
        pad_tgt_length=-1
    ):  
        self.pad_tgt_length = pad_tgt_length
        self.wer_json = wer_json
        self.stage = stage
        self.infos, self.lengthes = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size, self.wer_json,
        )
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.pad_token = pad_token
        self.pad_audio = pad_audio
        self.normalize = normalize
        self.random_crop = random_crop
        self.max_sample_size = max_sample_size
        self.pitch_home = pitch_home
        self.label_rate = label_rate
        self.label2wav_rate = self.sample_rate // self.label_rate
        self.max_label_size = self.max_sample_size // self.label2wav_rate
        self.fbank_home = fbank_home
        self.audio_home = audio_home.split("$")
        print(self.audio_home)
        self.asr_dict = asr_dict
        assert self.max_sample_size >= max_keep_sample_size, "max_sample_size should be larger than max_keep_sample_size"
        
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )
        
        self.resampleers = {
            44100: torchaudio.transforms.Resample(44100, self.sample_rate).eval(),
            48000: torchaudio.transforms.Resample(48000, self.sample_rate).eval(),
            24000: torchaudio.transforms.Resample(24000, self.sample_rate).eval(),
            8000: torchaudio.transforms.Resample(8000, self.sample_rate).eval(),
            22050: torchaudio.transforms.Resample(22050, self.sample_rate).eval(),
        }

    def get_audio(self, path):
        wav_normed = None
        wav, cur_sample_rate = sf.read(path)
        wav = torch.from_numpy(wav).float()
        if cur_sample_rate != self.sample_rate:
            wav = self.resampleers[cur_sample_rate](wav.unsqueeze(0)).squeeze(0)
        if self.normalize:
            with torch.no_grad():
                wav_normed = F.layer_norm(wav, wav.shape)
        return wav, wav_normed
    
    def __getitem__(self, index):
        path, sr, length, spk, emo, level, o_trans, emo2 = self.infos[index]
        assert int(spk) < 2900, "speaker error: " +spk 
        tgt_asr = self.asr_dict.encode_line(o_trans)
        spk = int(spk)
        if emo2.find("-") != -1:
            emo1 = int(emo2.split("-")[0])
            emo2 = int(emo2.split("-")[1])
        else:
            emo1 = int(emo2.strip())
            emo2 = int(-1)
        
        wav, wav_normed = self.get_audio(path)
        pitch = self.load_pitch(self.change_path2npy(path, self.audio_home, self.pitch_home))
        fbank = None
        if self.fbank_home.find("denoised") != -1:
            path_save = path.split("wangtianrui/")[-1]
            fbank = np.load(self.change_path2npy(os.path.join(self.fbank_home, path_save))).T
        elif os.path.exists(self.fbank_home):
            fbank = np.load(self.change_path2npy(path, self.audio_home, self.fbank_home)).T # T,D
        # print("loaded："+str(index))
        if self.normalize:
            return {"id": index, "source": wav_normed, "pitch": pitch, "target": wav, "fbank": torch.Tensor(fbank), "tgt_asr": tgt_asr, "spk_label": spk, "emo1": emo1, "emo2": emo2}
        else:
            return {"id": index, "source": wav, "pitch": pitch, "target": wav, "fbank": torch.Tensor(fbank), "tgt_asr": tgt_asr, "spk_label": spk, "emo1": emo1, "emo2": emo2}

    def load_pitch(self, filename):
        f0 = np.load(filename)
        nonzeros_indices = np.nonzero(f0)
        if len(nonzeros_indices[0]) == 0:
            return torch.Tensor(f0)
        pitch = f0.copy()
        pitch[nonzeros_indices] = np.log(f0[nonzeros_indices])
        mean, std = np.mean(pitch[nonzeros_indices]), np.std(pitch[nonzeros_indices])
        pitch[nonzeros_indices] = (pitch[nonzeros_indices] - mean) / (std + 1e-8)
        return torch.Tensor(pitch) # 1, D
    
    def change_path2npy(self, path, ori_home=None, tgt_home=None):
        if ori_home is not None and tgt_home is not None:
            for ori_item in ori_home:
                if path.find(ori_item) != -1:
                    path = path.replace(ori_item, tgt_home)
                    break
        return path.replace(".wav", ".npy").replace(".flac", ".npy").replace(".ogg", ".npy").replace(".mp3", ".npy")
    
    def __len__(self):
        return len(self.lengthes)

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        if len(samples) == 0:
            return {}
        # print([str(s["id"])+"-0" for s in samples])
        samples = [s for s in samples if s["source"] is not None]
        audio_lens = [len(s) for s in samples if s["source"] is not None]
        pitchs = [s["pitch"] for s in samples]
        pitch_sizes = [len(s) for s in pitchs] 
        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        target = [s["target"] for s in samples]
        tgt_asr = [s["tgt_asr"] for s in samples]
        tgt_asr_len = torch.IntTensor([v.size(0) for v in tgt_asr])
        spk_label = torch.LongTensor([s["spk_label"] for s in samples])
        emo1 = torch.LongTensor([s["emo1"] for s in samples])
        emo2 = torch.LongTensor([s["emo2"] for s in samples])
        
        # print([str(s["id"])+"-1" for s in samples])
        tgt_asr = data_utils.collate_tokens(
            tgt_asr,
            self.asr_dict.pad(),
            self.asr_dict.eos(),
            False,
            move_eos_to_beginning=False,
        )
        
        if samples[0]["fbank"] is not None:
            fbanks = [s["fbank"] for s in samples]
            fbank_sizes = [s.shape[0] for s in fbanks]
        else:
            fbanks = None
            fbank_sizes = None
        
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
            pitch_size = min(max(pitch_sizes), self.max_label_size)
            if fbanks is not None:
                fbank_size = min(max(fbank_sizes), int(np.floor(self.max_sample_size//320)))
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
            pitch_size = min(min(pitch_sizes), self.max_label_size)
            if fbanks is not None:
                fbank_size = min(min(fbank_sizes), int(np.floor(self.max_sample_size//320)))
        
        collated_audios, padding_mask, collated_pitchs, collated_targets, audio_starts, feature_lens, no_paded_source = self.collater_audio(
            audios, audio_size, pitchs, pitch_size, target
        )
        # print([str(s["id"])+"-2" for s in samples])
        
        if fbanks is not None:
            collated_fbanks = self.collate_frames(fbanks, wav_starts=audio_starts, fbank_size=fbank_size)
        net_input = {
            "source": collated_audios, 
            "padding_mask": padding_mask,
            "pitch": collated_pitchs,
            "target": collated_targets,
            "fbank": collated_fbanks,
            "tgt_content": tgt_asr,
            "tgt_content_len": tgt_asr_len,
            "tgt_spk": spk_label,
            "tgt_emo": torch.stack([emo1, emo2], dim=-1),
            "feature_len": torch.IntTensor(feature_lens),
            "only_downstream": self.stage==1,
            "no_paded_source": no_paded_source,
            "audio_lens": torch.IntTensor(audio_lens)
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }
        # print([str(s["id"])+"-3" for s in samples])
        return batch

    def collater_audio(self, audios, audio_size, pitchs, pitch_size, targets):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        collated_targets = audios[0].new_zeros(len(audios), audio_size)
        collated_pitchs = audios[0].new_zeros(len(pitchs), pitch_size)
        no_paded_source = []
        
        audio_starts = [0 for _ in audios]
        pitch_starts = [0 for _ in pitchs]
        # fbank_starts = [0 for _ in pitchs]
        
        temp_audio_downsampled_lengthes = []
        for i, (pitch, audio, target) in enumerate(zip(pitchs, audios, targets)):
            temp_audio_downsampled_lengthes.append(len(audio)//self.label2wav_rate)
            diff_audio = len(audio) - audio_size
            if diff_audio == 0:
                collated_pitchs[i] = pitch[:pitch_size]
                collated_audios[i] = audio
                collated_targets[i] = target
                no_paded_source.append(audio)
            elif diff_audio < 0:
                assert self.pad_audio
                collated_pitchs[i, :len(pitch)] = pitch[:pitch_size]
                collated_audios[i, :len(audio)] = audio
                collated_targets[i, :len(target)] = target
                no_paded_source.append(audio)
            else:
                collated_pitchs[i], pitch_starts[i], pitch_end = self.crop_to_max_size(pitch, pitch_size)
                audio_starts[i] = pitch_starts[i]*self.label2wav_rate
                # logger.info(str(pitch.size()) + str(audio.size()) + str(pitch_starts[i]) + "," + str(pitch_end)+ str(audio[audio_starts[i]:pitch_end*self.label2wav_rate].size()) + str(collated_audios[i].size()) + str(collated_pitchs.size()))
                collated_audios[i] = audio[audio_starts[i]:audio_starts[i]+audio_size]
                collated_targets[i] = target[audio_starts[i]:audio_starts[i]+audio_size]
                no_paded_source.append(audio[audio_starts[i]:audio_starts[i]+audio_size])
        padding_mask = make_pad_mask(temp_audio_downsampled_lengthes, max_len=max(temp_audio_downsampled_lengthes))
        return collated_audios, padding_mask, collated_pitchs, collated_targets, audio_starts, temp_audio_downsampled_lengthes, no_paded_source


    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0, size

        start, end = 0, target_size
        if self.random_crop and abs(diff) > 2:
            start = np.random.randint(0, diff)
            end = size - diff + start
        return wav[start:end], start, end
    
    def num_tokens(self, index):
        return max(self.size(index), self.pad_tgt_length)

    def size(self, index):
        if self.pad_audio:
            return self.lengthes[index]
        return min(self.lengthes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.lengthes)
        return np.lexsort(order)[::-1] # 长的优先
        # return np.lexsort(order) # 短的优先
        
    def collate_frames(self, frames, wav_starts, fbank_size) -> torch.Tensor:
        # frames
        out = frames[0].new_zeros((len(frames), fbank_size, frames[0].size(1)))
        for i, v in enumerate(frames):
            fbank_start = wav_starts[i] // int(np.floor(self.max_sample_size//320))
            out[i, : v.size(0)] = v[fbank_start:fbank_start+fbank_size]
        return out

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