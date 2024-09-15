from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from models import Generator
import numpy as np
h = None
device = torch.device('cuda')

MAX_WAV_VALUE = 32768.0

import os
import shutil
from pathlib import Path
def get_all_wavs(root):
    files = []
    for p in Path(root).iterdir():
        if str(p).endswith(".npy"):
            files.append(str(p))
        for s in p.rglob('*.npy'):
            files.append(str(s))
    return list(set(files))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def inference(fbank_path, wav_path, ckpt):
    with open(r"./config.json") as f:
        data = f.read()
    h = AttrDict(json.loads(data))
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(ckpt, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    x = torch.Tensor([np.load(fbank_path)]).to(device)
    print(x.size())
    with torch.no_grad():
        y_g_hat = generator(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')

        output_file = wav_path
        write(output_file, h.sampling_rate, audio)
        print(output_file)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--checkpoint_path', type=str)
    parser.add_argument('-f', '--fbank', type=str)
    args = parser.parse_args()
    out_path = args.fbank.replace(".npy", ".wav")
    inference(
        args.fbank,
        out_path,
        args.checkpoint_path
    )
    

