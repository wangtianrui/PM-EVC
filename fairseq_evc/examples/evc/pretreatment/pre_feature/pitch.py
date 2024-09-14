import os
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa as lib
from multiprocessing import Process
import pyworld as pw # pip install pyworld
import numpy as np

def get_all_wavs(root):
    files = []
    for p in Path(root).iterdir():
        if str(p).endswith(".wav"):
            files.append(str(p))
        for s in p.rglob('*.wav'):
            files.append(str(s))
        if str(p).endswith(".flac"):
            files.append(str(p))
        for s in p.rglob('*.flac'):
            files.append(str(s))
    return list(set(files))

def save_numpy(save_path, f0):
    '''Function to write audio'''
    save_path = os.path.abspath(save_path)
    destdir = os.path.dirname(save_path)
    if not os.path.exists(destdir):
        try:
            os.makedirs(destdir)
        except:
            pass
    np.save(save_path, f0)
    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument('--data-home', type=str)
    parser.add_argument('--cluster-speaker-num', type=int)
    args = parser.parse_args()
    info = args.data_home+f"/english_emo_data/all_info_with_cluster_{args.cluster_speaker_num}_spk.tsv"
    with open(info, "r") as rf:
        for line in tqdm(rf.readlines()):
            path = line.split("\t")[0]
            sr = line.split("\t")[1]
            save_path = path.replace(
                    args.data_home,
                    args.data_home+"/pitch"
                ).split(".")[0] + ".npy"
            if os.path.exists(save_path):
                if np.load(save_path).shape[0] != 80:
                    continue
            ori_audio, sr = sf.read(path)
            if sr != 16000:
                ori_audio = lib.resample(ori_audio, orig_sr=sr, target_sr=16000)
            if len(ori_audio.shape) == 2:
                ori_audio = ori_audio[:, 0]
            # print(save_path, ori_audio.shape)
            frame_period = 320/16000*1000
            f0, timeaxis = pw.dio(ori_audio.astype('float64'), 16000, frame_period=frame_period)
            save_numpy(save_path, f0)