import os
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa as lib
from multiprocessing import Process
import pyworld as pw
import numpy as np
from scipy.io.wavfile import write as write_wav

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

def save_wav(save_path, audio, sr=16000):
    '''Function to write audio'''

    save_path = os.path.abspath(save_path)
    destdir = os.path.dirname(save_path)

    if not os.path.exists(destdir):
        try:
            os.makedirs(destdir)
        except:
            pass
    write_wav(save_path, sr, audio)
    return

def extract_pitch(flac_paths):
    for flac_path in tqdm(flac_paths):
        save_path = flac_path.replace("/CDShare2/LS_HuBERT/wavcaps/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/", "/CDShare2/LS_HuBERT/wavcaps/16k/").split(".")[0] + ".wav"
        if os.path.exists(save_path):
            continue
        try:
            ori_audio, sr = lib.load(flac_path, sr=16000)
            save_wav(save_path, ori_audio, sr)
        except Exception as e:
            print(flac_path+":"+str(e))
        
if __name__ == "__main__":
    n_p = 32
    all_wavs = get_all_wavs(r"/CDShare2/LS_HuBERT/wavcaps/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/")
    slice_len = len(all_wavs) // n_p
    ps = []
    for i in range(n_p):
        temp_pathes = all_wavs[i*slice_len:(i+1)*slice_len] if i != n_p - 1 else all_wavs[i*slice_len:]
        ps.append(
            Process(target=extract_pitch, args=(temp_pathes, ))
        )
    for i in ps:
        i.start()
    for i in ps:
        i.join()
    
