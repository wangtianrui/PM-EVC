import argparse
import glob
import os
import random
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import numpy as np
from moviepy.editor import AudioFileClip
import ffmpeg
import re
from scipy.io.wavfile import write as write_wav
import librosa as lib

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

# 
def convert2wav(path):
    save_path = path.split(".")[0] + ".wav"
    if os.path.exists(save_path):
        return save_path
    stream = ffmpeg.input(path)
    stream = ffmpeg.output(stream, save_path, ar=16000, ac=1)
    ffmpeg.run(stream)
    return save_path

def find_spk(text):
    pattern = r'\bsubject ([0-9]|[1-4][0-9]|50)\b'
    matches = re.findall(pattern, text)
    return matches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument('--data-home', type=str)
    args = parser.parse_args()
    emo_dict = {
        "sad": "sad",
        "neutral": "neutral",
        "happy": "happy",
        "excited": "excited",
        "encouraging": "encouraging",
        "concerned": "concerned",
        "assertive": "assertive",
        "apologetic": "apologetic",
        "anxious": "anxious",
        "angry": "angry",
    }
    # download data from https://www.kaggle.com/tli725/jl-corpus
    data_name = "LJ Corpus"
    root = os.path.join(args.data_home, data_name, "Raw JL corpus (unchecked and unannotated)/JL(wav+txt)")
    search_path = os.path.join(root, "**/*." + "wav")
    
    # name, sample_rate, length, spk, emo, level, trans
    with open(os.path.join(root, "../", "info.tsv"), "w") as train_f:
        for fname in tqdm(glob.iglob(search_path, recursive=True)):
            file_path = os.path.realpath(fname)
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio[0]
            if sr != 16000:
                file_path = file_path.replace(f"/{data_name}/", f"/{data_name}_16k/")
                audio = lib.resample(audio, orig_sr=sr, target_sr=16000)
                save_wav(file_path, audio)
                sr = 16000
            
            spk, emo, _, _ = os.path.basename(file_path).split("_")
            with open(file_path.split(".")[0]+".txt", "r") as rf:
                trans = rf.readline().strip()
            
            emo = emo_dict[emo]
            
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), spk, emo, "_", trans), file=train_f
            )
