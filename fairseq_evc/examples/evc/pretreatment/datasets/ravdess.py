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
# https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

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
    emo_dict = {
        "04": "sad",
        "03": "happy",
        "07": "disgust",
        "05": "angry",
        "08": "surprised",
        "06": "fear",
        "01": "neutral",
        "02": "calm",
    }
    
    trans_dict = {
        "01": "Kids are talking by the door",
        "02": "Dogs are sitting by the door",
    }

    
    root = r"/CDShare2/2023/wangtianrui/dataset/emo/RAVDESS"
    search_path = os.path.join(root, "**/*." + "wav")
    
    # name, sample_rate, length, emo, trans
    with open(os.path.join(root, "info.tsv"), "w") as train_f:
        for fname in tqdm(glob.iglob(search_path, recursive=True)):
            file_path = os.path.realpath(fname)
            if file_path.find("audio_speech_actors_01-24") != -1:
                continue
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio[0]
            
            mod, vocal, emo, level, trans, rep, spk = os.path.basename(file_path).split(".")[0].split("-")
            
            print(file_path)
            emo = emo_dict[emo]
            trans = trans_dict[trans]
            
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), spk, emo, level, trans), file=train_f
            )
