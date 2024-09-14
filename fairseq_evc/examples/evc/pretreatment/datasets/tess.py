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
        "happy": "happy",
        "disgust": "disgust",
        "angry": "angry",
        "ps": "surprised",
        "fear": "fear",
        "neutral": "neutral",
    }
    
    # https://tspace.library.utoronto.ca/handle/1807/24487
    data_name = "TESS"
    root = os.path.join(args.data_home, data_name)
    search_path = os.path.join(root, "**/*." + "wav")
    
    # name, sample_rate, length, emo, trans
    with open(os.path.join(root, "info.tsv"), "w") as train_f:
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
                
            spk, trans, emo = os.path.basename(file_path).split(".")[0].split("_")
            
            emo = emo_dict[emo]
            trans = "say the word %s"%trans
            
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), spk, emo, "_", trans), file=train_f
            )
