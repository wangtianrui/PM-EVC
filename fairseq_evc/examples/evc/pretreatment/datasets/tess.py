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
# https://tspace.library.utoronto.ca/handle/1807/24487
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
        "sad": "sad",
        "happy": "happy",
        "disgust": "disgust",
        "angry": "angry",
        "ps": "surprised",
        "fear": "fear",
        "neutral": "neutral",
    }
    
    
    root = r"/CDShare2/2023/wangtianrui/dataset/emo/TESS"
    search_path = os.path.join(root, "**/*." + "wav")
    
    # name, sample_rate, length, emo, trans
    with open(os.path.join(root, "info.tsv"), "w") as train_f:
        for fname in tqdm(glob.iglob(search_path, recursive=True)):
            file_path = os.path.realpath(fname)
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio[0]
            
            spk, trans, emo = os.path.basename(file_path).split(".")[0].split("_")
            
            emo = emo_dict[emo]
            trans = "say the word %s"%trans
            
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), spk, emo, "_", trans), file=train_f
            )
