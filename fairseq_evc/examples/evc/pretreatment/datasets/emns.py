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
# https://www.openslr.org/136
def convert2wav(path):
    save_path = path.split(".")[0] + ".wav"
    if os.path.exists(save_path):
        return save_path
    stream = ffmpeg.input(path)
    stream = ffmpeg.output(stream, save_path, ar=16000)
    ffmpeg.run(stream)
    return save_path

if __name__ == "__main__":
    emo_dict = {
        "Sad": "sad",
        "Neutral": "neutral",
        "Happy": "happy",
        "Disgust": "disgust",
        "Angry": "angry",
        "Excited": "excited",
        "Surprised": "surprised",
        "Sarcastic": "sarcastic"
    }
    
    emo_level_dict = {
        "LO": "low",
        "MD": "medium",
        "HI": "high",
        "XX": "unspecified"
    }
    
    meta_infos = pd.read_csv(r"/CDShare2/2023/wangtianrui/dataset/emo/EMNS/metadata.csv", sep='|')
    print(meta_infos.head(5))
    
    root = r"/CDShare2/2023/wangtianrui/dataset/emo/EMNS/"
    
    # name, sample_rate, length, emo, trans
    with open(os.path.join(root, "info.tsv"), "w") as train_f:
        for id, utterance, _, emo, _, _, gender, age, level, name, spk in tqdm(np.array(meta_infos)):
            file_path = os.path.join(root, "raw_webm/raw_webm", os.path.basename(name))
            
            file_path = convert2wav(file_path)
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio[0]
            
            emo = emo_dict[emo]
            
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), spk, emo, level, utterance), file=train_f
            )
