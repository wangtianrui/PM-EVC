import argparse
import glob
import os
import random
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import numpy as np
import ffmpeg
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

def convert2wav(path):
    save_path = path.split(".")[0] + ".wav"
    if os.path.exists(save_path):
        return save_path
    stream = ffmpeg.input(path)
    stream = ffmpeg.output(stream, save_path, ar=16000)
    ffmpeg.run(stream)
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument('--data-home', type=str)
    args = parser.parse_args()
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
    # download data from https://www.openslr.org/136
    data_name = "EMNS"
    root = os.path.join(args.data_home, data_name)
    
    meta_infos = pd.read_csv(os.path.join(root, data_name, "metadata.csv"), sep='|')
    # name, sample_rate, length, emo, trans
    with open(os.path.join(root, "info.tsv"), "w") as train_f:
        for id, utterance, _, emo, _, _, gender, age, level, name, spk in tqdm(np.array(meta_infos)):
            file_path = os.path.join(root, "raw_webm/raw_webm", os.path.basename(name))
            
            file_path = convert2wav(file_path)
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio[0]
            
            if sr != 16000:
                file_path = file_path.replace(f"/{data_name}/", f"/{data_name}_16k/")
                audio = lib.resample(audio, orig_sr=sr, target_sr=16000)
                save_wav(file_path, audio)
                sr = 16000
            
            emo = emo_dict[emo]
            
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), spk, emo, level, utterance), file=train_f
            )
