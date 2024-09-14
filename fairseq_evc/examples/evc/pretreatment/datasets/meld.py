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

from pathlib import Path
def get_all_wavs(root, suffix):
    files = []
    for p in Path(root).iterdir():
        if str(p).endswith(".%s"%suffix):
            files.append(str(p))
        for s in p.rglob("*.%s"%suffix):
            files.append(str(s))
    return list(set(files))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument('--data-home', type=str)
    args = parser.parse_args()
    emo_dict = {
        "sadness": "sad",
        "joy": "happy",
        "disgust": "disgust",
        "anger": "angry",
        "surprise": "surprised",
        "fear": "fear",
        "neutral": "neutral",
        "contempt": "contempt",
    }
    # https://affective-meld.github.io/
    data_name = "MELD"
    root = os.path.join(args.data_home, data_name)
    
    with open(os.path.join(root, "MELD_Raw", "train_info.tsv"), "w") as train_f:
        df = pd.read_csv(os.path.join(root, "MELD_Raw", "train_sent_emo.csv"))
        for No,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime in np.array(df):
            file_path = os.path.join(
                os.path.join(root, "MELD_Raw"),
                "train_splits",
                "dia%s_utt%s.mp4"%(Dialogue_ID, Utterance_ID),
            )
            if file_path.find(r"dia125_utt3.mp4") != -1:
                continue
            file_path = convert2wav(file_path)
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio[0]
            trans = Utterance
            emo = emo_dict[Emotion]
            
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), Speaker, emo, "_", trans), file=train_f
            )
    
    with open(os.path.join(root, "MELD_Raw", "test_info.tsv"), "w") as train_f:
        df = pd.read_csv(os.path.join(root, "MELD_Raw", "test_sent_emo.csv"))
        for No,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime in np.array(df):
            file_path = os.path.join(
                os.path.join(root, "MELD_Raw"),
                "output_repeated_splits_test",
                "dia%s_utt%s.mp4"%(Dialogue_ID, Utterance_ID),
            )
            if file_path.find(r"dia125_utt3.mp4") != -1:
                continue
            file_path = convert2wav(file_path)
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio[0]
            trans = Utterance
            emo = emo_dict[Emotion]
            
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), Speaker, emo, "_", trans), file=train_f
            )
            
    with open(os.path.join(root, "MELD_Raw", "dev_info.tsv"), "w") as train_f:
        df = pd.read_csv(os.path.join(root, "MELD_Raw", "dev_sent_emo.csv"))
        for No,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime in np.array(df):
            file_path = os.path.join(
                os.path.join(root, "MELD_Raw"),
                "dev_splits_complete",
                "dia%s_utt%s.mp4"%(Dialogue_ID, Utterance_ID),
            )
            if file_path.find(r"dia110_utt7.mp4") != -1:
                continue
            file_path = convert2wav(file_path)
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio[0]
            trans = Utterance
            emo = emo_dict[Emotion]
            
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), Speaker, emo, "_", trans), file=train_f
            )
