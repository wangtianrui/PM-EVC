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
# https://enterface.net/enterface05/main.php?frame=emotion
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
        "sadness": "sad",
        "happiness": "happy",
        "disgust": "disgust",
        "anger": "angry",
        "surprise": "surprised",
        "fear": "fear"
    }
    
    trans_dict = {
        "anger/sentence 1": "What??? No, no, no, listen! I need this money!",
        "anger/sentence 2": "I don't care about your coffee! Please serve me!",
        "anger/sentence 3": "I can have you fired you know!",
        "anger/sentence 4": "Is your coffee more important than my money?",
        "anger/sentence 5": "You're getting paid to work, not drink coffee!",
        
        "disgust/sentence 1": "That's horrible! I'll never eat noodles again.",
        "disgust/sentence 2": "Something is moving inside my plate",
        "disgust/sentence 3": "Aaaaah a cockroach!!!",
        "disgust/sentence 4": "Eeeek, this is disgusting!!!",
        "disgust/sentence 5": "That's gross!",
        
        "fear/sentence 1": "Oh my god, there is someone in the house!",
        "fear/sentence 2": "Someone is climbing up the stairs",
        "fear/sentence 3": "Please don't kill me...",
        "fear/sentence 4": "I'm not alone! Go away!",
        "fear/sentence 5": "I have nothing to give you! Please don't hurt me!",
        
        "happiness/sentence 1": "That's great, I'm rich now!!!",
        "happiness/sentence 2": "I won: this is great! I’m so happy!!",
        "happiness/sentence 3": "Wahoo... This is so great.",
        "happiness/sentence 4": "I'm so lucky!",
        "happiness/sentence 5": "I'm so excited!",
        
        "sadness/sentence 1": "Life won't be the same now",
        "sadness/sentence 2": "Oh no, tell me this is not true, please!",
        "sadness/sentence 3": "Everything was so perfect! I just don't understand!",
        "sadness/sentence 4": "I still loved him (her)",
        "sadness/sentence 5": "He (she) was my life",
        
        "surprise/sentence 1": "You have never told me that!",
        "surprise/sentence 2": "I didn't expect that!",
        "surprise/sentence 3": "Wahoo, I would never have believed this!",
        "surprise/sentence 4": "I never saw that coming!",
        "surprise/sentence 5": "Oh my God, that’s so weird!",
    }
    
    meta_infos = pd.read_csv(r"/CDShare2/2023/wangtianrui/dataset/emo/EMNS/metadata.csv", sep='|')
    print(meta_infos.head(5))
    
    root = r"/CDShare2/2023/wangtianrui/dataset/emo/eNTERFACE/enterface database"
    search_path = os.path.join(root, "**/*." + "avi")
    
    # name, sample_rate, length, emo, trans
    with open(os.path.join(root, "info.tsv"), "w") as train_f:
        for fname in tqdm(glob.iglob(search_path, recursive=True)):
            file_path = os.path.realpath(fname)
            file_path = convert2wav(file_path)
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio[0]
            
            spk = find_spk(file_path)[0]
            if "6" == spk:
                continue
            for key in trans_dict.keys():
                if file_path.find(key) != -1:
                    trans = trans_dict[key]
                    emo = key.split("/")[0]
                    break
            
            print(file_path)
            emo = emo_dict[emo]
            
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), spk, emo, "_", trans), file=train_f
            )
