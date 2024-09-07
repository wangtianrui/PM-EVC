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
import textgrid
from praatio import textgrid
# https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html
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
        "S": "sad",
        "N": "neutral",
        "H": "happy",
        "F": "fear",
        "U": "surprised",
        "O": "other",
        "X": "unknow",
        "C": "contempt",
        "D": "disgust",
        "A": "angry",
    }
    
    import json
    # 打开JSON文件并读取数据
    with open('/CDShare2/2023/wangtianrui/dataset/emo/MSP/podcast/Labels/labels_consensus.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    root = r"/CDShare2/2023/wangtianrui/dataset/emo/MSP/podcast"
    
    # name, sample_rate, length, spk, emo, level, trans
    with open(os.path.join(root, "train_info.tsv"), "w") as train_f:
        with open(os.path.join(root, "test_info.tsv"), "w") as test_f:
            with open(os.path.join(root, "dev_info.tsv"), "w") as dev_f:
                with open(r"/CDShare2/2023/wangtianrui/dataset/emo/MSP/podcast/Partitions.txt", "r") as rf:
                    for line in tqdm(rf.readlines()):
                        split, name = line.strip().split(";")
                        name = name.strip()
                        file_path = os.path.join(root, "Audios", name)
                        audio, sr = sf.read(file_path)
                        if len(audio.shape) > 1:
                            audio = audio[0]
                        
                        if name not in data:
                            print(name, "miss data")
                            continue
                        emo = emo_dict[data[name]["EmoClass"]]
                        spk = data[name]["SpkrID"]
                        
                        tg_path = os.path.join(root, "ForceAligned", name.split(".")[0]+".TextGrid")
                        tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
                        # 获取所有 tier 的名称
                        tier = tg.getTier("words")
                        trans = " ".join([label for start_time, end_time, label in tier.entries if label != ""])
                        
                        if split.find("Train") != -1:
                            out_file = train_f
                        elif split.find("Test") != -1:
                            out_file = test_f
                        else:
                            out_file = dev_f
                        
                        print(
                            "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), spk, emo, "_", trans), file=out_file
                        )
