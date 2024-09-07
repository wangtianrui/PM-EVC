import os
import pandas
import numpy as np
import random
from tqdm import tqdm
from collections import Counter

random.seed(42)

tsv_list = [
    ("/CDShare2/2023/wangtianrui/dataset/emo/CREMA-D/info.tsv", "cremad"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/EMNS/info.tsv", "emns"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/EmoV-DB/EmoV-DB_sorted/info.tsv", "emovdb"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/eNTERFACE/enterface database/info.tsv", "enterface"),
    ("/CDShare3/2023/wangtianrui/IEMOCAP/info.tsv", "iemocap"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/LJ Corpus/Raw JL corpus (unchecked and unannotated)/info.tsv", "jl"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/mead/info.tsv", "mead"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/MELD/MELD_Raw/train_info.tsv", "meld"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/MELD/MELD_Raw/test_info.tsv", "meld"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/MELD/MELD_Raw/dev_info.tsv", "meld"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/RAVDESS/info.tsv", "ravdess"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/TESS/info.tsv", "tess"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/Emotional_Speech_Dataset/train_info.tsv", "esd"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/Emotional_Speech_Dataset/test_info.tsv", "esd"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/Emotional_Speech_Dataset/dev_info.tsv", "esd"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/MSP/podcast/train_info.tsv", "msp"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/MSP/podcast/test_info.tsv", "msp"),
    ("/CDShare2/2023/wangtianrui/dataset/emo/MSP/podcast/dev_info.tsv", "msp"),
]

"""
高能量正面情感：happy、excited、amused、assertive、encouraging、surprised、concerned
低能量正面情感：sleepiness、neutral、calm、
高能量负面情感：angry、sarcastic、fear、anxious
低能量负面情感：sad、apologetic、disgust、contempt、frustration
"""

emo_dict = {
    "happy": "0-1",
    "excited": "0-1",
    "amused": "0-1",
    "assertive": "0-1",
    "encouraging": "0-1",
    "surprised": "0-1",
    "concerned": "0-1",
    
    "sleepiness": "0-0",
    "neutral": "0-0",
    "calm": "0-0",
    
    "angry": "1-1",
    "sarcastic": "1-1",
    "fear": "1-1",
    "anxious": "1-1",
    "contempt": "1-1",
    
    "sad": "1-0",
    "apologetic": "1-0",
    "disgust": "1-0",
    "frustration": "1-0",
}

emo_dict_uniq = {'neutral': 0, 'disgust': 1, 'angry': 2, 'happy': 3, 'fear': 4, 'sad': 5, 'surprised': 6, 'contempt': 7}

emo_count = {}
all_info = []
for tsv, name in tsv_list:
    with open(tsv, "r") as rf:
        for line in tqdm(rf.readlines()):
            line = line.strip()
            # print(line)
            if len(line.split("\t")) != 7:
                (path, sr, length, spk, emo, level), trans = line.split("\t")[:6], " ".join(line.split("\t")[7:])
            else:
                path, sr, length, spk, emo, level, trans = line.split("\t")
            if emo in ["unknow", "other", "unk"]:
                continue
            if int(length) / int(sr) < 1:
                continue
            if emo not in emo_dict_uniq.keys():
                continue
            all_info.append("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s"%(path, sr, length, name+"|"+str(spk), emo, level, trans, emo_dict_uniq[emo]))
            if emo not in emo_count.keys():
                emo_count[emo] = 1
            else:
                emo_count[emo] += 1
random.shuffle(all_info)

print(emo_count)
root = r"/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data"
with open(os.path.join(root, "all_info_oriemo.tsv"), "w") as wf:
    for line in all_info:
        print(line.strip(), file=wf)

"""
{'neutral': 69211, 'disgust': 10718, 'angry': 24318, 'happy': 42394, 'fear': 8689, 'sad': 20645, 'sarcastic': 142, 'excited': 1434, 'surprised': 15493, 'sleepiness': 1720, 'amused': 1317, 'frustration': 1842, 'assertive': 240, 'encouraging': 240, 'concerned': 240, 'anxious': 240, 'apologetic': 240, 'contempt': 9337, 'calm': 190}
{'neutral': 69211, 'disgust': 10718, 'angry': 24318, 'happy': 42394, 'fear': 8689, 'sad': 20645, 'surprised': 15493, 'contempt': 9337}
"""