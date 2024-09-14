import os
import pandas
import argparse
import numpy as np
import random
from tqdm import tqdm
from collections import Counter

random.seed(55)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument('--data-home', type=str)
    args = parser.parse_args()
    tsv_list = [
        (args.data_home+"/CREMA-D/info.tsv", "cremad"),
        (args.data_home+"/EMNS/info.tsv", "emns"),
        (args.data_home+"/EmoV-DB_sorted/info.tsv", "emovdb"),
        (args.data_home+"/eNTERFACE/enterface database/info.tsv", "enterface"),
        (args.data_home+"/IEMOCAP/info.tsv", "iemocap"),
        (args.data_home+"/LJ Corpus/Raw JL corpus (unchecked and unannotated)/info.tsv", "jl"),
        (args.data_home+"/mead/info.tsv", "mead"),
        (args.data_home+"/MELD/MELD_Raw/train_info.tsv", "meld"),
        (args.data_home+"/MELD/MELD_Raw/test_info.tsv", "meld"),
        (args.data_home+"/MELD/MELD_Raw/dev_info.tsv", "meld"),
        (args.data_home+"/RAVDESS/info.tsv", "ravdess"),
        (args.data_home+"/TESS/info.tsv", "tess"),
        (args.data_home+"/Emotional_Speech_Dataset/train_info.tsv", "esd"),
        (args.data_home+"/Emotional_Speech_Dataset/test_info.tsv", "esd"),
        (args.data_home+"/Emotional_Speech_Dataset/dev_info.tsv", "esd"),
        (args.data_home+"/MSP/podcast/train_info.tsv", "msp"),
        (args.data_home+"/MSP/podcast/test_info.tsv", "msp"),
        (args.data_home+"/MSP/podcast/dev_info.tsv", "msp"),
    ]

    """
    high-energy positive：happy、excited、amused、assertive、encouraging、surprised、concerned
    low-energy positive：sleepiness、neutral、calm、
    high-energy negative：angry、sarcastic、fear、anxious
    low-engergy negative：sad、apologetic、disgust、contempt、frustration
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
    root = args.data_home+"/english_emo_data"
    try:
        os.makedirs(root)
    except:
        pass
    with open(os.path.join(root, "all_info.tsv"), "w") as wf:
        for line in all_info:
            print(line.strip(), file=wf)

    """
    {'neutral': 69211, 'disgust': 10718, 'angry': 24318, 'happy': 42394, 'fear': 8689, 'sad': 20645, 'sarcastic': 142, 'excited': 1434, 'surprised': 15493, 'sleepiness': 1720, 'amused': 1317, 'frustration': 1842, 'assertive': 240, 'encouraging': 240, 'concerned': 240, 'anxious': 240, 'apologetic': 240, 'contempt': 9337, 'calm': 190}
    {'neutral': 69211, 'disgust': 10718, 'angry': 24318, 'happy': 42394, 'fear': 8689, 'sad': 20645, 'surprised': 15493, 'contempt': 9337}
    """