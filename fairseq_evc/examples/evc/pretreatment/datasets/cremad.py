import argparse
import glob
import os
import random
import soundfile as sf
from tqdm import tqdm
# https://github.com/CheyneyComputerScience/CREMA-D
if __name__ == "__main__":
    trans_dict = {
        "IEO": "It's eleven o'clock",
        "TIE": "That is exactly what happened",
        "IOM": "I'm on my way to the meeting",
        "IWW": "I wonder what this is about",
        "TAI": "The airplane is almost fullk",
        "MTI": "Maybe tomorrow it will be cold",
        "IWL": "I would like a new alarm clock",
        "ITH": "I think I have a doctor's appointment",
        "DFA": "Don't forget a jacket",
        "ITS": "I think I've seen this before",
        "TSI": "The surface is slick",
        "WSI": "We'll stop in a couple of minutes",
    }
    
    emo_dict = {
        "SAD": "sad",
        "NEU": "neutral",
        "HAP": "happy",
        "FEA": "fear",
        "DIS": "disgust",
        "ANG": "angry",
    }
    
    emo_level_dict = {
        "LO": "low",
        "MD": "medium",
        "HI": "high",
        "XX": "unspecified"
    }
    
    root = r"/CDShare2/2023/wangtianrui/dataset/emo/CREMA-D"
    search_path = os.path.join(root, "**/*." + "wav")
    
    # name, sample_rate, length, emo, trans
    with open(os.path.join(root, "info.tsv"), "w") as train_f:
        for fname in tqdm(glob.iglob(search_path, recursive=True)):
            file_path = os.path.realpath(fname)
            audio, sr = sf.read(fname)
            if len(audio.shape) > 1:
                audio = audio[0]
            
            name = os.path.basename(file_path).split(".")[0]
            
            spk, trans, emo, level = name.split("_")
            trans = trans_dict[trans]
            emo = emo_dict[emo]
            level = emo_level_dict[level]
            
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), str(spk), emo, level, trans), file=train_f
            )
