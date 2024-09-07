import argparse
import glob
import os
import random
import soundfile as sf
from tqdm import tqdm
import numpy as np
import re
import whisper
import torch
import argparse
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer
import os
# https://mega.nz/#F!KBp32apT!gLIgyWf9iQ-yqnWFUFuUHg!mYwUnI4K
def load_tsv(path):
    with open(path, "r") as rf:
        lines = rf.readlines()
    return lines

def recognition(model, wav_path):
    with torch.no_grad():
        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
        options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False)
        result = whisper.decode(model, mel, options)
        return result.text
    
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
    emo_dict = {
        "Disgust": "disgust",
        "disgust": "disgust",
        "anger": "angry",
        "sleepiness": "sleepiness",
        "neutral": "neutral",
        "amused": "amused",
    }
    
    model = whisper.load_model("large-v3", download_root=r"/Work21/2023/cuizhongjian/python/adapter/whisper/model/")
    norm = EnglishTextNormalizer()
    
    root = r"/CDShare2/2023/wangtianrui/dataset/emo/EmoV-DB/EmoV-DB_sorted"
    
    with open(os.path.join(root, "info.tsv"), "w") as train_f:
        for path in tqdm(get_all_wavs(root, "wav")):
            path = os.path.realpath(path)
            # path = convert2wav(path)
            spk = path.split("EmoV-DB_sorted/")[-1].split("/")[0]
            print(spk)
            audio, sr = sf.read(path)
            trans = recognition(model, path)
            emo = emo_dict[os.path.basename(path).split("_")[0].lower()]
            # spk = path.
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(path, sr, len(audio), spk, emo, "_", trans), file=train_f
            )
        