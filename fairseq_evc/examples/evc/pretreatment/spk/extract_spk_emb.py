import wespeaker
import torch
import os
import numpy as np
from tqdm import tqdm
import librosa as lib
import soundfile as sf

def npywrite(destpath, arr):
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    np.save(destpath, arr)

def audiowrite(destpath, audio, sample_rate=16000):
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    sf.write(destpath, audio, sample_rate)
    return

if __name__ == "__main__":
    info = r"/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/all_info.tsv"
    model = wespeaker.load_model_local('/Work20/2023/wangtianrui/model_temp/wespeaker/voxceleb_resnet293_LM')
    model.set_gpu(0)
    # model.set_vad(True)
    with torch.no_grad():
        with open(info, "r") as rf:
            for line in tqdm(rf.readlines()):
                path = line.split("\t")[0]
                sr = line.split("\t")[1]
                save_path = path.replace(
                    "/CDShare2/2023/wangtianrui/dataset/emo",
                    "/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/spk_emb"
                ).replace(
                    "/CDShare3/2023/wangtianrui",
                    "/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/spk_emb"
                ).split(".")[0] + ".npy"
                wav_16k, sr = lib.load(path, sr=16000)
                # print(path)
                embedding = model.extract_embedding(path).detach().cpu().numpy()
                npywrite(save_path, embedding)
        