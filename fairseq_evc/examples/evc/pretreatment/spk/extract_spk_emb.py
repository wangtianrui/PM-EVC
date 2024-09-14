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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument('--data-home', type=str)
    parser.add_argument('--speaker-model', type=str)
    args = parser.parse_args()
    info = os.path.join(args.data_home, "english_emo_data/all_info.tsv")
    model = wespeaker.load_model_local(args.speaker_model)
    model.set_gpu(0)
    # model.set_vad(True)
    with torch.no_grad():
        with open(info, "r") as rf:
            for line in tqdm(rf.readlines()):
                path = line.split("\t")[0]
                sr = line.split("\t")[1]
                save_path = path.replace(
                    args.data_home,
                    args.data_home+"english_emo_data/spk_emb"
                ).split(".")[0] + ".npy"
                wav_16k, sr = lib.load(path, sr=16000)
                # print(path)
                embedding = model.extract_embedding(path).detach().cpu().numpy()
                npywrite(save_path, embedding)
        