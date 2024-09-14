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
from pathlib import Path
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

def get_all_wavs(root):
    files = []
    for p in Path(root).iterdir():
        if str(p).endswith(".wav"):
            files.append(str(p))
        for s in p.rglob('*.wav'):
            files.append(str(s))
    return list(set(files))

def get_emo_info(root):
    infos = {}
    trans_dict = {}
    for name in glob.iglob(os.path.join(root, "/Session*/dialog/transcriptions/*.") + "txt", recursive=True):
        with open(name) as rf:
            for line in rf.readlines():
                if line.strip() == "":
                    continue
                print(line)
                name = line.strip().split(" [")[0]
                if not str(name).startswith("Ses"):
                    continue
                start, end = line.strip().split("]:")[0].split(" [")[1].split("-")
                print(start, end)
                trans = line.strip().split("]: ")[-1]
                assert name not in trans_dict.keys(), name
                trans_dict[name+"%.2f-%.2f"%(float(start), float(end))] = trans
    print(trans_dict.keys())
    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
    start_times, end_times, wav_file_names, emotions, vals, acts, doms, trans = [], [], [], [], [], [], [], []
    for sess in range(1, 6):
        emo_evaluation_dir = os.path.join(root, '/Session{}/dialog/EmoEvaluation/'.format(sess))
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
        for file in evaluation_files:
            print(emo_evaluation_dir + file)
            with open(emo_evaluation_dir + file) as f:
                content = f.read()
            info_lines = re.findall(info_line, content)
            for line in info_lines[1:]:  # the first line is a header
                start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)
                trans.append(trans_dict[wav_file_name+"%.2f-%.2f"%(start_time, end_time)])
                infos[wav_file_name] = [emotion, trans_dict[wav_file_name+"%.2f-%.2f"%(start_time, end_time)]]
    df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom', "trans"])
    df_iemocap['start_time'] = start_times
    df_iemocap['end_time'] = end_times
    df_iemocap['wav_file'] = wav_file_names
    df_iemocap['emotion'] = emotions
    df_iemocap['val'] = vals
    df_iemocap['act'] = acts
    df_iemocap['dom'] = doms
    df_iemocap['trans'] = trans
    # print(df_iemocap.head(5))
    return df_iemocap, infos

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
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument('--data-home', type=str)
    args = parser.parse_args()
    emo_dict = {
        "sad": "sad",
        "neu": "neutral",
        "hap": "happy",
        "ang": "angry",
        "fru": "frustration",
        "xxx": "unk",
        "sur": "surprised",
        "exc": "excited",
        "fea": "fear",
        "oth": "other",
        "dis": "disgust",
    }
    
    # download data from https://sail.usc.edu/iemocap/iemocap_release.htm
    data_name = "IEMOCAP"
    root = os.path.join(args.data_home, data_name)
    df, infos = get_emo_info(root)
    print(df.head(5))
    search_path = os.path.join(root, "*/Session*/sentences/wav/*/*." + "wav")
    # name, sample_rate, length, emo, trans
    with open(os.path.join(root, "info.tsv"), "w") as train_f:
        for file_path in tqdm(get_all_wavs(root)):
            if file_path.find("dialog") != -1:
                continue
            basename = os.path.basename(file_path)
            print(file_path)
            emo, trans = infos[basename.split(".")[0]]
            audio, sr = sf.read(file_path)
            if sr != 16000:
                file_path = file_path.replace(f"/{data_name}/", f"/{data_name}_16k/")
                audio = lib.resample(audio, orig_sr=sr, target_sr=16000)
                save_wav(file_path, audio)
                sr = 16000
            if len(audio.shape) > 1:
                audio = audio[0]
            emo = emo_dict[emo]
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), "_", emo, "_", trans), file=train_f
            )
