import argparse
import glob
import os
import random
import soundfile as sf
import librosa as lib
from tqdm import tqdm
from scipy.io.wavfile import write as write_wav

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument('--data-home', type=str)
    args = parser.parse_args()
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
    # download data from https://github.com/CheyneyComputerScience/CREMA-D
    data_name = "CREMA-D"
    root = os.path.join(args.data_home, data_name)
    search_path = os.path.join(root, "**/*." + "wav")
    
    # name, sample_rate, length, emo, trans
    with open(os.path.join(root, "info.tsv"), "w") as train_f:
        for fname in tqdm(glob.iglob(search_path, recursive=True)):
            file_path = os.path.realpath(fname)
            audio, sr = sf.read(fname)
            if len(audio.shape) > 1:
                audio = audio[0]
            name = os.path.basename(file_path).split(".")[0]
            if sr != 16000:
                file_path = file_path.replace(f"/{data_name}/", f"/{data_name}_16k/")
                audio = lib.resample(audio, orig_sr=sr, target_sr=16000)
                save_wav(file_path, audio)
                sr = 16000
            spk, trans, emo, level = name.split("_")
            trans = trans_dict[trans]
            emo = emo_dict[emo]
            level = emo_level_dict[level]
            
            print(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_path, sr, len(audio), str(spk), emo, level, trans), file=train_f
            )
