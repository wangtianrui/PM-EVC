import whisper
import torch
import argparse
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer
import os

def recognition(model, wav_path):
    with torch.no_grad():
        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
        options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False)
        result = whisper.decode(model, mel, options)
        return result.text

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Example script with argparse")
    # Add arguments
    parser.add_argument('--data-home', type=str)
    args = parser.parse_args()
    
    model = whisper.load_model("large-v3")
    norm = EnglishTextNormalizer()
    
    info = args.data_home+"/english_emo_data/all_info.tsv"
    with torch.no_grad():
        with open(args.data_home+"/english_emo_data/trans_ori.tsv", "w") as o_wf:
            with open(args.data_home+"/english_emo_data/trans_whisper.tsv", "w") as t_wf:
                with open(info, "r") as rf:
                    all_infos = rf.readlines()
                    for line in tqdm(all_infos):
                        temp = line.split("\t")
                        if len(temp) != 8:
                            (path, sr, length, spk, emo, level), o_trans, emo2 = temp[:6], " ".join(temp[7:-1]), temp[-1]
                        else:
                            path, sr, length, spk, emo, level, o_trans, emo2 = temp
                        trans = recognition(model, path)
                        print("%s %s"%(path, norm(o_trans)), file=o_wf)
                        print("%s %s"%(path, norm(trans)), file=t_wf)
                        