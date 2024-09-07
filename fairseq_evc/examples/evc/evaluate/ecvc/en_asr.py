import whisper
import torch
import argparse
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer
import os
import re

def normalize_text(text):
    # 使用正则表达式替换掉不规则的 Unicode 字符
    normalized_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xa0]', ' ', text)
    normalized_text = normalized_text.replace('，', ',').replace('。', '.').replace('；', ';').replace('：', ':').replace('？', '?').replace('！', '!').replace('（', '(').replace('）', ')').replace('【', '[').replace('】', ']').replace('“', "'").replace('”', '"').replace('‘', "'").replace('’', "'")
    normalized_text = re.sub(r"[^\w\s']", "", normalized_text)
    return normalized_text

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
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-t', '--tsv', type=str)
    parser.add_argument('-o', '--out_home', type=str)
    args = parser.parse_args()
    
    model = whisper.load_model("large-v3", download_root=r"/Work21/2023/cuizhongjian/python/adapter/whisper/model/")
    
    with open(args.tsv, "r") as rf:
        lines = rf.readlines()
    norm = EnglishTextNormalizer()
    if os.path.exists(os.path.join(args.out_home, "wer_detail.log")):
        with open(os.path.join(args.out_home, "wer_detail.log"), "r") as rf:
            if len(rf.readlines()) > 500:
                exit()
    # if os.path.exists(os.path.join(args.out_home, "ref.txt")) and os.path.exists(os.path.join(args.out_home, "est.txt")):
    #     exit()
        
    with open(os.path.join(args.out_home, "ref.txt"), "w") as ref_f:
        with open(os.path.join(args.out_home, "est.txt"), "w") as est_f:
            for idx, line in enumerate(tqdm(lines)):
                path, trans, dur, emo, trans_len, spk, \
                tgt_path, tgt_trans, tgt_dur, tgt_emo, tgt_translen, tgt_spk = line.strip().split("\t")
                save_name = str(idx) + '_' + os.path.basename(path).split(".")[0] + "_srcspk%s"%spk + "_tgtspk%s"%tgt_spk + "_srcemo%s"%emo + "_tgtemo%s"%tgt_emo + ".wav"
                generated_wav = os.path.join(args.out_home, save_name)
                est_txt = recognition(model, generated_wav)
                
                est_txt = normalize_text(norm(est_txt.lower()))
                src_txt = normalize_text(norm(trans.lower()))
                print("%s %s"%(save_name, est_txt), file=est_f)
                print("%s %s"%(save_name, src_txt), file=ref_f)
        