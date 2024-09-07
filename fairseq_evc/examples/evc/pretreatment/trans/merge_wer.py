import os
from tqdm import tqdm
import random
import json
import sentencepiece as spm
import re
import soundfile as sf

def normalize_text(text):
    # 使用正则表达式替换掉不规则的 Unicode 字符
    normalized_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xa0]', ' ', text)
    normalized_text = normalized_text.replace('，', ',').replace('。', '.').replace('；', ';').replace('：', ':').replace('？', '?').replace('！', '!').replace('（', '(').replace('）', ')').replace('【', '[').replace('】', ']').replace('“', "'").replace('”', '"').replace('‘', "'").replace('’', "'")
    normalized_text = re.sub(r"[^\w\s']", "", normalized_text)
    return normalized_text.upper()

sp = spm.SentencePieceProcessor()
sp.load('/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/sp_model_10000.model')

with open(r"/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/ori_emo/eval_asr/path.txt", "r") as rf:
    lines = rf.readlines()
idx2path = {}
infos = {}
for line in tqdm(lines):
    if len(line.strip()) < 1:
        continue
    idx, path = line.strip().split(" ")[0], " ".join(line.strip().split(" ")[1:])
    audio, sr = sf.read(path)
    if sr != 16000:
        path = path.replace(
            "/CDShare2/2023/wangtianrui/dataset/emo",
            "/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/resampled16k"
        ).replace(
            "/CDShare3/2023/wangtianrui",
            "/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/resampled16k"
        ).split(".")[0] + ".wav"
    idx2path[int(idx)] = path
idx2trans = {}
with open(r"/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/ori_emo/eval_asr/wer_detail.log", "r") as rf:
    lines = rf.readlines()
    last_id = -1
    for line in tqdm(lines):
        if len(line.strip()) < 1:
            continue
        if line.startswith("utt"):
            id = int(line.strip().split(" ")[-1])
        if line.startswith("WER"):
            wer = float(line.strip().split(": ")[-1].split(" %")[0])
        if line.startswith("lab"):
            lab = line.strip().replace("lab: ", "")
        if line.startswith("rec"):
            rec = line.strip().replace("rec: ", "")
            if id != last_id:
                idx2trans[idx2path[id]] = {
                    "lab": " ".join(sp.encode_as_pieces(normalize_text(lab))), 
                    "rec": " ".join(sp.encode_as_pieces(normalize_text(rec))), 
                    "wer":wer
                }
                last_id = id

with open('/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/ori_emo/eval_asr/wer.json', 'w') as json_file:
    json.dump(idx2trans, json_file, indent=5, ensure_ascii=False)
        