import os
from tqdm import tqdm
import random
import json
import sentencepiece as spm
import re

def normalize_text(text):
    normalized_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xa0]', ' ', text)
    normalized_text = normalized_text.replace('，', ',').replace('。', '.').replace('；', ';').replace('：', ':').replace('？', '?').replace('！', '!').replace('（', '(').replace('）', ')').replace('【', '[').replace('】', ']').replace('“', "'").replace('”', '"').replace('‘', "'").replace('’', "'")
    normalized_text = re.sub(r"[^\w\s']", "", normalized_text)
    return normalized_text.upper()

with open('/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/spk_emb/km_models/spkdict_100.json', 'r', encoding='utf-8') as file:
    spk_dick = json.load(file)

sp = spm.SentencePieceProcessor()
sp.load('/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/sp_model_10000.model')

info = r"/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/spk_emb/km_models/all_info_100.tsv"
root = os.path.dirname(info)
all_infos_processed = []
with open(info, "r") as rf:
    all_infos = rf.readlines()
for line in tqdm(all_infos):
    temp = line.split("\t")
    if len(temp) != 8:
        (path, sr, length, spk, emo, level), o_trans, emo2 = temp[:6], " ".join(temp[7:-1]), temp[-1]
    else:
        path, sr, length, spk, emo, level, o_trans, emo2 = temp
    save_path_16k = path.replace(
        "/CDShare2/2023/wangtianrui/dataset/emo",
        "/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/resampled16k"
    ).replace(
        "/CDShare3/2023/wangtianrui",
        "/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/resampled16k"
    ).split(".")[0] + ".wav"
    
    if int(sr) != 16000:
        path = save_path_16k
    o_trans = " ".join(sp.encode_as_pieces(normalize_text(o_trans)))
    all_infos_processed.append("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s"%(path, sr, length, spk_dick[spk], emo, level, o_trans, emo2))

random.shuffle(all_infos_processed)
tot_len = len(all_infos_processed)
train_info = all_infos_processed[:-tot_len//5]
dev_info = all_infos_processed[-tot_len//5:-tot_len//10]
test_info = all_infos_processed[-tot_len//10:]
print(len(train_info), len(dev_info), len(test_info))

with open(os.path.join(root, "train_info.tsv"), "w") as wf:
    for line in train_info:
        print(line.strip(), file=wf)

with open(os.path.join(root, "test_info.tsv"), "w") as wf:
    for line in test_info:
        print(line.strip(), file=wf)

with open(os.path.join(root, "dev_info.tsv"), "w") as wf:
    for line in dev_info:
        print(line.strip(), file=wf)