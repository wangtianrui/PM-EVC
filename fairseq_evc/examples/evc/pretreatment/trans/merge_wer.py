from tqdm import tqdm
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument('--data-home', type=str)
    args = parser.parse_args()
    sp = spm.SentencePieceProcessor()
    sp.load(args.data_home+'/english_emo_data/sp_model.model')

    idx2trans = {}
    with open(args.data_home+"/english_emo_data/wer_detail.log", "r") as rf:
        lines = rf.readlines()
        last_id = -1
        for line in tqdm(lines):
            if len(line.strip()) < 1:
                continue
            if line.startswith("utt"):
                path = int(line.strip().split(" ")[-1])
            if line.startswith("WER"):
                wer = float(line.strip().split(": ")[-1].split(" %")[0])
            if line.startswith("lab"):
                lab = line.strip().replace("lab: ", "")
            if line.startswith("rec"):
                rec = line.strip().replace("rec: ", "")
                if path != last_path:
                    idx2trans[path] = {
                        "lab": " ".join(sp.encode_as_pieces(normalize_text(lab))), 
                        "rec": " ".join(sp.encode_as_pieces(normalize_text(rec))), 
                        "wer":wer
                    }
                    last_path = path

    with open(args.data_home+'/english_emo_data/whisper_wer.json', 'w') as json_file:
        json.dump(idx2trans, json_file, indent=5, ensure_ascii=False)
            