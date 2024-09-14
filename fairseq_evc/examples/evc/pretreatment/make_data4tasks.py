import librosa as lib
import numpy as np
import sentencepiece as spm
from copy import copy
import json
import os

def add_dict(key, v, dict_temp):
    if key in dict_temp.keys():
        dict_temp[key].append(v)
    else:
        dict_temp[key] = [v]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument('--data-home', type=str)
    parser.add_argument('--cluster-speaker-num', type=int)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.data_home+'/english_emo_data/sp_model.model')
    with open(args.data_home+'/english_emo_data/spk_emb/spkdict_{args.cluster_speaker_num}.json', 'r', encoding='utf-8') as file:
        spk_dick = json.load(file)
        spk_dick = {v: k for k, v in spk_dick.items()}
    spk_dict = {}
    emo_dict = {}
    spk_emo_dict = {}
    n_dict = {}
    trans = []
    with open(f"{args.data_home}/english_emo_data/wer.json", 'r') as file:
        trans_wer = json.load(file)
    with open(args.data_home+"/english_emo_data/test_info.tsv", "r") as rf:
        for idx, line in enumerate(rf.readlines()):
            temp = line.split("\t")
            path, sr, length, spk, emo, level, o_trans, emo_id = temp
            wer = trans_wer[idx]
            if wer > 2:
                continue
            if path.find("wangtianrui/IEMOCAP") != -1: # containing overlapping
                continue
            spk = spk_dick[int(spk)]
            if len(o_trans.split(" ")) > 8 and len(o_trans.split(" ")) < 40: # filter out too short and too long speech
                trans.append(sp.decode(o_trans.split(" ")))
            else:
                continue
                
            add_dict(spk, (path, sp.decode(o_trans.split(" ")), int(length)/float(sr), emo, len(o_trans.split(" ")), spk), spk_dict)
            add_dict(emo, (path, sp.decode(o_trans.split(" ")), int(length)/float(sr), emo, len(o_trans.split(" ")), spk), emo_dict)
            add_dict(spk+"||"+emo, (path, sp.decode(o_trans.split(" ")), int(length)/float(sr), emo, len(o_trans.split(" ")), spk), spk_emo_dict)
            if emo.find("neutr") != -1:
                add_dict(spk, (path, sp.decode(o_trans.split(" ")), int(length)/float(sr), emo, len(o_trans.split(" ")), spk), n_dict)
                
    for key in emo_dict.keys():
        print(key, len(emo_dict[key]))

    for item in ["ec" "vc" "evc" "rec"]:
        try:
            os.makedirs(os.path.join(args.data_home, "/english_emo_data/evals", item), exist_ok=True)
        except:
            pass
    
    with open(args.data_home+"/english_emo_data/evals/vc/test_info.tsv", "w") as wf:
        num = 0
        for key in n_dict.keys():
            for path, trans, dur, emo, trans_len, spk in n_dict[key]:
                if num == 1200:
                    break
                spks = copy(list(n_dict.keys()))
                spks.remove(key)
                np.random.shuffle(spks)
                random_spk = spks[0]
                
                random_idx = np.random.randint(low=0, high=len(n_dict[random_spk]))
                print(random_idx, len(n_dict[random_spk]))
                tgt_path, tgt_trans, tgt_dur, tgt_emo, tgt_translen, tgt_spk = n_dict[random_spk][random_idx]
                print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s"%(path, trans, str(dur), emo, str(trans_len), spk, tgt_path, tgt_trans, str(tgt_dur), tgt_emo, str(tgt_translen), tgt_spk), file=wf)
                num+=1
                
                
    with open(args.data_home+"/english_emo_data/evals/evc/test_info.tsv", "w") as wf:
        num = 0
        for key in emo_dict.keys():
            for path, trans, dur, emo, trans_len, spk in emo_dict[key]:
                if num == 1200:
                    break
                spks = copy(list(emo_dict.keys()))
                spks.remove(key)
                np.random.shuffle(spks)
                random_spk = spks[0]
                
                random_idx = np.random.randint(low=0, high=len(emo_dict[random_spk]))
                print(random_idx, len(emo_dict[random_spk]))
                tgt_path, tgt_trans, tgt_dur, tgt_emo, tgt_translen, tgt_spk = emo_dict[random_spk][random_idx]
                print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s"%(path, trans, str(dur), emo, str(trans_len), spk, tgt_path, tgt_trans, str(tgt_dur), tgt_emo, str(tgt_translen), tgt_spk), file=wf)
                num+=1

    with open(args.data_home+"/english_emo_data/evals/rec/test_info.tsv", "w") as wf:
        num = 0
        for key in emo_dict.keys():
            for path, trans, dur, emo, trans_len, spk in emo_dict[key]:
                if num == 1200:
                    break
                spks = copy(list(emo_dict.keys()))
                spks.remove(key)
                np.random.shuffle(spks)
                random_spk = spks[0]
                
                random_idx = np.random.randint(low=0, high=len(emo_dict[random_spk]))
                print(random_idx, len(emo_dict[random_spk]))
                tgt_path, tgt_trans, tgt_dur, tgt_emo, tgt_translen, tgt_spk = emo_dict[random_spk][random_idx]
                print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s"%(path, trans, str(dur), emo, str(trans_len), spk, path, trans, str(dur), emo, str(trans_len), spk), file=wf)
                num+=1

    num_dict = {}
    with open(args.data_home+"/english_emo_data/evals/ec/test_info.tsv", "w") as wf:
        num = 0
        for key in spk_emo_dict.keys():
            for path, trans, dur, emo, trans_len, spk in spk_emo_dict[key]:
                if num == 1200:
                    break
                if trans_len < 2 or dur < 2:
                    continue
                if emo in num_dict.keys():
                    if len(num_dict[emo]) >= 150:
                        continue
                spks = copy(list(spk_emo_dict.keys()))
                spks.remove(key)
                cur_spk, cur_emo = key.split("||")
                np.random.shuffle(spks)
                find = False
                for spk_emo in spks:
                    spk_temp, emo_temp = spk_emo.split("||")
                    if spk_temp == cur_spk and emo_temp != cur_emo:
                        random_spk = spk_emo
                        find = True
                if not find:
                    continue
                random_idx = np.random.randint(low=0, high=len(spk_emo_dict[random_spk]))
                print(random_idx, len(spk_emo_dict[random_spk]))
                add_dict(emo, (path, trans, dur, emo, trans_len, spk), num_dict)
                tgt_path, tgt_trans, tgt_dur, tgt_emo, tgt_translen, tgt_spk = spk_emo_dict[random_spk][random_idx]
                print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s"%(path, trans, str(dur), emo, str(trans_len), spk, tgt_path, tgt_trans, str(tgt_dur), tgt_emo, str(tgt_translen), tgt_spk), file=wf)
                num+=1