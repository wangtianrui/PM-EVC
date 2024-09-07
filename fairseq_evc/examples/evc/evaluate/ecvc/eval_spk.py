import torch
import argparse
from tqdm import tqdm
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import os
import torch.nn.functional as F
import soundfile as sf
import fire
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL

MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]

def init_model(model_name, checkpoint=r"/Work20/2023/wangtianrui/model_temp/speaker/wavlm_large_finetune.pth"):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model

def load_tsv(path):
    with open(path, "r") as rf:
        lines = rf.readlines()
    return lines

def cos_simi_resemb(model, generated_path, tgt_path):
    with torch.no_grad():
        gen_wav = preprocess_wav(Path(generated_path))
        tgt_wav = preprocess_wav(Path(tgt_path))
        
        gen_embed = model.embed_utterance(gen_wav)
        tgt_embed = model.embed_utterance(tgt_wav)
        
        sim = F.cosine_similarity(torch.tensor([gen_embed]), torch.tensor([tgt_embed])).item()
        return float(sim)

def cos_simi_wavlm_ecapa(model, generated_path, tgt_path):
    with torch.no_grad():
        wav1, sr1 = sf.read(generated_path)
        wav2, sr2 = sf.read(tgt_path)

        wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
        wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1, new_freq=16000)
        resample2 = Resample(orig_freq=sr2, new_freq=16000)
        wav1 = resample1(wav1)
        wav2 = resample2(wav2)

        wav1 = wav1.cuda()
        wav2 = wav2.cuda()

        emb1 = model(wav1)
        emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2).detach().cpu().item()
        return float(sim)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-t', '--tsv', type=str)
    parser.add_argument('-o', '--out_home', type=str)
    args = parser.parse_args()
    if os.path.exists(os.path.join(args.out_home, "spk.log")):
        with open(os.path.join(args.out_home, "spk.log"), "r") as rf:
            if len(rf.readlines()) > 500:
                exit()
    
    model = VoiceEncoder(device=torch.device("cuda"))
    model.eval()
    
    wavlm_ecapa = init_model("wavlm_large", r"/Work20/2023/wangtianrui/model_temp/speaker/wavlm_large_finetune.pth").cuda()
    wavlm_ecapa.eval()
    
    with open(args.tsv, "r") as rf:
        lines = rf.readlines()
    simis_resemb = []
    simis_wavlm = []
    with open(os.path.join(args.out_home, "spk.log"), "w") as f:
        for idx, line in enumerate(tqdm(lines)):
            path, trans, dur, emo, trans_len, spk, \
            tgt_path, tgt_trans, tgt_dur, tgt_emo, tgt_translen, tgt_spk = line.strip().split("\t")
            save_name = str(idx) + '_' + os.path.basename(path).split(".")[0] + "_srcspk%s"%spk + "_tgtspk%s"%tgt_spk + "_srcemo%s"%emo + "_tgtemo%s"%tgt_emo + ".wav"
            
            generated_wav = os.path.join(args.out_home, save_name)
            
            simi_resemb = cos_simi_resemb(model, generated_wav, tgt_path)
            simis_resemb.append(simi_resemb)
            
            simi_wavlm = cos_simi_wavlm_ecapa(wavlm_ecapa, generated_wav, tgt_path)
            simis_wavlm.append(simi_wavlm)
            print("%s %s %f %f"%(generated_wav, tgt_path, simi_resemb, simi_wavlm), file=f)
        print("------------------------------------------", file=f)
        print("resemb:", np.mean(simis_resemb), "wavlm:", np.mean(simis_wavlm), file=f)
        