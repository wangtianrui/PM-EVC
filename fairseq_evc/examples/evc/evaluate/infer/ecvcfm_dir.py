from fairseq.checkpoint_utils import load_model_ensemble, load_model_ensemble_and_task
from fairseq.utils import import_user_module_via_dir
import numpy as np
import librosa as lib
import torch, os
from scipy.io.wavfile import write as write_wav
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
import pyworld as pw # pip install pyworld
from tqdm import tqdm
from scipy.io.wavfile import write

def extract_pitch(ori_audio):
    if len(ori_audio.shape) == 2:
        ori_audio = ori_audio[:, 0]
    # print(save_path, ori_audio.shape)
    frame_period = 320/16000*1000
    f0, timeaxis = pw.dio(ori_audio.astype('float64'), 16000, frame_period=frame_period)
    nonzeros_indices = np.nonzero(f0)
    if len(nonzeros_indices[0]) == 0:
        return torch.Tensor(f0)
    pitch = f0.copy()
    pitch[nonzeros_indices] = np.log(f0[nonzeros_indices])
    mean, std = np.mean(pitch[nonzeros_indices]), np.std(pitch[nonzeros_indices])
    pitch[nonzeros_indices] = (pitch[nonzeros_indices] - mean) / (std + 1e-8)
    return torch.Tensor(pitch)

def norm(wav):
    mu = np.mean(wav)
    sigma = np.std(wav, axis=0) + 1e-8
    wav = (wav - mu) / sigma
    return wav

def norm2wav(audio):
    return audio / np.max(abs(audio)) * 0.95

def load_audio(audio_path, norm_wav):
    origin_inp = lib.load(audio_path, sr=16000)[0]
    # origin_inp = norm2wav(origin_inp)
    if norm_wav:
        src_audio = norm(origin_inp)
    else:
        src_audio = origin_inp
    src_len = src_audio.shape[0]
    src_pitch = extract_pitch(origin_inp).unsqueeze(0)
    src_audio = torch.Tensor([src_audio])
    return src_audio, src_pitch, torch.zeros(1, src_len//320), torch.IntTensor([src_len//320])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--checkpoint_path', type=str)
    parser.add_argument('-t', '--tsv', type=str)
    parser.add_argument('-o', '--out_home', type=str)
    parser.add_argument('-ts', '--task', type=str)
    parser.add_argument('-nt', '--n_timesteps', type=int)
    parser.add_argument('-temp', '--temperature', type=float)
    args = parser.parse_args()
    
    import_user_module_via_dir(r"examples/reconstruct_dhubert")
    model, cfg, task = load_model_ensemble_and_task([args.checkpoint_path])
    model = model[0]
    model.eval()
    model = model.cuda().float()
    
    print(f"check norm_wav is ok? {task.cfg.normalize}")
    
    with open(args.tsv, "r") as rf:
        lines = rf.readlines()
    
    for idx, line in enumerate(tqdm(lines)):
        path, trans, dur, emo, trans_len, spk, \
            tgt_path, tgt_trans, tgt_dur, tgt_emo, tgt_translen, tgt_spk = line.strip().split("\t")
        
        save_name = str(idx) + '_' + os.path.basename(path).split(".")[0] + "_srcspk%s"%spk + "_tgtspk%s"%tgt_spk + "_srcemo%s"%emo + "_tgtemo%s"%tgt_emo + ".wav"
        
        if args.task == "vc":
            save_path = os.path.join(args.out_home, "npy", "vc", save_name)
        if args.task == "ec":
            save_path = os.path.join(args.out_home, "npy", "ec", save_name)
        if args.task == "ecvc":
            save_path = os.path.join(args.out_home, "npy", "ecvc", save_name)
        if args.task == "rec":
            save_path = os.path.join(args.out_home, "npy", "rec", save_name)
        
        if os.path.exists(save_path+".npy"):
            continue
        
        src_audio, src_pitch, src_pm, src_len = load_audio(path, task.cfg.normalize)
        tgt_audio, tgt_pitch, tgt_pm, tgt_len = load_audio(tgt_path, task.cfg.normalize)
        
        with torch.no_grad():
            # [content_feature, spk_pooled, emo_pooled, conv_out]
            src_embs = model.encode(src_audio.cuda(), src_pitch.cuda(), src_pm.cuda(), src_len)
            tgt_embs = model.encode(tgt_audio.cuda(), tgt_pitch.cuda(), tgt_pm.cuda(), tgt_len)
            
            if args.task == "vc":
                result = model.decode(
                    [src_embs[0], tgt_embs[1], src_embs[2], src_embs[3], src_embs[4]], 
                    n_timesteps=args.n_timesteps, temperature=args.temperature, feature_len=src_len
                )
                save_path = os.path.join(args.out_home, "npy", "vc", save_name)
            if args.task == "ec":
                result = model.decode(
                    [src_embs[0], src_embs[1], tgt_embs[2], src_embs[3], src_embs[4]], 
                    n_timesteps=args.n_timesteps, temperature=args.temperature, feature_len=src_len
                )
                save_path = os.path.join(args.out_home, "npy", "ec", save_name)
            if args.task == "ecvc":
                result = model.decode(
                    [src_embs[0], tgt_embs[1], tgt_embs[2], src_embs[3], src_embs[4]], 
                    n_timesteps=args.n_timesteps, temperature=args.temperature, feature_len=src_len
                )
                save_path = os.path.join(args.out_home, "npy", "ecvc", save_name)
            if args.task == "rec":
                result = model.decode(
                    [src_embs[0], src_embs[1], src_embs[2], src_embs[3], src_embs[4]], 
                    n_timesteps=args.n_timesteps, temperature=args.temperature, feature_len=src_len
                )
                save_path = os.path.join(args.out_home, "npy", "rec", save_name)
                
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.save(save_path, result.detach().cpu().numpy()[0])
        