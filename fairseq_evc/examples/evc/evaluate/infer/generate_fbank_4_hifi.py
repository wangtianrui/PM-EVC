from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.utils import import_user_module_via_dir
import numpy as np
import librosa as lib
import torch, os
import argparse
import pyworld as pw # pip install pyworld
from tqdm import tqdm

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
    dur = len(origin_inp) / 16000
    # origin_inp = norm2wav(origin_inp)
    if norm_wav:
        src_audio = norm(origin_inp)
    else:
        src_audio = origin_inp
    src_len = src_audio.shape[0]
    src_pitch = extract_pitch(origin_inp).unsqueeze(0)
    src_audio = torch.Tensor([src_audio])
    return src_audio, src_pitch, torch.zeros(1, src_len//320), torch.IntTensor([src_len//320]), dur

def change_path2npy(path, ori_home=None, tgt_home=None):
    if ori_home is not None and tgt_home is not None:
        for ori_item in ori_home:
            if path.find(ori_item) != -1:
                path = path.replace(ori_item, tgt_home)
                break
    return path.replace(".wav", ".npy").replace(".flac", ".npy").replace(".ogg", ".npy").replace(".mp3", ".npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--checkpoint_path', type=str)
    parser.add_argument('-t', '--tsv', type=str)
    parser.add_argument('--np', type=int)
    parser.add_argument('--p', type=int)
    parser.add_argument('-o', '--out_home', type=str)
    args = parser.parse_args()
    import_user_module_via_dir(r"examples/evc")
    model, cfg, task = load_model_ensemble_and_task([args.checkpoint_path])
    model = model[0]
    model.eval()
    model = model.cuda().float()
    
    print(f"check norm_wav is ok? {task.cfg.normalize}")
    
    audio_home = "/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/resampled16k;/CDShare2/2023/wangtianrui/dataset/emo;/CDShare3/2023/wangtianrui".split(";")
    
    with open(args.tsv, "r") as rf:
        lines = rf.readlines()
        if args.p == args.np - 1:
            start = args.p*len(lines)//args.np
            end = len(lines)
        else:
            start = args.p*len(lines)//args.np
            end = (args.p+1)*len(lines)//args.np
        print(start, end)
        lines = lines[start:end]
    with open(os.path.join(args.out_home, os.path.basename(args.tsv)+f"_{args.p}_{args.np}"), "w") as wf:
        for idx, line in enumerate(tqdm(lines)):
            temp = line.strip().split("\t")
            if len(temp) != 8:
                (path, sr, length, spk, emo, level), o_trans, emo2 = temp[:6], " ".join(temp[7:-1]), temp[-1]
            else:
                path, sr, length, spk, emo, level, o_trans, emo2 = temp
            
            save_path = change_path2npy(path, audio_home, args.out_home)
            if os.path.exists(save_path):
                origin_inp = lib.load(path, sr=16000)[0]
                dur = len(origin_inp) / 16000
                print(f"{path}\t{dur}\t{spk}\t{emo2}\t{o_trans}\t{save_path}", file=wf)
                continue
            
            src_audio, src_pitch, src_pm, src_len, dur = load_audio(path, task.cfg.normalize)
            if dur > 60:
                continue
            
            try:
                with torch.no_grad():
                    # [content_feature, spk_pooled, emo_pooled, conv_out]
                    src_embs = model.encode(src_audio.cuda(), src_pitch.cuda(), src_pm.cuda(), src_len)
                    result = model.decode([src_embs[0], src_embs[1], src_embs[2], src_embs[3]], src_len)
                        
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                np.save(save_path, result.detach().cpu().numpy()[0])
                print(f"{path}\t{dur}\t{spk}\t{emo2}\t{o_trans}\t{save_path}", file=wf)
            except Exception as e:
                print(path, e)