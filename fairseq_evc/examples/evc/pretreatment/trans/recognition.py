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
    parser.add_argument('--num_p', type=int, help="Input file")
    parser.add_argument('--cur_p', type=int, help="Input file")
    args = parser.parse_args()
    
    model = whisper.load_model("large-v3", download_root=r"/Work21/2023/cuizhongjian/python/adapter/whisper/model/")
    norm = EnglishTextNormalizer()
    
    info = r"/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/all_info.tsv"
    with torch.no_grad():
        with open("/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/trans_ori_%d.tsv"%args.cur_p, "w") as o_wf:
            with open("/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/trans_whisper_%d.tsv"%args.cur_p, "w") as t_wf:
                with open(info, "r") as rf:
                    all_infos = rf.readlines()
                    slice_num = len(all_infos) // args.num_p
                    if args.cur_p != args.num_p - 1:
                        end = (args.cur_p + 1) * slice_num
                    else:
                        end = len(all_infos)
                    start = args.cur_p * slice_num
                    
                    print(start, end, len(all_infos))
                    for line in tqdm(all_infos[start:end]):
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
                            
                        trans = recognition(model, path)
                        print("%s %s"%(path, norm(o_trans)), file=o_wf)
                        print("%s %s"%(path, norm(trans)), file=t_wf)
                        