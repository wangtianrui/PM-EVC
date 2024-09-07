from fairseq.checkpoint_utils import load_model_ensemble, load_model_ensemble_and_task
from fairseq.utils import import_user_module_via_dir
import numpy as np
import librosa as lib
import torch, os
from scipy.io.wavfile import write as write_wav
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sentencepiece as spm
import argparse
import pyworld as pw # pip install pyworld
from tqdm import tqdm
from scipy.io.wavfile import write
from sklearn import metrics
import fairseq.utils as utils
import soundfile
from operator import itemgetter
import re
import numpy 
from whisper.normalizers import EnglishTextNormalizer



def normalize_text(text):
    # 使用正则表达式替换掉不规则的 Unicode 字符
    normalized_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xa0]', ' ', text)
    normalized_text = normalized_text.replace('，', ',').replace('。', '.').replace('；', ';').replace('：', ':').replace('？', '?').replace('！', '!').replace('（', '(').replace('）', ')').replace('【', '[').replace('】', ']').replace('“', "'").replace('”', '"').replace('‘', "'").replace('’', "'")
    normalized_text = re.sub(r"[^\w\s']", "", normalized_text)
    return normalized_text.upper()

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
	
	fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
	fnr = 1 - tpr
	tunedThreshold = []
	if target_fr:
		for tfr in target_fr:
			idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
			tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	for tfa in target_fa:
		idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
		tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
	eer  = max(fpr[idxE],fnr[idxE])*100
	
	return tunedThreshold, eer, fpr, fnr

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def accuracy(output, target, topk=(1,)):

	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	
	return res

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
    return audio / np.max(abs(audio))

def load_audio(audio_path):
    origin_inp = lib.load(audio_path, sr=16000)[0]
    # origin_inp = norm2wav(origin_inp)
    # write(os.path.join(out_home, os.path.basename(audio_path)), 16000, origin_inp)
    src_audio = norm(origin_inp)
    src_len = src_audio.shape[0]
    src_pitch = extract_pitch(origin_inp).unsqueeze(0)
    src_audio = torch.Tensor([src_audio])
    return src_audio, src_pitch, torch.zeros(1, src_len//320), torch.IntTensor([src_len//320])

def asr_sid_emo(outputs, asr_dict, sp):
    con_4_ctc, spk_4_ce, emo_4_ce, spk_pooled_utt = outputs
    pred_token_ids = con_4_ctc.argmax(dim=-1).unique_consecutive()
    pred_token_ids = pred_token_ids[pred_token_ids != asr_dict.pad()].tolist()
    pred_tokens = asr_dict.string(pred_token_ids)
    est_str = sp.decode(pred_tokens.split(" "))
    emo_dict_uniq = {'neutral': 0, 'disgust': 1, 'angry': 2, 'happy': 3, 'fear': 4, 'sad': 5, 'surprised': 6, 'contempt': 7}
    swapped_dict = {value: key for key, value in emo_dict_uniq.items()}
    emo = swapped_dict[emo_4_ce.argmax(-1).item()]
    return est_str, emo, F.normalize(spk_pooled_utt, p=2, dim=1)
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--checkpoint_path', type=str)
    parser.add_argument('-o', '--out_home', type=str)
    args = parser.parse_args()
    
    import_user_module_via_dir(r"examples/reconstruct_dhubert")
    model, cfg, task = load_model_ensemble_and_task([args.checkpoint_path])
    model = model[0]
    model.eval()
    model = model.cuda().float()
    asr_dict = task.asr_dict
    sp = spm.SentencePieceProcessor()
    sp.load('/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/sp_model_10000.model')
    text_norm = EnglishTextNormalizer()
    
    spk_scores = []
    spk_labels = []
    emo_corr = {'neutral': [], 'disgust': [], 'angry': [], 'happy': [], 'fear': [], 'sad': [], 'surprised': [], 'contempt': []}
    with open(os.path.join(args.out_home, "ref.tsv"), "w") as ref_w:
        with open(os.path.join(args.out_home, "est.tsv"), "w") as est_w:
            with open(r"/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/ori_emo/evals/spk/test_info.tsv", "r") as rf:
                for idx, line in enumerate(tqdm(rf.readlines())):
                    path, trans, dur, emo, trans_len, spk, \
                        tgt_path, tgt_trans, tgt_dur, tgt_emo, tgt_translen, tgt_spk = line.strip().split("\t")
                    src_audio, src_pitch, src_pm, src_len = load_audio(path)
                    tgt_spk_audio, tgt_spk_pitch, tgt_spk_pm, tgt_spk_len = load_audio(tgt_path)
                    with torch.no_grad():
                        src_embs = model.downstream(src_audio.cuda(), src_pitch.cuda(), src_pm.cuda(), src_len)
                        src_est_str, src_emo, src_spk_pooled_utt = asr_sid_emo(src_embs, asr_dict, sp)
                        tgt_spk_embs = model.downstream(tgt_spk_audio.cuda(), tgt_spk_pitch.cuda(), tgt_spk_pm.cuda(), tgt_spk_len)
                        
                        trans = sp.decode(" ".join(sp.encode_as_pieces(normalize_text(text_norm(trans.lower())))).split(" ")) 
                        tgt_trans = sp.decode(" ".join(sp.encode_as_pieces(normalize_text(text_norm(tgt_trans.lower())))).split(" ")) 
                        
                        tgt_est_str, tgt_emo_est, tgt_spk_pooled_utt = asr_sid_emo(tgt_spk_embs, asr_dict, sp)
                        emo_corr[emo].append(int(emo==src_emo))
                        emo_corr[tgt_emo].append(int(tgt_emo==tgt_emo_est))
                        # path2spk[path] = src_spk_pooled_utt
                        # path2spk[tgt_path] = tgt_spk_pooled_utt
                        print("%d-%d-%s %s"%(idx, 0, path, src_est_str), file=est_w)
                        print("%d-%d-%s %s"%(idx, 1, tgt_path, tgt_est_str), file=est_w)
                        print("%d-%d-%s %s"%(idx, 0, path, trans), file=ref_w)
                        print("%d-%d-%s %s"%(idx, 1, tgt_path, tgt_trans), file=ref_w)
                        
                        score = torch.mean(torch.matmul(src_spk_pooled_utt, tgt_spk_pooled_utt.T)) # higher is positive
                        score = score.detach().cpu().numpy()
                        spk_scores.append(score)
                        # print(int(spk==tgt_spk))
                        spk_labels.append(int(spk==tgt_spk))
    
    # Coumpute EER and minDCF
    EER = tuneThresholdfromScore(spk_scores, spk_labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(spk_scores, spk_labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    print("spk:", EER, minDCF)
    for key in emo_corr.keys():
        print(key, np.mean(emo_corr[key]))

    
                    