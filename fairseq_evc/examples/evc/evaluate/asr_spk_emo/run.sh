cd /Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu 
tasks=("fbank" "hubertbase" "wavlmbase" "whispersmall" "hubertlarge" "progrebase" "wavlmlarge" "whispermedium" "wav2vec2base" "wav2vec2large" "data2vecbase" "data2veclarge")
for task in "${tasks[@]}"; do
OUT_HOME=/Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu/examples/evc/logs/ecvc/prog_resnet/$task

PYTHONPATH=/Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu \
CUDA_VISIBLE_DEVICES=3 \
/Work21/2023/wangtianrui/miniconda3/envs/tts/bin/python -u /Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu/examples/evc/evaluate/asr_spk_emo/asr_spk_emo_eval.py \
--checkpoint_path ${OUT_HOME}/1_stage_best.pt \
--out_home ${OUT_HOME} 2>&1 | tee ${OUT_HOME}/eval_asr_spk_emo_best.out &

PYTHONPATH=/Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu \
CUDA_VISIBLE_DEVICES=3 \
/Work21/2023/wangtianrui/miniconda3/envs/tts/bin/python -u /Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu/examples/evc/evaluate/asr_spk_emo/asr_spk_emo_eval.py \
--checkpoint_path ${OUT_HOME}/1_stage_last.pt \
--out_home ${OUT_HOME} 2>&1 | tee ${OUT_HOME}/eval_asr_spk_emo_last.out

wait

python /Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu/examples/evc/evaluate/compute_wer.py --char=1 --v=1 \
$OUT_HOME/ref_best.tsv \
$OUT_HOME/est_best.tsv \
> $OUT_HOME/wer_detail_best.log

python /Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu/examples/evc/evaluate/compute_wer.py --char=1 --v=1 \
$OUT_HOME/ref_last.tsv \
$OUT_HOME/est_last.tsv \
> $OUT_HOME/wer_detail_last.log
done

# OUT_HOME=/Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu/examples/evc/logs/ecvc/prog_resnet/debug
# mkdir -p $OUT_HOME
# PYTHONPATH=/Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu \
# CUDA_VISIBLE_DEVICES=0 \
# /Work21/2023/wangtianrui/miniconda3/envs/tts/bin/python -u /Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu/examples/evc/evaluate/asr_spk_emo/asr_spk_emo_eval.py \
# --checkpoint_path /Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu/examples/evc/logs/ecvc/prog_resnet/progrelarge/checkpoint_last.pt \
# --out_home ${OUT_HOME} 2>&1 | tee ${OUT_HOME}/eval_asr_spk_emo.out

# python /Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu/examples/evc/evaluate/compute_wer.py --char=1 --v=1 \
# $OUT_HOME/ref.tsv \
# $OUT_HOME/est.tsv \
# > $OUT_HOME/wer_detail.log