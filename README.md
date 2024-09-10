<div align="center">
    <h1>
    PM-EVC
    </h1>
    <p>
    This is the official implement of A Controllable Emotion Voice Conversion Framework with Pre-trained Speech Representations <br>
    </p>
    <p>
    </p>
</div>

## Configure the Environment for Codes

## Data Preparation

Follow the steps in `./pretreatment` to create:

 * `{train_info,dev_info,test_info}.tsv` waveform list files with transcription, emotion label, and speaker label.
 * Mel-spectrogram features of all speech data.
 * `WER.json` Whisper's recognition results, WER of each speech, original transcriptions.

## Training

* ### Training 4-stage Decoder with Pre-trained Models

    * Download pre-trained model checkpoint and preset hyperparameters.

        ```shell
        cd fairseq_evc
        wget https://dl.fbaipublicfiles.com/fairseq/data2vec/vox_pretrained.pt -P ./model_temp/data2vec -nc
        CONFIG=ssllarge
        PM_CKPT=./model_temp/data2vec/vox_pretrained.pt
        PM_NAME=data2vec_local
        NUM_LAYER=24 # keep consistent with pre-trained model
        NUM_DIM=1024 # keep consistent with pre-trained model 

        GPU_NUM=2

        DATA=    # a directory contains {train_info,dev_info}.tsv 
        AUDIO_HOME=    # a root path of audio data
        PITCH_HOME=   # a root path of pitch data (it's noly needed for ProgRE)
        FBANK_HOME=   # a root path of mel-spec data
        ```

    * Training 1-st stage (Speech Disentanglement Module)

        ```shell
        cd fairseq_evc
        SAVE_HOME=./examples/evc/logs/
        mkdir -p ${SAVE_HOME}/${PM_NAME}/1_stage

        python fairseq_cli/hydra_train.py \
        --config-dir examples/evc/configs/prog_resnet \
        --config-name ${CONFIG}_1s \
        common.tensorboard_logdir=${SAVE_HOME}/${PM_NAME}/1_stage \
        checkpoint.save_dir=${SAVE_HOME}/${PM_NAME}/1_stage \
        distributed_training.distributed_world_size=${GPU_NUM} \
        task.data=${DATA} \ 
        task.audio_home=${AUDIO_HOME} \ 
        task.pitch_home=${PITCH_HOME} \  
        task.fbank_home=${FBANK_HOME} \  
        model.upstream=${PM_NAME} \   
        model.upstream_ckpt=${PM_CKPT} \
        model.upstream_num_hidden=${NUM_LAYER} \ 
        model.upstream_hidden_dim=${NUM_DIM} \ 
        2>&1 | tee ${SAVE_HOME}/${PM_NAME}/1_stage/log.out
        ```

    
    * Training 2-nd stage (Progressive generator, Compensator, and Flow Predictor)

        ```shell
        cd fairseq_evc
        SAVE_HOME=./examples/evc/logs/
        mkdir -p ${SAVE_HOME}/${PM_NAME}/2_stage_fm
        python fairseq_cli/hydra_train.py \
        --config-dir examples/evc/configs/prog_resnet_fm \
        --config-name ${CONFIG}_2s \
        common.tensorboard_logdir=${SAVE_HOME}/${PM_NAME}/2_stage_fm \
        checkpoint.save_dir=${SAVE_HOME}/${PM_NAME}/2_stage_fm \
        distributed_training.distributed_world_size=${GPU_NUM} \
        task.downstream_checkpoint=${SAVE_HOME}/${PM_NAME}/1_stage/checkpoint_best.pt \
        task.audio_home=${AUDIO_HOME} \ 
        task.pitch_home=${PITCH_HOME} \  
        task.fbank_home=${FBANK_HOME} \  
        model.upstream=${PM_NAME} \   
        model.upstream_ckpt=${PM_CKPT} \
        model.upstream_num_hidden=${NUM_LAYER} \ 
        model.upstream_hidden_dim=${NUM_DIM} \ 
        2>&1 | tee ${SAVE_HOME}/${PM_NAME}/2_stage_fm/log.out
        ```

    * Training 2-nd stage (Progressive generator and Compensator, without Flow Predictor)

        ```shell
        cd fairseq_evc
        SAVE_HOME=./examples/evc/logs/
        mkdir -p ${SAVE_HOME}/${PM_NAME}/2_stage_wofm
        python fairseq_cli/hydra_train.py \
        --config-dir examples/evc/configs/prog_resnet \
        --config-name ${CONFIG}_2s \
        common.tensorboard_logdir=${SAVE_HOME}/${PM_NAME}/2_stage_wofm \
        checkpoint.save_dir=${SAVE_HOME}/${PM_NAME}/2_stage_wofm \
        distributed_training.distributed_world_size=${GPU_NUM} \
        task.downstream_checkpoint=${SAVE_HOME}/${PM_NAME}/1_stage/checkpoint_best.pt \
        task.audio_home=${AUDIO_HOME} \ 
        task.pitch_home=${PITCH_HOME} \  
        task.fbank_home=${FBANK_HOME} \  
        model.upstream=${PM_NAME} \   
        model.upstream_ckpt=${PM_CKPT} \
        model.upstream_num_hidden=${NUM_LAYER} \ 
        model.upstream_hidden_dim=${NUM_DIM} \ 
        2>&1 | tee ${SAVE_HOME}/${PM_NAME}/2_stage_wofm/log.out
        ```
* Representating speech and reconstructing mel-spectrum via our trained model

    * For model with flow predictor

        ```shell
        cd fairseq_evc
        n_ts=10
        temp=0.9
        train_tsv=${DATA}/train_info.tsv
        dev_tsv=${DATA}/dev_info.tsv
        nps=()
        for i in $(seq 0 3); do
            nps+=("$i")
        done
        GPUS=("0" "1")
        gpu_index=0  
        mel_dir_name=${CONFIG}_fm
        MEL_SAVE_HOME=./emo_datasets/mels/${mel_dir_name}/${PM_NAME}
        mkdir -p $MEL_SAVE_HOME

        for p in "${nps[@]}"; do
            CUDA_VISIBLE_DEVICES=${GPUS[$gpu_index]} \
            python -u examples/evc/evaluate/infer/generate_fbank_4_hifi_fm.py \
            --checkpoint_path ${SAVE_HOME}/${PM_NAME}/2_stage_fm/checkpoint_best.pt \
            --out_home ${MEL_SAVE_HOME} \
            --tsv ${train_tsv} \
            --p $p \
            --np ${#nps[@]} \
            --n_timesteps $n_ts \
            --temperature $temp &
            gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
        done
        wait
        file_name=$(basename "$train_tsv")
        cat ${MEL_SAVE_HOME}/${file_name}_*_${#nps[@]} > ${MEL_SAVE_HOME}/${file_name}

        for p in "${nps[@]}"; do
            CUDA_VISIBLE_DEVICES=${GPUS[$gpu_index]} \
            python -u examples/evc/evaluate/infer/generate_fbank_4_hifi_fm.py \
            --checkpoint_path ${SAVE_HOME}/${PM_NAME}/2_stage_fm/checkpoint_best.pt \
            --out_home ${MEL_SAVE_HOME} \
            --tsv ${dev_tsv} \
            --p $p \
            --np ${#nps[@]} \
            --n_timesteps $n_ts \
            --temperature $temp &
            gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
        done
        wait
        file_name=$(basename "$dev_tsv")
        cat ${MEL_SAVE_HOME}/${file_name}_*_${#nps[@]} > ${MEL_SAVE_HOME}/${file_name}
        ```

    * For model without flow predictor
        ```shell
        cd fairseq_evc
        train_tsv=${DATA}/train_info.tsv
        dev_tsv=${DATA}/dev_info.tsv
        nps=()
        for i in $(seq 0 3); do
            nps+=("$i")
        done
        GPUS=("0" "1")
        gpu_index=0  
        mel_dir_name=${CONFIG}_wofm
        MEL_SAVE_HOME=./emo_datasets/mels/${dir_name}/${PM_NAME}
        mkdir -p $MEL_SAVE_HOME

        for p in "${nps[@]}"; do
            CUDA_VISIBLE_DEVICES=${GPUS[$gpu_index]} \
            python -u examples/evc/evaluate/infer/generate_fbank_4_hifi.py \
            --checkpoint_path ${SAVE_HOME}/${PM_NAME}/2_stage_fm/checkpoint_best.pt \
            --out_home ${MEL_SAVE_HOME} \
            --tsv ${train_tsv} \
            --p $p \
            --np ${#nps[@]} &
            gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
        done
        wait
        file_name=$(basename "$train_tsv")
        cat ${MEL_SAVE_HOME}/${file_name}_*_${#nps[@]} > ${MEL_SAVE_HOME}/${file_name}

        for p in "${nps[@]}"; do
            CUDA_VISIBLE_DEVICES=${GPUS[$gpu_index]} \
            python -u examples/evc/evaluate/infer/generate_fbank_4_hifi.py \
            --checkpoint_path ${SAVE_HOME}/${PM_NAME}/2_stage_fm/checkpoint_best.pt \
            --out_home ${MEL_SAVE_HOME} \
            --tsv ${dev_tsv} \
            --p $p \
            --np ${#nps[@]} &
            gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
        done
        wait
        file_name=$(basename "$dev_tsv")
        cat ${MEL_SAVE_HOME}/${file_name}_*_${#nps[@]} > ${MEL_SAVE_HOME}/${file_name}
        ```

* Fine-tune Hifi-GAN with reconstructed mel-spectrum

    * First, download pre-trained generator and discriminator of hifi-gan from https://drive.google.com/drive/folders/11T-Z5Y8ijmvnEV2uv-9SniKXtefJ7fOA?usp=sharing to `model_temp` dir

    * Start Fine-tune

        ```shell
        cd hifi_gan
        HIFI_SAVE=./logs/${mel_dir_name}/${PM_NAME}
        mkdir -p $HIFI_SAVE
        python -u train_ft.py \
        --config config.json \
        --input_training_file ${MEL_SAVE_HOME}/train_info.tsv \
        --input_validation_file ${MEL_SAVE_HOME}/dev_info.tsv \
        --checkpoint_path $HIFI_SAVE \
        --pretrained_g ./model_temp/g_00500000 \
        --pretrained_d ./model_temp/do_00500000 \
        2>&1 | tee $HIFI_SAVE/log.out
        ```
## Inference

```shell
max_hifi_file=$(ls  /Work20/2023/wangtianrui/codes/util_repos/hifi-gan/exps/$CONFIGHOME/${MODEL_NAME} | grep -E '^g_[0-9]{8}$' | sort -V | tail -n 1)
hifi_ckpt=${HIFI_SAVE}/${max_file}

GPU_DEVICES=0
# test
neutral_man=/CDShare2/2023/wangtianrui/dataset/emo/MSP/podcast/Audios/MSP-PODCAST_2560_0309.wav
sleep_woman=/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/resampled16k/EmoV-DB/EmoV-DB_sorted/bea/Sleepy/sleepiness_1-28_0015.wav
angry_man=/CDShare2/2023/wangtianrui/dataset/emo/EmoV-DB/EmoV-DB_sorted/sam/Angry/anger_336-364_0349.wav
sad_woman=/CDShare2/2023/wangtianrui/dataset/emo/Emotional_Speech_Dataset/0019/Sad/train/0019_001372.wav

neutral_woman=/CDShare2/2023/wangtianrui/dataset/emo/MSP/podcast/Audios/MSP-PODCAST_0574_0042.wav
happy_woman=/CDShare2/2023/wangtianrui/dataset/emo/Emotional_Speech_Dataset/0016/Happy/test/0016_000721.wav
sad_man=/CDShare2/2023/wangtianrui/dataset/emo/MSP/podcast/Audios/MSP-PODCAST_2894_0284.wav

tasks=("wav2vec2base")
for task in "${tasks[@]}"; do
CKPT_NAME=${task}
CONFIG_NAME=prog_resnet
log_home=/Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu/examples/reconstruct_dhubert/logs/ecvc/${CONFIG_NAME}/
PYTHONPATH=/Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu \
CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
/Work21/2023/wangtianrui/miniconda3/envs/tts/bin/python -u /Work20/2023/wangtianrui/codes/util_repos/fairseq_zhikangniu/examples/reconstruct_dhubert/test/ecvc_single.py \
--checkpoint_path $log_home/$CKPT_NAME/checkpoint_best.pt \
--source_wav $sad_man \
--target_spk_wav $happy_woman \
--target_emo_wav $angry_man \
--out_home $log_home/$CKPT_NAME


# hifi_ckpt="/Work20/2023/wangtianrui/codes/util_repos/hifi-gan/exps/emo/g_00660000"
max_file=$(ls  /Work20/2023/wangtianrui/codes/util_repos/hifi-gan/exps/$CONFIG_NAME/${CKPT_NAME} | grep -E '^g_[0-9]{8}$' | sort -V | tail -n 1)
hifi_ckpt=/Work20/2023/wangtianrui/codes/util_repos/hifi-gan/exps/$CONFIG_NAME/${CKPT_NAME}/${max_file}

PYTHONPATH=/Work20/2023/wangtianrui/codes/util_repos/hifi-gan \
CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
/Work21/2023/wangtianrui/miniconda3/envs/tts/bin/python -u /Work20/2023/wangtianrui/codes/util_repos/hifi-gan/inference_ecvc.py \
--fbank $log_home/$CKPT_NAME/spk_cv_fbank.npy \
--checkpoint_path $hifi_ckpt

PYTHONPATH=/Work20/2023/wangtianrui/codes/util_repos/hifi-gan \
CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
/Work21/2023/wangtianrui/miniconda3/envs/tts/bin/python -u /Work20/2023/wangtianrui/codes/util_repos/hifi-gan/inference_ecvc.py \
--fbank $log_home/$CKPT_NAME/emo_cv_fbank.npy \
--checkpoint_path $hifi_ckpt

PYTHONPATH=/Work20/2023/wangtianrui/codes/util_repos/hifi-gan \
CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
/Work21/2023/wangtianrui/miniconda3/envs/tts/bin/python -u /Work20/2023/wangtianrui/codes/util_repos/hifi-gan/inference_ecvc.py \
--fbank $log_home/$CKPT_NAME/spk_emo_fbank.npy \
--checkpoint_path $hifi_ckpt

PYTHONPATH=/Work20/2023/wangtianrui/codes/util_repos/hifi-gan \
CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
/Work21/2023/wangtianrui/miniconda3/envs/tts/bin/python -u /Work20/2023/wangtianrui/codes/util_repos/hifi-gan/inference_ecvc.py \
--fbank $log_home/$CKPT_NAME/reco_fbank.npy \
--checkpoint_path $hifi_ckpt
done
```

