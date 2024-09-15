# Data Preparation

* *datasets* directory
  * Each dataset can be requested and downloaded, with both the download link and the data retrieval script located in the `datasets` directory.

    ```shell
    # for example cremad.py
    DATA_HOME=/Work20/wangtianrui/codes/PM-ECVC/emo_datasets/
    cd $DATA_HOME
    git clone https://github.com/CheyneyComputerScience/CREMA-D  # each link of dataset is described in ***.py
    python -u /Work20/wangtianrui/codes/PM-ECVC/fairseq_evc/examples/evc/pretreatment/datasets/cremad.py \
      --data-home ${DATA_HOME}
    ```

  * After processing the 12 datasets, use `mix_data.py` to aggregate them for subsequent processing, will get a `all_info.tsv` file at `DATA_HOME`.

    ```shell
    python -u /Work20/wangtianrui/codes/PM-ECVC/fairseq_evc/examples/evc/pretreatment/datasets/mix_data.py \
      --data-home ${DATA_HOME}
    ```

* *spk* directory
  * A small portion of the data lacks speaker identifiers, so we isolate these samples and use pre-trained models from WeSpeaker to extract representations.

    ```shell
    mkdir -p /Work20/wangtianrui/codes/PM-ECVC/model_temp
    cd /Work20/wangtianrui/codes/PM-ECVC/model_temp
    download model from https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md
    unzip voxceleb_resnet293_LM.zip
    python -u /Work20/wangtianrui/codes/PM-ECVC/fairseq_evc/examples/evc/pretreatment/spk/extract_spk_emb.py \
      --data-home ${DATA_HOME} \
      --speaker-model /Work20/wangtianrui/codes/PM-ECVC/model_temp/voxceleb_resnet293_LM
    ```

  * Clustering is then applied to assign weak labels to this data, will get a `all_info_with_cluster_{CLUSTER_NUM}_spk.tsv` file.
      
    ```shell
    CLUSTER_NUM=100
    python -u /Work20/wangtianrui/codes/PM-ECVC/fairseq_evc/examples/evc/pretreatment/spk/cluster.py \
      --data-home ${DATA_HOME} \
      --cluster-speaker-num ${CLUSTER_NUM}
    ```

* *trans* directory
  
  * The `word2token.py` script tokenizes English text using a unigram model with a vocabulary size of `VOCAL_SIZE` for ASR training. This unigram model is saved at `DATA_HOME/english_emo_data/sp_model.model` and vocab dict is saved at `DATA_HOME/english_emo_data/sp_model.txt`

    ```shell
    VOCAL_SIZE=10000
    python -u /Work20/wangtianrui/codes/PM-ECVC/fairseq_evc/examples/evc/pretreatment/trans/word2token.py \
      --data-home ${DATA_HOME} \
      --vocab-size ${VOCAL_SIZE}
    ```

  * Some of the text annotations in the data are inaccurate, so Whisper is used to transcribe all data. During training, Whisper's transcriptions will be used as ASR targets when the WER exceeds 25%.

    ```shell
    python -u /Work20/wangtianrui/codes/PM-ECVC/fairseq_evc/examples/evc/pretreatment/trans/recognition.py \
      --data-home ${DATA_HOME} 

    python /Work20/wangtianrui/codes/PM-ECVC/fairseq_evc/examples/evc/pretreatment/trans/compute_wer.py --char=1 --v=1 \
    ${DATA_HOME}/english_emo_data/trans_ori.tsv \
    ${DATA_HOME}/english_emo_data/trans_whisper.tsv \
    > ${DATA_HOME}/english_emo_data/wer_detail.log

    # will get a whisper_wer.json, can be fed into training task
    python /Work20/wangtianrui/codes/PM-ECVC/fairseq_evc/examples/evc/pretreatment/trans/merge_wer.py \
      --data-home ${DATA_HOME} 
    ```


* Use `merge_and_split.py` to merge and split the data for training, development, and testing (will get `{train_info, test_info, dev_info}.tsv` at `${DATA_HOME}/english_emo_data`).

  ```shell
  python -u /Work20/wangtianrui/codes/PM-ECVC/fairseq_evc/examples/evc/pretreatment/merge_and_split.py \
    --data-home ${DATA_HOME} \
    --cluster-speaker-num ${CLUSTER_NUM}
  ```

  ```shell
  # make dataset for evaluation
  python -u /Work20/wangtianrui/codes/PM-ECVC/fairseq_evc/examples/evc/pretreatment/make_data4tasks.py
    --data-home ${DATA_HOME} \
    --cluster-speaker-num ${CLUSTER_NUM}
  ```

* *pre_feature* directory

    Pre-extract mel-spectrogram, which will be used during training.

    ```shell
    python -u /Work20/wangtianrui/codes/PM-ECVC/fairseq_evc/examples/evc/pretreatment/pre_feature/extract_fbank.py
      --data-home ${DATA_HOME} \
      --cluster-speaker-num ${CLUSTER_NUM}
    ```

    Pre-extract pitch for ProgRE training.

    ```shell
    python -u /Work20/wangtianrui/codes/PM-ECVC/fairseq_evc/examples/evc/pretreatment/pre_feature/pitch.py
      --data-home ${DATA_HOME} \
      --cluster-speaker-num ${CLUSTER_NUM}
    ```