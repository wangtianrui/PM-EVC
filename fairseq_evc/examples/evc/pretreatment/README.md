# Data Preparation

* *datasets* directory
  * The dataset can be requested and downloaded, with both the download link and the data retrieval script located in the `datasets` directory.
  * After processing the 12 datasets, use `mix_data.py` to aggregate them for subsequent processing.

* *spk* directory
  * A small portion of the data lacks speaker identifiers, so we isolate these samples and use pre-trained models from WeSpeaker to extract representations.
  * Clustering is then applied to assign weak labels to this data.

* *trans* directory
  * Some of the text annotations in the data are inaccurate, so Whisper is used to transcribe all data. During training, Whisper's transcriptions will be used as ASR targets when the WER exceeds 25%.
  * The `word2token.py` script tokenizes English text using a unigram model with a vocabulary size of 1000 for ASR training.

* Use `merge_and_split.py` to merge and split the data for training, development, and testing.

* *pre_feature* directory

    Pre-extract mel-spectrogram, which will be used during training.

    Pre-extract pitch for ProgRE training.