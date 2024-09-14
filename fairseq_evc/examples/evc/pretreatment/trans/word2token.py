from pathlib import Path
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Union
import sentencepiece as sp
import os
from whisper.normalizers import EnglishTextNormalizer
import re
from pathlib import Path

UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1

def normalize_text(text):
    # 使用正则表达式替换掉不规则的 Unicode 字符
    normalized_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xa0]', ' ', text)
    normalized_text = normalized_text.replace('，', ',').replace('。', '.').replace('；', ';').replace('：', ':').replace('？', '?').replace('！', '!').replace('（', '(').replace('）', ')').replace('【', '[').replace('】', ']').replace('“', "'").replace('”', '"').replace('‘', "'").replace('’', "'")
    normalized_text = re.sub(r"[^\w\s']", "", normalized_text)
    return normalized_text

def gen_vocab(
    input_path: Path, output_path_prefix: Path, model_type="bpe",
    vocab_size=1000, special_symbols: Optional[List[str]] = None
):
    # Train SentencePiece Model
    arguments = [
        f"--input={input_path.as_posix()}",
        f"--model_prefix={output_path_prefix.as_posix()}",
        f"--model_type={model_type}",
        f"--vocab_size={vocab_size}",
        "--character_coverage=1.0",
        f"--num_threads={cpu_count()}",
        f"--unk_id={UNK_TOKEN_ID}",
        f"--bos_id={BOS_TOKEN_ID}",
        f"--eos_id={EOS_TOKEN_ID}",
        f"--pad_id={PAD_TOKEN_ID}",
    ]
    if special_symbols is not None:
        _special_symbols = ",".join(special_symbols)
        arguments.append(f"--user_defined_symbols={_special_symbols}")
    sp.SentencePieceTrainer.Train(" ".join(arguments))
    # Export fairseq dictionary
    spm = sp.SentencePieceProcessor()
    spm.Load(output_path_prefix.as_posix() + ".model")
    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}
    assert (
        vocab.get(UNK_TOKEN_ID) == UNK_TOKEN
        and vocab.get(PAD_TOKEN_ID) == PAD_TOKEN
        and vocab.get(BOS_TOKEN_ID) == BOS_TOKEN
        and vocab.get(EOS_TOKEN_ID) == EOS_TOKEN
    )
    vocab = {
        i: s
        for i, s in vocab.items()
        if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    }
    with open(output_path_prefix.as_posix() + ".txt", "w") as f_out:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            f_out.write(f"{s} 1\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument('--data-home', type=str)
    parser.add_argument('--vocab-size', type=int)
    args = parser.parse_args()
    root = args.data_home + "/english_emo_data"
    with open(os.path.join(root, "all_info.tsv"), "r") as rf:
        with open(os.path.join(root, "all_trans.tsv"), "w") as wf:
            for line in rf.readlines():
                temp = line.split("\t")
                if len(temp) != 8:
                    (path, sr, length, spk, emo, level), o_trans, emo2 = temp[:6], " ".join(temp[7:-1]), temp[-1]
                else:
                    path, sr, length, spk, emo, level, o_trans, emo2 = temp
                print(normalize_text(o_trans).upper(), file=wf)
    gen_vocab(
        Path(os.path.join(root, "all_trans.tsv")),
        Path(os.path.join(root, "sp_model")),
        model_type="unigram",
        vocab_size=args.vocab_size,
    )