import glob
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (SEQ_CUTOFF, POS_TST_PATH, TST_SETS_FILEPATHS, MODEL_NAME, ADDED_TOKENS_PER_SEQ, MODEL_PATH,
                    PREDS_PATH, makedir)
from utils.datasets import load_fasta_as_lists
from utils.tokenizer import tokenize_seqs

tf.get_logger().setLevel('ERROR')


def _fragment_sequences(sequences: List[str], max_seq_len: int) -> Tuple[List[str], List[int]]:
    frags = []
    scopes = []

    for seq in sequences:
        seq_len = len(seq)

        if seq_len > max_seq_len:
            frags_number = seq_len - max_seq_len + 1

            for i in range(frags_number):
                frags.append(seq[i:i + max_seq_len])

            scopes.append(frags_number)
        else:
            frags.append(seq)
            scopes.append(1)

    return frags, scopes


class FragmentedSet:

    def __init__(self, fasta_filepath: str, max_seq_len: int) -> None:
        filepath_comps = fasta_filepath.split(os.sep)
        self.set_name = filepath_comps[-1].split(".")[0]
        self.is_positive = filepath_comps[2] == POS_TST_PATH.split(os.sep)[2]

        self.ids, seqs = load_fasta_as_lists(fasta_filepath)
        self.frags, self.scopes = _fragment_sequences(seqs, max_seq_len)


def predict(model_dir: str) -> None:
    cv_models_filepaths = glob.glob(os.path.join(model_dir, "*"))

    comb_model_name = MODEL_NAME + "comb" + "".join(str(i) for i in range(1, len(cv_models_filepaths) + 1))

    for set_filepath in TST_SETS_FILEPATHS:
        print(set_filepath)
        fs = FragmentedSet(set_filepath, SEQ_CUTOFF)

        # Tokenize text
        x_tst = tokenize_seqs(fs.frags, SEQ_CUTOFF + ADDED_TOKENS_PER_SEQ)

        y_pred = []
        for i, model_filepath in enumerate(cv_models_filepaths):
            model = tf.keras.models.load_model(model_filepath)

            # Predict
            preds = model.predict(
                [x_tst, np.zeros((len(fs.frags), 8943), dtype=np.int8)], verbose=0)

            y_pred.append(preds.flatten())

            # Save cv results
            model_name = os.path.basename(model_filepath)
            save_model_prediction(model_name, fs, y_pred[i])

        # Save comb results
        if i > 0:
            print("comb saved as", comb_model_name)
            y_pred = np.mean(y_pred, axis=0)
        save_model_prediction(comb_model_name, fs, y_pred)


def save_model_prediction(model_name: str, fs: FragmentedSet, fragments_prediction) -> None:
    pred, frags = to_sequence_prediction(fs, fragments_prediction)
    save_prediction(
        os.path.join(PREDS_PATH, f"{fs.set_name}.{model_name}.csv"),
        fs, pred, frags
    )


def to_sequence_prediction(fs: FragmentedSet, fragments_prediction) \
        -> Tuple[List[float], List[str]]:
    pred = []
    frags = []

    p = 0
    for ss in fs.scopes:
        scoped_frags_pred = fragments_prediction[p:p + ss]
        max_pred_index = np.argmax(scoped_frags_pred)
        pred.append(scoped_frags_pred[max_pred_index])
        frags.append(fs.frags[p + max_pred_index])
        p += ss

    return pred, frags


def save_prediction(filepath: str, fs: FragmentedSet, prediction: List[float], fragments: List[str]) -> None:
    df = pd.DataFrame({
        "id": fs.ids,
        "prob": prediction,
        "class": np.full(len(fs.ids), int(fs.is_positive)),
        "frag": fragments
    })
    df.to_csv(makedir(filepath), sep="\t", index=False)


if __name__ == "__main__":
    predict(MODEL_PATH)
