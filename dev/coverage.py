import glob
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (SEQ_CUTOFF, POS_TST_PATH, COV_TST_SETS_FILEPATHS_PAIRS, MODEL_NAME, ADDED_TOKENS_PER_SEQ,
                    MODEL_PATH,
                    PREDS_PATH, makedir, CUTOFF_VALUE)
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
    # Save modelcomb name to config
    cv_models_filepaths = glob.glob(os.path.join(model_dir, "*"))

    comb_model_name = MODEL_NAME + "comb" + "".join(str(i) for i in range(1, len(cv_models_filepaths) + 1))

    for set_filepath in COV_TST_SETS_FILEPATHS_PAIRS:
        print(set_filepath[0])
        # Fragment protein sequences
        fs = FragmentedSet(set_filepath[0], SEQ_CUTOFF)

        # Tokenize text
        x_tst = tokenize_seqs(fs.frags, SEQ_CUTOFF + ADDED_TOKENS_PER_SEQ)

        y_pred = []
        for i, model_filepath in enumerate(cv_models_filepaths):
            # Load model
            model = tf.keras.models.load_model(model_filepath)

            # Predict
            preds = model.predict(
                [x_tst, np.zeros((len(fs.frags), 8943), dtype=np.int8)], verbose=0)

            y_pred.append(preds.flatten())  # [[1], [1], ..., [1]] -> [1, 1, ..., 1]

            # Save cv results
            model_name = os.path.basename(model_filepath)
            save_model_prediction(model_name, fs, y_pred[i])

        # Save comb results
        if i > 0:
            print("comb saved as", comb_model_name)
            y_pred = np.mean(y_pred, axis=0)
        save_model_prediction(comb_model_name, fs, y_pred)


def save_model_prediction(model_name: str, fs: FragmentedSet, fragments_prediction) -> None:
    indexes, max_indexes, max_indexes_pred = to_sequence_prediction(fs, fragments_prediction)
    save_prediction(
        os.path.join(PREDS_PATH, "coords", f"{fs.set_name}.{model_name}.csv"),
        fs, CUTOFF_VALUE, indexes, max_indexes, max_indexes_pred
    )


def to_sequence_prediction(fs: FragmentedSet, fragments_prediction):
    indexes = []
    max_indexes = []
    max_indexes_pred = []

    p = 0
    for ss in fs.scopes:
        scoped_frags_pred = fragments_prediction[p:p + ss]
        pred_indexes = np.where(scoped_frags_pred >= CUTOFF_VALUE)
        max_indexes.append(np.argmax(scoped_frags_pred))
        max_indexes_pred.append(scoped_frags_pred[max_indexes[-1]])
        indexes.append(pred_indexes[0])
        p += ss

    return indexes, max_indexes, max_indexes_pred


def save_prediction(filepath: str, fs: FragmentedSet, cutoff, indexes, max_indexes, max_indexes_pred) -> None:
    df = pd.DataFrame({
        "id": fs.ids,
        "cutoff": cutoff,
        "indexes": indexes,
        "max_indexes": max_indexes,
        "max_indexes_pred": max_indexes_pred
    })
    df.to_csv(makedir(filepath), sep="\t", index=False)


if __name__ == "__main__":
    predict(MODEL_PATH)
