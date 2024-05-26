import pandas as pd
from Bio import SeqIO
import random

from typing import List, Tuple

FOLD_NO = "1"


def load_fasta_as_lists(file_path: str) -> Tuple[List[str], List[str]]:
    """Read a FASTA file and return a tuple of ids and sequences."""
    ids = []
    sequences = []
    with open(file_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            ids.append(str(record.id))
            sequences.append((str(record.seq)))
    return ids, sequences


def _read_fasta(file_path: str, label: int):
    """Read a FASTA file and return a list of (sequence, label) tuples."""
    sequences = []
    with open(file_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append((str(record.seq), label))
    return sequences


def _combine_and_shuffle(sequences1: List[Tuple[str, int]], sequences2: List[Tuple[str, int]]):
    """Combine two lists of sequences, shuffle them, and return a DataFrame."""
    combined = sequences1 + sequences2
    random.shuffle(combined)
    return pd.DataFrame(combined, columns=['seq', 'label'])


def create_train_val_csvs(fold_no: str = FOLD_NO):
    # Paths to the FASTA files
    path_negative_train = "data/negative/PB40/PB40_1z20_clu50_trn" + FOLD_NO + ".fa"
    path_positive_train = "data/positive/bass_motif/pad/bass_ctm_motif_trn" + FOLD_NO + ".fa"
    path_negative_val = "data/negative/PB40/PB40_1z20_clu50_val" + FOLD_NO + ".fa"
    path_positive_val = "data/positive/bass_motif/pad/bass_ctm_motif_val" + FOLD_NO + ".fa"

    # Read sequences and assign labels
    negative_sequences_train = _read_fasta(path_negative_train, 0)
    positive_sequences_train = _read_fasta(path_positive_train, 1)
    negative_sequences_val = _read_fasta(path_negative_val, 0)
    positive_sequences_val = _read_fasta(path_positive_val, 1)

    # Combine and shuffle datasets
    shuffled_data_train = _combine_and_shuffle(negative_sequences_train, positive_sequences_train)
    shuffled_data_val = _combine_and_shuffle(negative_sequences_val, positive_sequences_val)

    # Save to CSV
    shuffled_data_train.to_csv("data/training/" + fold_no + "/bass_pb40.train.csv", index=False)
    shuffled_data_val.to_csv("data/training/" + fold_no + "/bass_pb40.val.csv", index=False)
