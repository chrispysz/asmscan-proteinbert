import numpy as np
import pandas as pd
from config import ALL_AMINO_ACIDS, SPECIAL_TOKENS, SEQUENCE_LENGTH

num_amino_acids = len(ALL_AMINO_ACIDS)
amino_acid_to_index = {aa: i for i, aa in enumerate(ALL_AMINO_ACIDS)}
special_token_to_index = {token: i + num_amino_acids for i, token in enumerate(SPECIAL_TOKENS)}
token_to_index = {**amino_acid_to_index, **special_token_to_index}

def decode_sequence(seq):
    """Decodes sequence if it is in bytes."""
    if isinstance(seq, str):
        return seq
    elif isinstance(seq, bytes):
        return seq.decode('utf8')
    else:
        raise TypeError(f'Unexpected sequence type: {type(seq)}')

def tokenize_sequence(seq):
    """Converts sequence of amino acids into a list of token indices."""
    other_token_index = special_token_to_index['<OTHER>']
    return [special_token_to_index['<START>']] + [amino_acid_to_index.get(aa, other_token_index) for aa in decode_sequence(seq)] + [special_token_to_index['<END>']]

def tokenize_sequences(seqs, seq_len):
    """Converts list of sequences into array of token indices."""
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [special_token_to_index['<PAD>']] for seq_tokens in map(tokenize_sequence, seqs)], dtype=np.int32)

def parse_fasta(data_path):
    sequences = {}
    with open(data_path, 'r') as file:
        sequence_id = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence_id is not None:
                    sequences[sequence_id] = ''.join(sequence)
                sequence_id = line
                sequence = []
            else:
                sequence.append(line)
        if sequence_id is not None:
            sequences[sequence_id] = ''.join(sequence)
    return sequences