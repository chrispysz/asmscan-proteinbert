from dev.config import ALL_AAS, ADDITIONAL_TOKENS

import numpy as np


n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i + n_aas for i, token in enumerate(ADDITIONAL_TOKENS)}
token_to_index = {**aa_to_token_index, **additional_token_to_index}
index_to_token = {index: token for token, index in token_to_index.items()}
n_tokens = len(token_to_index)


def _parse_seq(seq):
    """Decodes sequence if it is in bytes"""
    if isinstance(seq, str):
        return seq
    elif isinstance(seq, bytes):
        return seq.decode('utf8')
    else:
        raise TypeError('Unexpected sequence type: %s' % type(seq))


def _tokenize_seq(seq):
    """Converts sequence of amino acids into a list of token indices."""
    other_token_index = additional_token_to_index['<OTHER>']
    return [additional_token_to_index['<START>']] + [aa_to_token_index.get(aa, other_token_index) for aa in
                                                     _parse_seq(seq)] + [additional_token_to_index['<END>']]


def tokenize_seqs(seqs, seq_len):
    """Converts list of sequences into array of token indices."""
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in
                     map(_tokenize_seq, seqs)], dtype=np.int32)
