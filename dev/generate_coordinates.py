import os
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (SEP, PREDS_PATH, MODEL_NAME, ADDED_TOKENS_PER_SEQ, SEQ_CUTOFF, MODEL_PATH, makedir)
from utils.dim_reduction import (calculate_umap, calculate_tsne)
from utils.tokenizer import tokenize_seqs

VIS_COMBS = [
    # ["NLReff_test", "bass_ntm_domain_test", "fass_ntm_domain_test", "fass_ctm_domain_test", "csga2203_nr40"],
    # ["NLReff_test", "bass_ntm_domain_test", "fass_ntm_domain_test", "fass_ctm_domain_test"],
    ["bass_c01_ntm_domain_test", "bass_c02_ntm_domain_test", "bass_c03_ntm_domain_test", "bass_c06_ntm_domain_test",
     "het-s_ntm_domain_test", "sigma_ntm_domain_test", "pp_ntm_domain_test"]
]


def collect_fragments(comb, preds_path, MODEL_NAME, SEP, cutoff=1.0):
    """
    Parameters:
    comb (list): List of dataset names to process.
    preds_path (str): Path to the directory containing prediction CSV files.
    MODEL_NAME (str): Name of the model used for predictions.
    SEP (str): Separator used in the CSV files.
    cutoff (float, optional): Threshold for filtering predictions based on
                              absolute difference between class and probability.
                              Defaults to 1.0.

    Returns:
    tuple: A tuple containing:
        - frags (list): Filtered fragment sequences
        - norm_preds (list): Normalized predictions for filtered fragments
        - ids (list): IDs of filtered fragments
        - classes (list): Classes of filtered fragments
        - set_names (list): Names of datasets for each filtered fragment
    """
    frags = []
    norm_preds = []
    ids = []
    classes = []
    set_names = []

    for set_name in comb:
        pred = pd.read_csv(os.path.join(preds_path, f'{set_name}.{MODEL_NAME}comb123456.csv'), sep=SEP)
        d = abs(pred["class"] - pred["prob"])
        valid_indices = d <= cutoff

        frags.extend(pred.loc[valid_indices, "frag"])
        norm_preds.extend(pred.loc[valid_indices, "prob"])
        ids.extend(pred.loc[valid_indices, "id"])
        classes.extend(pred.loc[valid_indices, "class"])
        set_names.extend([set_name] * valid_indices.sum())

    return frags, norm_preds, ids, classes, set_names


def extract_embeddings(frags, model_path, layer_name="dropout"):
    """
    Parameters:
    frags (numpy.ndarray): Tokenized sequence fragments to extract embeddings for.
    model_path (str): Path to the directory containing model files.
    layer_name (str, optional): Name of the layer to extract embeddings from.
                                Defaults to "dropout".

    Returns:
    dict: A dictionary containing:
        - 'tsne_x': t-SNE x-coordinates
        - 'tsne_y': t-SNE y-coordinates
        - 'umap_x': UMAP x-coordinates
        - 'umap_y': UMAP y-coordinates
    """
    mdim_rep = []
    vis = {}
    print(f'Extracting embeddings from specified layer: {layer_name}')

    for model_filepath in glob(os.path.join(model_path, "*")):
        model = tf.keras.models.load_model(model_filepath)
        layer_out = model.get_layer(layer_name).output
        fun = tf.keras.backend.function(model.input, layer_out)
        mdim_rep.append(fun([frags, np.zeros((len(frags), 8943), dtype=np.int8)]))

    mdim_rep = np.concatenate(mdim_rep, axis=1)
    vis["tsne_x"], vis["tsne_y"] = calculate_tsne(mdim_rep)
    vis["umap_x"], vis["umap_y"] = calculate_umap(mdim_rep)

    return vis


def process_combinations(vis_combs, preds_path, model_name, sep, seq_cutoff, added_tokens_per_seq, model_path):
    """
    Parameters:
    vis_combs (list): List of dataset combinations to process.
    preds_path (str): Path to the directory containing prediction files.
    model_name (str): Name of the model used for predictions.
    sep (str): Separator used in CSV files.
    seq_cutoff (int): Maximum sequence length for tokenization.
    added_tokens_per_seq (int): Number of additional tokens per sequence.
    model_path (str): Path to the directory containing model files.
    """
    for comb in vis_combs:
        frags, norm_preds, ids, classes, set_names = collect_fragments(comb, preds_path, model_name, sep)

        tokenized_frags = tokenize_seqs(frags, seq_cutoff + added_tokens_per_seq)
        umap = extract_embeddings(tokenized_frags, model_path)

        new_preds = pd.DataFrame({"id": ids, "prob": norm_preds, "class": classes, "frag": frags, "dataset": set_names,
                                  "tsne_x": umap["tsne_x"], "tsne_y": umap["tsne_y"], "umap_x": umap["umap_x"],
                                  "umap_y": umap["umap_y"]})

        save_path = os.path.join(preds_path, f'{".".join(comb)}.csv')

        print(f'Writing {save_path}')

        new_preds.to_csv(makedir(save_path), sep=SEP, index=False)


process_combinations(VIS_COMBS, PREDS_PATH, MODEL_NAME, SEP, SEQ_CUTOFF, ADDED_TOKENS_PER_SEQ, MODEL_PATH)
