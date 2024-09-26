import os
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (SEP, PREDS_PATH, MODEL_NAME, ADDED_TOKENS_PER_SEQ, SEQ_CUTOFF, MODEL_PATH, makedir)
from utils.dim_reduction import (calculate_umap, calculate_tsne)
from utils.tokenizer import tokenize_seqs

VIS_COMBS = [
    #["NLReff_test", "bass_ntm_domain_test", "fass_ntm_domain_test", "fass_ctm_domain_test", "csga2203_nr40"],
   # ["NLReff_test", "bass_ntm_domain_test", "fass_ntm_domain_test", "fass_ctm_domain_test"],
    ["bass_c01_ntm_domain_test", "bass_c02_ntm_domain_test", "bass_c03_ntm_domain_test", "bass_c06_ntm_domain_test", "het-s_ntm_domain_test", "sigma_ntm_domain_test", "pp_ntm_domain_test"]
]

def collect_fragments(comb, preds_path, MODEL_NAME, SEP, cutoff = 1.0):
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
    for comb in vis_combs:
        frags, norm_preds, ids, classes, set_names = collect_fragments(comb, preds_path, model_name, sep)

        tokenized_frags = tokenize_seqs(frags, seq_cutoff + added_tokens_per_seq)
        umap = extract_embeddings(tokenized_frags, model_path)

        new_preds = pd.DataFrame({"id": ids,"prob": norm_preds, "class": classes,"frag": frags, "dataset": set_names,"tsne_x": umap["tsne_x"], "tsne_y": umap["tsne_y"], "umap_x": umap["umap_x"], "umap_y": umap["umap_y"]})

        new_preds.to_csv(makedir(os.path.join(model_path, preds_path, f'{".".join(comb)}.csv')), sep=SEP, index=False)

process_combinations(VIS_COMBS, PREDS_PATH, MODEL_NAME, SEP, SEQ_CUTOFF, ADDED_TOKENS_PER_SEQ, MODEL_PATH)