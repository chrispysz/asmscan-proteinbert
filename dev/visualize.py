import os
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt, cycler

from config import (SEP, PREDS_PATH, MODEL_NAME, ADDED_TOKENS_PER_SEQ, SEQ_CUTOFF, MODEL_PATH, MARKER_SCALE_ALL,
                    MARKER_SCALE_POS_ONLY)
from utils.dim_reduction import (calculate_umap, calculate_tsne)
from utils.tokenizer import tokenize_seqs

TSNE_COMBS = [
    # ["PB40_1z20_clu50_test_sampled10000", "NLReff_test", "bass_ntm_domain_test", "fass_ntm_domain_test",
    #  "fass_ctm_domain_test"],
    # ["PB40_1z20_clu50_test_sampled10000", "NLReff_test", "bass_ntm_motif_test", "fass_ntm_motif_test",
    #  "fass_ctm_motif_test"],
    # ["PB40_1z20_clu50_test_sampled10000", "NLReff_test", "bass_ntm_motif_test", "bass_ntm_motif_env5_test",
    #  "bass_ntm_motif_env10_test"],
    # ["PB40_1z20_clu50_test_sampled10000", "NLReff_test", "fass_ntm_motif_test", "fass_ntm_motif_env5_test",
    #  "fass_ntm_motif_env10_test"],
    # ["PB40_1z20_clu50_test_sampled10000", "NLReff_test", "het-s_ntm_domain_test", "sigma_ntm_domain_test",
    #  "pp_ntm_domain_test"],
    # ["het-s_ntm_domain_test", "sigma_ntm_domain_test", "pp_ntm_domain_test", "bass_other_ctm_domain_test", "bass_other_ntm_domain_test"],
    # ["bass_01_ntm_domain_test", "bass_02_ntm_domain_test", "bass_03_ntm_domain_test", "bass_06_ntm_domain_test", "bass_fam_others_ntm_domain_test"],
    # ["bass_01_ntm_domain_test", "bass_02_ntm_domain_test", "bass_03_ntm_domain_test", "sigma_het-s_pp_ntm_domain_test"],
    # ["NLReff_test", "bass_ntm_domain_test", "fass_ntm_domain_test", "fass_ctm_domain_test"],
    # ["NLReff_test", "bass_ntm_motif_test", "fass_ntm_motif_test", "fass_ctm_motif_test"],
    # ["NLReff_test", "bass_ntm_motif_test", "bass_ntm_motif_env5_test", "bass_ntm_motif_env10_test"],
    # ["NLReff_test", "fass_ntm_motif_test", "fass_ntm_motif_env5_test", "fass_ntm_motif_env10_test"],
    ["NLReff_test", "het-s_ntm_domain_test", "sigma_ntm_domain_test", "pp_ntm_domain_test"],
    #["bass_01_ntm_domain_test", "bass_02_ntm_domain_test", "bass_03_ntm_domain_test", "bass_06_ntm_domain_test", "het-s_ntm_domain_test", "sigma_ntm_domain_test", "pp_ntm_domain_test"]
]


def create_and_save_plot(x, y, title, filename, comb, sets_sizes, colors, marker_sizes):
    plt.figure(figsize=(12.8, 9.6))
    plt.rc("axes", prop_cycle=cycler(color=colors))
    plt.axis("off")

    i = 0
    for set in comb:
        ss = sets_sizes[set]
        if len(marker_sizes) == 1:
            plt.scatter(x[i:i + ss], y[i:i + ss], label=set, s=marker_sizes)
        else:
            plt.scatter(x[i:i + ss], y[i:i + ss], label=set, s=marker_sizes[i:i + ss])
        i += ss

    plt.legend()
    save_path = os.path.join("plots", title, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def visualize(model_dir: str, layer_name: str) -> None:
    for comb in TSNE_COMBS:
        # Collect the most significant fragments (comb results)
        frags = []
        sets_sizes = {}
        marker_sizes = []
        for set in comb:
            pred = pd.read_csv(os.path.join(PREDS_PATH, f'{set}.{MODEL_NAME}comb123456.csv'), sep=SEP)
            p = pred["prob"]
            c = pred["class"]
            f = pred["frag"]

            d = abs(c - p)

            # Filter out fragments above prediction error (1.0 for all fragments)
            valid_indices = [i for i, val in enumerate(d) if val <= 1.0]

            # Use these indices to filter f and d
            filtered_p = [p[i] for i in valid_indices]
            filtered_f = [f[i] for i in valid_indices]

            sets_sizes[set] = len(filtered_f)
            frags.extend(filtered_f)
            marker_sizes.extend(filtered_p)

        if "PB40" in comb[0] and "NLReff" in comb[1]:
            marker_size = [MARKER_SCALE_ALL]
            marker_sizes = np.array(marker_sizes) * MARKER_SCALE_ALL
            colors = ["gainsboro", "darkgray", "blue", "green", "red"]
        elif "NLReff" in comb[0]:
            marker_size = [1.2 * MARKER_SCALE_ALL]
            marker_sizes = np.array(marker_sizes) * (1.2 * MARKER_SCALE_ALL)
            colors = ["darkgray", "blue", "green", "red"]
        else:
            marker_size = [MARKER_SCALE_POS_ONLY]
            marker_sizes = np.array(marker_sizes) * MARKER_SCALE_POS_ONLY
            colors = ["blue", "green", "red", "deeppink", "orange", "cyan", "moccasin"]

        print("Marker size: ", marker_size[0])

        # Tokenize text
        frags = tokenize_seqs(frags, SEQ_CUTOFF + ADDED_TOKENS_PER_SEQ)

        # Collect multidimensional representations from cv models
        mdim_rep = []
        print("Extracting embeddings from specified layer: ", layer_name)
        for model_filepath in glob(os.path.join(model_dir, "*")):
            # Load model
            model = tf.keras.models.load_model(model_filepath)

            # Get output from selected layer
            layer_out = model.get_layer(layer_name).output
            fun = tf.keras.backend.function(model.input, layer_out)
            mdim_rep.append(fun([frags, np.zeros((len(frags), 8943), dtype=np.int8)]))

        mdim_rep = np.concatenate(mdim_rep, axis=1)

        print("Calculating UMAP and t-SNE...")
        x_umap, y_umap = calculate_umap(mdim_rep)
        x_tsne, y_tsne = calculate_tsne(mdim_rep)

        print("Saving plots...")
        # Plot and save UMAP
        create_and_save_plot(x_umap, y_umap, "umap", f'scaled.{".".join(comb)}.pdf', comb, sets_sizes, colors,
                             marker_sizes)
        create_and_save_plot(x_umap, y_umap, "umap", f'{".".join(comb)}.pdf', comb, sets_sizes, colors, marker_size)

        # Plot and save t-SNE
        create_and_save_plot(x_tsne, y_tsne, "tsne", f'scaled.{".".join(comb)}.pdf', comb, sets_sizes, colors,
                             marker_sizes)
        create_and_save_plot(x_tsne, y_tsne, "tsne", f'{".".join(comb)}.pdf', comb, sets_sizes, colors, marker_size)

        print("Finished")


if __name__ == "__main__":
    visualize(MODEL_PATH, "dropout")
