import os
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PATH, SEQUENCE_CUTOFF
from preprocessing import parse_fasta
from model_utils import load_models_from_directory, predict_window

tf.get_logger().setLevel('ERROR')

def main():
    tsne_model = TSNE(n_components=2, verbose=1, random_state=42)
    sequence_labels = []
    sequence_embeddings = []
    all_values = []

    unique_labels = set()

    models = load_models_from_directory()
    for dataset in os.listdir(DATA_PATH):
        print(f"Dataset: {dataset}...")
        sequences = parse_fasta(DATA_PATH + dataset)
        embeddings, labels, coverage, values = predict_window(sequences, models, dataset, SEQUENCE_CUTOFF)
        sequence_embeddings.append(embeddings)
        sequence_labels.extend(labels)
        all_values.extend(values)

        unique_labels.add(dataset)
    
        if "bass_ntm_domain" in dataset or "fass_ntm_domain" in dataset:
            print("Mean, Std, Min, Max:")
            print(np.mean(coverage), np.std(coverage), np.min(coverage), np.max(coverage))


            plt.violinplot(coverage)
            plt.title(dataset[:-3])
            plt.ylabel('Coverage')
            plt.yticks(np.arange(0, 1.1, step=0.1))

            # Save the histogram to a file
            histogram_path = './results/violin_plots/' + dataset[:-3] + '.pdf'
            plt.savefig(histogram_path)

            # Clear the plot data to avoid duplication in further plots
            plt.clf()

    tsne_embeddings = tsne_model.fit_transform(np.concatenate(sequence_embeddings))

    # Visualization
    sns.scatterplot(
        x=[x[0] for x in tsne_embeddings],
        y=[x[1] for x in tsne_embeddings],
        hue=sequence_labels,
        size=all_values
    )
    plt.title("Combined t-SNE")
    plt.xlabel("1")
    plt.ylabel("2")
    # hide values on x and y axis
    plt.xticks([], [])
    plt.yticks([], [])
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles[0:len(unique_labels)], labels=labels[0:len(unique_labels)], fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plot_path = './results/tsne/' + '.'.join(os.listdir(DATA_PATH)) + '.pdf'
    plt.savefig(plot_path)

    plt.show()

if __name__ == "__main__":
    main()