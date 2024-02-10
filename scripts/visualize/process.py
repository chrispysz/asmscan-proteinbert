import os
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PATH, SEQUENCE_CUTOFF
from preprocessing import load_data
from model_utils import load_models_from_directory, predict_window

tf.get_logger().setLevel('ERROR')

def main():
    tsne_model = TSNE(n_components=2, verbose=1, random_state=42)
    sequence_labels = []
    sequence_embeddings = []

    unique_labels = set()

    models = load_models_from_directory()
    for dataset in os.listdir(DATA_PATH):
        sequences = load_data(DATA_PATH + dataset)
        embeddings, labels, relative_positions = predict_window(sequences, models, dataset, SEQUENCE_CUTOFF)
        sequence_embeddings.append(embeddings)
        sequence_labels.extend(labels)

        unique_labels.add(dataset)

        if len(relative_positions) > 0:
            plt.hist(relative_positions, bins=range(1, 121, 20), edgecolor='black')
            plt.title(dataset[:-4])
            plt.xlabel('Position in Sequence (%)')
            plt.ylabel('Frequency')

            # Save the histogram to a file
            histogram_path = './results/histograms/'+dataset[:-4]+'.png'
            plt.savefig(histogram_path)

            # Clear the plot data to avoid duplication in further plots
            plt.clf()

    tsne_embeddings = tsne_model.fit_transform(np.concatenate(sequence_embeddings))

    # Visualization
    sns.scatterplot(
        x=[x[0] for x in tsne_embeddings],
        y=[x[1] for x in tsne_embeddings],
        hue=sequence_labels,
        s=9
    )
    plt.title("Combined t-SNE")
    plt.xlabel("1")
    plt.ylabel("2")
    # hide values on x and y axis
    plt.xticks([], [])
    plt.yticks([], [])
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()