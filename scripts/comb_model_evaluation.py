import os
import csv

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from tqdm import tqdm


tf.get_logger().setLevel('ERROR')


# Constants
ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']
ADDED_TOKENS_PER_SEQ = 2
MODEL_PATH = './proteinbert_models/proteinBERT_full/'
DATA_PATH = './test_sets/'
SEQ_LENGTH = 42
SEQ_CUTOFF = 40

# Preprocessing 
n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i + n_aas for i, token in enumerate(ADDITIONAL_TOKENS)}
token_to_index = {**aa_to_token_index, **additional_token_to_index}
index_to_token = {index: token for token, index in token_to_index.items()}
n_tokens = len(token_to_index)


def parse_seq(seq):
    """Decodes sequence if it is in bytes"""
    if isinstance(seq, str):
        return seq
    elif isinstance(seq, bytes):
        return seq.decode('utf8')
    else:
        raise TypeError('Unexpected sequence type: %s' % type(seq))

def tokenize_seq(seq):
    """Converts sequence of amino acids into a list of token indices."""
    other_token_index = additional_token_to_index['<OTHER>']
    return [additional_token_to_index['<START>']] + [aa_to_token_index.get(aa, other_token_index) for aa in parse_seq(seq)] + [additional_token_to_index['<END>']]

def tokenize_seqs(seqs, seq_len):
    """Converts list of sequences into array of token indices."""
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in map(tokenize_seq, seqs)], dtype = np.int32)


def load_data(data_path):
    """Loads sequences from a CSV file."""
    df = pd.read_csv(data_path)
    labels = df['label']
    sequences = df['seq']
    return labels, sequences

def compute_metrics(y_true, y_scores, fprs):
    """Compute AUROC, average precision, and recall at specified FPRs."""
    # AUROC and average precision
    auroc = roc_auc_score(y_true, y_scores)
    avg_prec = average_precision_score(y_true, y_scores)
    
    # Precision and recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Compute recall (TPR) at specified FPRs
    recall_at_fprs = []
    for desired_fpr in fprs:
        # Find the TPR for this FPR
        # We're looking for the first threshold that gives a FPR <= our desired FPR
        try:
            threshold_index = next(i for i, current_fpr in enumerate(fpr) if current_fpr >= desired_fpr)
            recall_at_fprs.append(tpr[threshold_index])
        except StopIteration:
            recall_at_fprs.append(0.0)

    return auroc, avg_prec, recall_at_fprs


def predict_window(sequences, models, seq_cutoff=SEQ_CUTOFF, batch_size=32):
    """Predicts model output for a window in the sequence and return list of predictions."""
    all_predictions = []
    
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i:i+batch_size]
        for seq in batch_sequences:
            seq_predictions = {}
            for model in models:             
                batch_subsequences = [seq[i:seq_cutoff+i] for i in range(len(seq)-seq_cutoff)]
                if batch_subsequences == []:
                    batch_subsequences = [seq]
                preds = model.predict([tokenize_seqs(batch_subsequences, SEQ_LENGTH), np.zeros((len(batch_subsequences), 8943), dtype=np.int8)], verbose=0)
                flattened_preds = [pred[0] for pred in preds]
                seq_predictions[model] = flattened_preds
                # Take average of each index
            avg_predictions = np.mean(list(seq_predictions.values()), axis=0)
            max_pred = np.max(avg_predictions)
            all_predictions.append(max_pred)
            
    return all_predictions

def predict_window_40(sequences, models, batch_size=32):
    """Predicts model output for a window in the sequence and return list of predictions."""
    all_predictions = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        seq_predictions = {}
        for model in models:
            batch_sequences = sequences[i:i+batch_size]
            preds = model.predict([tokenize_seqs(batch_sequences, SEQ_LENGTH), np.zeros((len(batch_sequences), 8943), dtype=np.int8)], verbose=0)
            flattened_preds = [pred[0] for pred in preds]
            seq_predictions[model] = flattened_preds
        # Flattening the predictions and extending all_predictions list
        avg_predictions = np.mean(list(seq_predictions.values()), axis=0)
        all_predictions.extend(avg_predictions)
        
    return all_predictions


def plot_histogram(predictions, name):
    """Generates and shows a histogram plot of prediction values."""
    plt.hist(predictions, bins=np.arange(0, 1.05, 0.05))
    plt.xlabel('Prediction Values')
    plt.ylabel('Frequency')
    plt.title(name)

def get_dataset_pairs():
    """Gets all pairs of positive and negative sets in a given folder."""
    files = os.listdir(DATA_PATH)
    negative_sets = [f for f in files if '_' not in f]
    positive_sets = [f for f in files if '_' in f]
    pairs = [(neg, pos) for neg in negative_sets for pos in positive_sets]
    return pairs

def get_models():
    """Gets all models in a given folder."""
    print("Getting models")
    models = []
    model_paths = os.listdir(MODEL_PATH)
    for model_path in model_paths:               
            models.append(tf.keras.models.load_model(MODEL_PATH + model_path))
    return models

def main(dataset_path_neg, dataset_path_pos, predictions_cache, models):
    """Main function to execute the script."""
    labels_pos, sequences_pos = load_data(DATA_PATH + dataset_path_pos)
    labels_neg, sequences_neg = load_data(DATA_PATH + dataset_path_neg)
    

    fprs = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    print(f"Dataset: {dataset_path_pos} and {dataset_path_neg}")

    # Check if predictions are already calculated and cached for negative dataset
    if dataset_path_neg in predictions_cache:
        print("Using cached predictions for ", dataset_path_neg)
        y_scores_neg = predictions_cache[dataset_path_neg]
    else:
        if (dataset_path_neg == 'PB40.csv'):
            y_scores_neg = predict_window_40(sequences_neg, models)
        else:
            y_scores_neg = predict_window(sequences_neg, models)
        predictions_cache[dataset_path_neg] = y_scores_neg  # cache the predictions

    # Check if predictions are already calculated and cached for positive dataset
    if dataset_path_pos in predictions_cache:
        print("Using cached predictions for ", dataset_path_pos)
        y_scores_pos = predictions_cache[dataset_path_pos]
    else:
        y_scores_pos = predict_window(sequences_pos, models)
        predictions_cache[dataset_path_pos] = y_scores_pos  # cache the predictions

    labels = pd.concat([labels_pos, labels_neg])
    y_scores = pd.concat([pd.Series(y_scores_pos), pd.Series(y_scores_neg)])
    

    dataset_path_pos_clean = dataset_path_pos.replace('.csv', '')
    dataset_path_neg_clean = dataset_path_neg.replace('.csv', '')

    plt.figure()
    plot_histogram(y_scores_pos, dataset_path_pos_clean)
    plt.savefig('./histograms/comb_' + dataset_path_pos_clean + '_histogram.png')

    plt.figure()
    plot_histogram(y_scores_neg, dataset_path_neg_clean)
    plt.savefig('./histograms/comb_' + dataset_path_neg_clean + '_histogram.png')

    plt.close('all')

    auroc, avg_prec, recall_at_fprs = compute_metrics(labels, y_scores, fprs)

    # Open CSV file
    with open(f'./csvs/{dataset_path_pos_clean}_{dataset_path_neg_clean}.csv', 'a+', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Check if file is empty
        if os.stat(csvfile.name).st_size == 0:
            # Write the header
            csv_writer.writerow(['Model', '#pos', '#neg', 'AUROC', 'AP', 'Rc|FPR1e-1', 'Rc|FPR1e-2', 'Rc|FPR1e-3', 'Rc|FPR1e-4', 'Rc|FPR1e-5'])

        # Write results to CSV file
        row = ["bass_comb123456", len(labels_pos), len(labels_neg), auroc, avg_prec] + list(recall_at_fprs)
        csv_writer.writerow(row)

if __name__ == "__main__":
    pairs = get_dataset_pairs()
    models = get_models()
    predictions_cache = {}
    for neg, pos in pairs:
        main(neg, pos, predictions_cache, models)
    