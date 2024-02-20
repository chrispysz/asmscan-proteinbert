import os
import numpy as np
import tensorflow as tf
from preprocessing import tokenize_sequences
from config import MODEL_PATH, SEQUENCE_LENGTH, SEQUENCE_CUTOFF
from tqdm import tqdm


def load_models_from_directory():
    """Gets all models in a given folder."""
    print("Getting models")
    models = []
    model_paths = os.listdir(MODEL_PATH)
    for model_path in model_paths:    
        model = tf.keras.models.load_model(MODEL_PATH + model_path)
        layer_output = model.get_layer('dropout').output
        model = tf.keras.models.Model(inputs=model.input, outputs=[layer_output, model.layers[-1].output])           
        models.append(model)
    return models

def calculate_relative_positions(sequence_length, start_index):
    """Generates a list of relative positions for each point in the 40-position window."""
    if sequence_length <= SEQUENCE_CUTOFF:
        return []
    relative_positions = [(start_index + i + 1) / sequence_length * 100 for i in range(40)]
    return relative_positions


def predict_window(sequences, models, dataset_path, seq_cutoff=SEQUENCE_CUTOFF, batch_size=32):
    """Predicts model output for a window in the sequence and return list of predictions."""
    all_embeddings = []
    sequence_labels = []
    all_relative_positions = []

    dataset_label = dataset_path.split('/')[-1]
    dataset_label = dataset_label[:-4]
    
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i:i+batch_size]
        for seq in batch_sequences:
            sequence_labels.append(dataset_label)
            seq_predictions = {}
            seq_embeddings = {}

            for model in models:             
                batch_subsequences = [seq[i:seq_cutoff+i] for i in range(len(seq)-seq_cutoff)]
                if not batch_subsequences:
                    batch_subsequences = [seq]
                preds = model.predict([tokenize_sequences(batch_subsequences, SEQUENCE_LENGTH), np.zeros((len(batch_subsequences), 8943), dtype=np.int8)], verbose=0)
                flattened_preds = [pred[0] for pred in preds[1]]
                flattened_embeddings = preds[0]
                seq_predictions[model] = flattened_preds
                seq_embeddings[model] = flattened_embeddings
            
            avg_predictions = np.mean(list(seq_predictions.values()), axis=0)
            max_pred_index = np.argmax(avg_predictions)

            positions = calculate_relative_positions(len(seq), max_pred_index)
            
            if len(positions) > 0:
                all_relative_positions.extend(positions)

            max_embeddings = []
            for model in models:
            
            
                embedding = seq_embeddings[model][max_pred_index]
                max_embeddings.extend(embedding)
                
            all_embeddings.append(max_embeddings)

    return all_embeddings, sequence_labels, all_relative_positions
                