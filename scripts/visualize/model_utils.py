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

def calculate_relative_position(sequence_length, max_position_index):
    """Calculates the relative position of the max prediction."""
    if sequence_length <= SEQUENCE_CUTOFF:
        return 0
    relative_position = (max_position_index + 20) / sequence_length
    return int(relative_position*100)


def predict_window(sequences, models, dataset_path, seq_cutoff=SEQUENCE_CUTOFF, batch_size=32):
    """Predicts model output for a window in the sequence and return list of predictions."""
    all_embeddings = []
    sequence_labels = []
    relative_positions = []

    dataset_label = dataset_path.split('/')[-1]
    dataset_label = dataset_label[:-4]
    
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i:i+batch_size]
        for seq in batch_sequences:
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

            relative_position = calculate_relative_position(len(seq), max_pred_index)
            
            if relative_position != 0:
                relative_positions.append(calculate_relative_position(len(seq), max_pred_index))

            
            for model in models:
                sequence_labels.append(dataset_label)
            
                embedding = seq_embeddings[model][max_pred_index]
                all_embeddings.append(np.array(embedding))

    return all_embeddings, sequence_labels, relative_positions
                