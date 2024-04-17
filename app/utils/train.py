import numpy as np
import tensorflow as tf
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

RANDOM_SEED = 42

def load_dataset(filepath):
    """Load the dataset from a CSV file."""
    dataset = np.loadtxt(filepath, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    labels = np.loadtxt(filepath, delimiter=',', dtype='int32', usecols=(0))
    return dataset, labels

def create_model(input_dim, num_classes):
    """Create a TensorFlow neural network model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, X_train, y_train, X_test, y_test, model_save_path):
    """Train the model and save the best version."""
    cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, verbose=1, save_weights_only=False)
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_test, y_test), callbacks=[cp_callback, es_callback])
    
    model.save(model_save_path, include_optimizer=False)

def main_model():
    cwd = os.getcwd()
    dataset_path = cwd + '/model/keypoint_classifier/keypoint.csv'
    model_save_path = cwd + '/model/keypoint_classifier/keypoint_classifier.hdf5'
    label = cwd + '/model/keypoint_classifier/keypoint_classifier_label.csv'
    
    X_dataset, y_dataset = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

    df = pd.read_csv(label, header=None)
    NUM_CLASSES = len(df)
    model = create_model(21 * 2, NUM_CLASSES)
    train_model(model, X_train, y_train, X_test, y_test, model_save_path)