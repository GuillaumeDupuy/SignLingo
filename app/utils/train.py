import numpy as np
import tensorflow as tf
import pandas as pd
import os
import time

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

def main_model(model_path, tflite_path):
    cwd = os.getcwd()
    dataset_path = cwd + '/data/keypoint_classifier/keypoint.csv'
    label = cwd + '/data/keypoint_classifier/keypoint_classifier_label.csv'
    
    X_dataset, y_dataset = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

    df = pd.read_csv(label, header=None)
    NUM_CLASSES = len(df)
    model = create_model(21 * 2, NUM_CLASSES)
    train_model(model, X_train, y_train, X_test, y_test, model_path)

    # convert the model to a TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_path= cwd + f'/models/keypoint_classifier/{tflite_path}')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))
    interpreter.invoke()

    # get the current time
    current_time = time.strftime('%d-%m-%Y_%H-%M')

    # save the model to the model directory
    model.save(f'{cwd}/models/keypoint_classifier/{current_time}_keypoint_classifier.hdf5')

    # save the tflite model to the model directory
    with open(f'{cwd}/models/keypoint_classifier/{current_time}_keypoint_classifier.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    main_model('keypoint_classifier.hdf5', 'keypoint_classifier.tflite')