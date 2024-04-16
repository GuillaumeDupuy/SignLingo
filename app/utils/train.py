import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

# RANDOM_SEED = 42

# cwd = os.getcwd()

# dataset = cwd + '/app/model/keypoint_classifier/keypoint.csv'
# model_save_path = cwd + '/app/model/keypoint_classifier/keypoint_classifier.hdf5'
# label = cwd + '/app/model/keypoint_classifier/keypoint_classifier_label.csv'

# df = pd.read_csv(label, header=None)

# NUM_CLASSES = len(df)

# X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
# y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

# X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Input((21 * 2, )),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(20, activation='relu'),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
# ])

# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     model_save_path, verbose=1, save_weights_only=False)

# es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.fit(
#     X_train,
#     y_train,
#     epochs=1000,
#     batch_size=128,
#     validation_data=(X_test, y_test),
#     callbacks=[cp_callback, es_callback]
# )

# val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)

# predict_result = model.predict(np.array([X_test[0]]))
# print(np.squeeze(predict_result))
# print(np.argmax(np.squeeze(predict_result)))

# model.save(model_save_path, include_optimizer=False)

# tflite_save_path = '../model/keypoint_classifier/keypoint_classifier.tflite'

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_quantized_model = converter.convert()

# open(tflite_save_path, 'wb').write(tflite_quantized_model)

# interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))

# interpreter.invoke()
# tflite_results = interpreter.get_tensor(output_details[0]['index'])
# print(np.squeeze(tflite_results))
# print(np.argmax(np.squeeze(tflite_results)))