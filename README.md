# SignLingo

## Description

SignLingo is a web app that translates sign language to text & speech.
The web app use a trained model to predict the sign language from a video stream.
After the prediction, the text is translated to the language of choice and print it on the screen and read it out loud.

## Model

The model work this mediapipe, it allows us to recognizes hand signs and finger gestures with a simple MLP using the detected key points.

![project.gif](./project.gif)

This repository contains the following contents.

- Sample program
- Hand sign recognition model(TFLite)
- Finger gesture recognition model(TFLite)
- Learning data for hand sign recognition and notebook for learning
- Learning data for finger gesture recognition and notebook for learning

## Requirements

mediapipe==0.9.0.1
OpenCV==4.7.0.68
Tensorflow==2.12.0
scikit-learn==1.0.2
googletrans==4.0.0rc1
gtts==2.4.0
streamlit==1.29.0

Execute the following command to install the required packages.

```python
pip install -r requirements.txt
```

## Usage

For run the app, execute the following command.

```python
cd Model 2.0
python app.py
```

## Training

Hand sign recognition and finger gesture recognition can add and change training data and retrain the model.

### Hand sign recognition training
#### 1.Learning data collection

Press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」

If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv" as shown below.
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Key point coordinates

![keypoint.png](./keypoint.png)

#### 2.Model training

Open "train_model.ipynb" in Jupyter and execute all file.
To change the number of training data classes, change the value of "NUM_CLASSES = 3"
and modify the label of "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate.

### Finger gesture recognition training

#### 1.Learning data collection

Press "h" to enter the mode to save the history of fingertip coordinates (displayed as "MODE:Logging Point History").

If you press "0" to "9", the key points will be added to "model/point_history_classifier/point_history.csv" as shown below.
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Coordinate history

2.Model training
Open "train_model_history.ipynb" in Jupyter Notebook and execute from top to bottom.
To change the number of training data classes, change the value of "NUM_CLASSES = 4" and
modify the label of "model/point_history_classifier/point_history_classifier_label.csv" as appropriate.

## Ressources

- [WLASL](https://github.com/dxli94/WLASL)
- [Sign-lanuage-datasets](https://github.com/YAYAYru/sign-lanuage-datasets)
- [Sign Language Dataset](https://www.kaggle.com/datasets/ardamavi/27-class-sign-language-dataset?select=X.npy)
- [Leap Motion ASL & BSL](https://www.kaggle.com/datasets/birdy654/sign-language-recognition-leap-motion)
- [BSL Recognition & Transfer Learning ASL](https://www.mdpi.com/1424-8220/20/18/5151)
- [Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/data)
- [Real-Time-Interaction-Using-Sign-Language](https://github.com/Nikhilkohli1/Real-Time-Interaction-Using-Sign-Language)