import pandas as pd
import os
import csv

cwd = os.getcwd()

def record_video(landmark_list, point_history_list):
    df = pd.read_csv(cwd + '/data/keypoint_classifier/keypoint.csv')
    number = len(df) + 1

    with open(cwd + '/data/keypoint_classifier/keypoint.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])

    with open(cwd + '/data/keypoint_classifier/keypoint_history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([number, *point_history_list])
