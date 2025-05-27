import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

import os
print("Current Working Directory:", os.getcwd())
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Working directory changed to:", os.getcwd())

# create the StudentPerformanceDataset subclass
class StudentPerformanceDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.scaler = MinMaxScaler()

    def load_and_process_data(self):
        self.df = pd.read_csv(self.file_path)

        # remove non-numeric columns
        self.df.drop(columns=['Student_ID'], inplace=True)

        categorical_columns = ['Gender', 'Parental_Education_Level', 'Internet_Access_at_Home',
                               'Extracurricular_Activities', 'Pass_Fail']

        # label encoding
        le = LabelEncoder()
        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = le.fit_transform(self.df[col])

        # normalize numerical columns
        numerical_columns = ['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores', 'Final_Exam_Score']
        self.df[numerical_columns] = self.scaler.fit_transform(self.df[numerical_columns])

        X = self.df.drop(columns=['Final_Exam_Score', 'Pass_Fail']).values
        y = self.df['Final_Exam_Score'].values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2025)

        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_val_data(self):
        return self.X_val, self.y_val


dataset = StudentPerformanceDataset("data/student_performance_dataset.csv")
dataset.load_and_process_data()  

X_train, y_train = dataset.get_train_data()
X_val, y_val = dataset.get_val_data()

print("Training Data Shape:", X_train.shape)
print("Validation Data Shape:", X_val.shape)

