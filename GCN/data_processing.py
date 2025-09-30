
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import joblib
import os



print(os.getcwd())

index_nose = 0
index_left_eye_inner =3
index_left_eye = 6
index_left_eye_outer = 9
index_right_eye_inner = 12
index_right_eye = 15
index_right_eye_outer = 18
index_left_ear = 21
index_right_ear = 24
index_mouth_left = 27
index_mouth_right = 30
index_left_shoulder = 33
index_right_shoulder = 36
index_left_elbow = 39
index_right_elbow = 42
index_left_wrist = 45
index_right_wrist = 48
index_left_pinky = 51
index_right_pinky = 54
index_left_index = 57
index_right_index = 60
index_left_thumb = 63
index_right_thumb = 66
index_left_hip = 69
index_right_hip = 72
index_left_knee = 75
index_right_knee = 78
index_left_ankle = 81
index_right_ankle = 84
index_left_heel = 87
index_right_heel = 90
index_left_foot = 93
index_right_foot = 96


class FallDataLoader:
    def __init__(self, dir):
        super().__init__()
        self.num_channel = 3
        self.dir = dir
        self.body_part = self.body_parts()
        self.dataset = []
        self.sequence_length = []
        self.num_timestep = 100
        self.new_label = []
        self.train_x, self.train_y = self.import_dataset()
        self.batch_size =  self.train_y.shape[0] // self.num_timestep
        self.num_joints = len(self.body_part)
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        self.scaled_x, self.scaled_y = self.preprocessing()

    def body_parts(self):
        body_parts = [      

            index_nose,
            index_left_eye_inner,
            index_left_eye,
            index_left_eye_outer,
            index_right_eye_inner,
            index_right_eye,
            index_right_eye_outer,
            index_left_ear ,
            index_right_ear ,
            index_mouth_left ,
            index_mouth_right ,
            index_left_shoulder ,
            index_right_shoulder ,
            index_left_elbow ,
            index_right_elbow ,
            index_left_wrist ,
            index_right_wrist ,
            index_left_pinky ,
            index_right_pinky ,
            index_left_index ,
            index_right_index ,
            index_left_thumb ,
            index_right_thumb ,
            index_left_hip ,
            index_right_hip ,
            index_left_knee ,
            index_right_knee ,
            index_left_ankle ,
            index_right_ankle ,
            index_left_heel ,
            index_right_heel ,
            index_left_foot ,   
            index_right_foot ,                    
        ]
          

        return body_parts

    def import_dataset(self):
        train_x = (
            pd.read_csv(f"./{self.dir}/Train_X.csv", header=None).iloc[:, :].values
        )
        print(train_x.shape)
        train_y = (
            pd.read_csv(f"./{self.dir}/Train_Y.csv", header=None).iloc[:, :].values
        )
        print(train_y.shape)
        print(train_y)
        return train_x, train_y

    def preprocessing(self):
        X_train = np.zeros(
            (self.train_x.shape[0], self.num_joints * self.num_channel)
        ).astype("float32")
        for row in range(self.train_x.shape[0]):
            counter = 0
            for parts in self.body_part:
                for i in range(self.num_channel):
                    X_train[row, counter + i] = self.train_x[row, parts + i]
                counter += self.num_channel

        X_train = self.sc1.fit_transform(X_train)

        num_batches = X_train.shape[0] // self.num_timestep
        X_train_ = np.zeros(
            (num_batches, self.num_timestep, self.num_joints, self.num_channel)
        )
        Y_train_ = np.zeros((num_batches, self.num_timestep))

        for batch in range(num_batches):
            for timestep in range(self.num_timestep):
                for joint in range(self.num_joints):
                    for channel in range(self.num_channel):
                        X_train_[batch, timestep, joint, channel] = X_train[
                            (batch * self.num_timestep) + timestep,
                            (joint * self.num_channel) + channel,
                        ]
                Y_train_[batch, timestep] = self.train_y[
                    (batch * self.num_timestep) + timestep
                ]

        X_train_ = self.sc2.fit_transform(X_train_.reshape(-1, self.num_joints * self.num_channel)).reshape(X_train_.shape)
        # Y_train_ = self.sc2.fit_transform(Y_train_)

        print("X_train_ shape:", X_train_.shape)
        print("Y_train_ shape:", Y_train_.shape)

        return X_train_, Y_train_
 
    
   
# create train custom dataset class
class TrainData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        sample = {"data": data, "label": labels}
        sample = {"data": data, "label": labels}
        print("Shape of data in TrainData:", data.shape)
        print("Shape of labels in TrainData:", labels.shape)
        return sample


# create test custom dataset class
class TestData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

      
    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        sample = {"data": data, "label": labels}
      
        return sample