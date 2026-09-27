import pandas as pd
import glob
import os
dataset_location = r'C:\Codes\DP'
from torch.utils.data import Dataset
import torch
import numpy as np

terrain_labels =[
    'concrete',
    'grass',
    'pebbles',
    'sand',
    'paving stone',
    'synthetic running track',
]

sensor_types = ['ACC']#,'MAG','GYR']

def read_magnitudes():
    magnitude_list = {}
    for sensor_type in sensor_types:
        magnitude_list[sensor_type] = []
        for idx,terrain_label in enumerate(terrain_labels):
            # Use scandir to get only files in the current directory
            with os.scandir(dataset_location+f'.\\fft_data3\\{sensor_type}\\{terrain_label}') as entries:
                terrain_data_names = [entry.name for entry in entries if entry.is_file()]
            
            for terrain_name in terrain_data_names:
                sample = pd.read_csv(dataset_location+f'\\fft_data3\\{sensor_type}\\{terrain_label}\\'+terrain_name).loc[:,'magnitude']
                magnitude_list[sensor_type].append([idx, sample])

    print("Magnitudes data read successfully")
    return magnitude_list

# magnitudes = read_magnitudes()
# print(magnitudes['ACC'][5][0])
# print(type(magnitudes['ACC'][5][1][0:5].to_numpy()))


class TerrainMagnitudeDataset(Dataset):

    def __init__(self,magnitude_list,sensor = 'ACC',transform = None):
        self.magnitude_list = magnitude_list
        self.sensor = sensor

        self.terrain_dataset = self.magnitude_list[self.sensor]
        self.transform = transform

    def __len__(self):
        return len(self.terrain_dataset)

    def __getitem__(self,idx):
        signal = self.terrain_dataset[idx][1]
        label = self.terrain_dataset[idx][0]
        if self.transform:
            signal = self.transform(signal)
            #label = self.transform(label)

        else :
            signal = torch.tensor(signal, dtype = torch.float32)
        return signal,label

class TerrainTimeSeriesDataset(Dataset):
    def __init__(self, accelerometer,gyroscope,labels,transform = None):
        self.accelerometer = accelerometer
        self.gyroscope = gyroscope
        self.labels = labels
        self.transform = transform

        assert len(self.accelerometer) == len(self.gyroscope) == len(self.labels), "Length of accelerometer, gyroscope data, and labels must be the same"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        acc_sample = self.accelerometer[idx]
        gyr_sample = self.gyroscope[idx]

        sample = np.zeros([len(acc_sample), 2])
        sample[:,0] = acc_sample
        sample[:,1] = gyr_sample


        label_sample = self.labels[idx]

        if self.transform:
            acc_sample = self.transform(acc_sample)
            gyr_sample = self.transform(gyr_sample)

            sample = self.transform(sample)

        else:
            acc_sample = torch.tensor(acc_sample, dtype = torch.float32)
            gyr_sample = torch.tensor(gyr_sample, dtype = torch.float32)

            sample = torch.tensor(sample, dtype = torch.float32)

        label_sample = torch.tensor(label_sample,dtype = torch.long)

        return sample,label_sample



def train_step(model, train_loader, optimizer, criterion, epoch, device):
    model = model.to(device)

    loss_sum = 0
    total = 0
    correct = 0

    model.train()
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        #targets = targets.squeeze().long()
        loss = criterion(outputs, targets)        
        
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        
        _, predicted = torch.max(outputs,dim=1)
        total += targets.shape[0]
        correct += int((predicted == targets).sum())
        
    accuracy = correct / total 
    loss_avg = loss_sum / len(train_loader)
    
    return accuracy, loss_avg


def test_step(model, test_loader, optimizer, criterion, epoch, device):
    model = model.to(device)

    model.eval()
    loss_sum = 0
    total, correct = 0, 0
    
    y_predicted = []
    y_target = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            
            targets = targets.squeeze().long()
            #outputs = outputs.softmax(dim=-1)
            
            loss = criterion(outputs, targets)
            loss_sum += loss.item()
            
            _, predicted = torch.max(outputs,dim=1)
            total += targets.shape[0]
            correct += int((predicted == targets).sum())

            y_predicted.extend(predicted.cpu().numpy())
            y_target.extend(targets.cpu().numpy())
            
    avg_loss = loss_sum / len(test_loader)
    accuracy = correct / total
    
    return accuracy, avg_loss, y_predicted, y_target