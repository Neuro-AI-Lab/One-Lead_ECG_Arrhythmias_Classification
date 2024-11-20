from torch.utils.data import Dataset
from time import time
import torch
from math import pow
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
from scipy import signal 

def save_time_series_plot(time_series_data, predicted_label, label_name, correctness, idx):
    """
    시계열 데이터를 그래프로 그린 후 저장하는 함수.
    
    time_series_data: 1D 시계열 데이터 (텐서 또는 리스트)
    label_name: 'Normal', 'AF', 'Other' 등 라벨 이름
    correctness: 'correct' or 'incorrect'
    idx: 샘플 번호
    """
    # 저장할 경로 설정
    directory = f"./images/{label_name}/{correctness}/"
    os.makedirs(directory, exist_ok=True)  # 경로가 없으면 생성

    # 이미지 파일 이름
    file_path = os.path.join(directory, f"sample{idx+1}.png")

    time_series = time_series_data.squeeze(0).squeeze(0)

    # 시계열 데이터 플로팅
    plt.figure()
    plt.plot(time_series)

    if correctness == 'correct':
        plt.title(f'{label_name} - {correctness} - Sample {idx+1}')
    elif correctness == 'incorrect':
        plt.title(f'{label_name} - {correctness} (Predicted: {predicted_label}) - Sample {idx+1}')

    plt.xlabel('Sampled points(300 points=1 second)')
    plt.ylabel('Amplitude')
    
    # 그래프 이미지로 저장
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
def f1_score(labels, y_pred):
    labels = np.concatenate(labels).tolist()
    y_pred = np.concatenate(y_pred).tolist()
    print(classification_report(labels, y_pred, zero_division = 0, digits=3))

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_weights):
    

    train_loss_list = list()
    val_loss_list = list()
    
    train_acc_list = list()
    val_acc_list = list()
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time()
        
        for inputs, labels in train_loader:
           
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            # print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        end_time = time()
        epoch_time = end_time - start_time

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, time: {epoch_time}')

        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accuracy)

    if save_weights == True:
        torch.save(model.state_dict(), "./weights/physionet2017/weights_train_dbscan_tf_h8.pth")
    
    train_loss_npy = np.array(train_loss_list)
    train_acc_npy = np.array(train_acc_list)
    val_loss_npy = np.array(val_loss_list)
    val_acc_npy = np.array(val_acc_list)

    return train_loss_npy, train_acc_npy, val_loss_npy, val_acc_npy

def train_model_with_scheduler(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_weights, scheduler):
    

    train_loss_list = list()
    val_loss_list = list()
    
    train_acc_list = list()
    val_acc_list = list()
    
    model = model.to(device)


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time()
        
        for inputs, labels in train_loader:
                
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            # print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        end_time = time()
        epoch_time = end_time - start_time

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, time: {epoch_time}')

        #scheduler.step(val_loss / len(val_loader))  # validation loss를 기반으로 학습률 조정
        scheduler.step()
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accuracy)

    if save_weights == True:
        torch.save(model.state_dict(), "./weights/physionet2017/weights_train_dbscan_e150.pth")
    
    train_loss_npy = np.array(train_loss_list)
    train_acc_npy = np.array(train_acc_list)
    val_loss_npy = np.array(val_loss_list)
    val_acc_npy = np.array(val_acc_list)

    return train_loss_npy, train_acc_npy, val_loss_npy, val_acc_npy
        
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / total
    return val_loss, val_accuracy

def test_model(model, test_loader, criterion, device, save_plot):
    
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    inputs_list = list()
    labels_list = list()
    predicteds_list = list()


    with torch.no_grad():
        for inputs, labels in test_loader:
            
            inputs, labels = inputs.to(device), labels.to(device)
        
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)

            inputs_list.append(inputs.clone().detach().cpu())
            labels_list.append(labels.clone().detach().cpu())
            predicteds_list.append(predicted.clone().detach().cpu())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    f1_score(labels_list, predicteds_list)

     # Initialize a dictionary to store counts for each class (0: Normal, 1: AF, 2: Other)
    results_dict = {
        'Normal': {'Correct': 0, 'Incorrect': 0, 'Total': 0},
        'AF': {'Correct': 0, 'Incorrect': 0, 'Total': 0},
        'Other': {'Correct': 0, 'Incorrect': 0, 'Total': 0}
    }

    # 각 클래스에 대해 Correct, Incorrect 샘플 저장
    sample_counts = {
        'Normal': {'Correct': 0, 'Incorrect': 0},
        'AF': {'Correct': 0, 'Incorrect': 0},
        'Other': {'Correct': 0, 'Incorrect': 0}
    }

    # Mapping from label numbers to names
    label_names = {0: 'Normal', 1: 'AF', 2: 'Other'}
    
    # Count Correct, Incorrect, and Total for each class
    for u, actual, predicted in zip(inputs_list, labels_list, predicteds_list):
        label_name = label_names[actual.item()]  # Get the corresponding name (Normal, AF, Other)
        if actual == predicted:
            results_dict[label_name]['Correct'] += 1
            if sample_counts[label_name]['Correct'] < 10 and save_plot == True:
                save_time_series_plot(u, predicted.item(), label_name, 'correct', sample_counts[label_name]['Correct'])
                sample_counts[label_name]['Correct'] += 1
                
        else:
            results_dict[label_name]['Incorrect'] += 1
            if sample_counts[label_name]['Incorrect'] < 10 and save_plot == True:
                save_time_series_plot(u, predicted.item(), label_name, 'incorrect', sample_counts[label_name]['Incorrect'])
                sample_counts[label_name]['Incorrect'] += 1
                
        results_dict[label_name]['Total'] += 1
    
    # Convert dictionary to DataFrame
    results_df = pd.DataFrame(results_dict).T  # Transpose to get desired structure

    return results_df
 

class ECGDataset(Dataset):
    def __init__(self, data_paths):
        # CSV 파일 경로를 받아서 데이터프레임으로 저장합니다.
        self.data_path_df = pd.read_csv(data_paths, dtype={'File Path': 'object', 'Label': 'object'})                                       
        
    def __len__(self):
        return len(self.data_path_df)

    def __getitem__(self, idx):

        # [b, a] = signal.butter(6, [5 / 300 * 2, 35 / 300 * 2], btype='bandpass')
        
        # 현재 인덱스에 해당하는 파일 경로를 CSV에서 가져옵니다.
        data_path = self.data_path_df['File Path'].iloc[idx]
        
        # .mat 파일을 각각 로드합니다.
        mat_data = sio.loadmat(data_path)
        label_data = int(self.data_path_df['Label'].iloc[idx])
        
        # 데이터를 텐서로 변환합니다.
        tensor_data =  torch.tensor(mat_data['val'], dtype=torch.float32)
        tensor_label = torch.tensor(label_data, dtype=torch.long)

        #filtered_data = signal.filtfilt(b, a, np_data).copy()
        
        return tensor_data, tensor_label