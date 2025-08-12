import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import datetime
import os

from torch import optim
from torchmetrics import Accuracy
from torch.utils.data import Dataset
from make_dataset import label_dict_from_config_file


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.list_label = label_dict_from_config_file("hand_gesture.yaml")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128, len(self.list_label))
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

    def predict(self, x, threshold=0.8):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=-1)(logits)
        chosen_idx = torch.argmax(softmax_prob, dim=1)
        return torch.where(softmax_prob[0, chosen_idx] > threshold, chosen_idx, -1)

    def predict_with_know_class(self, x):
        logits = self(x)
        softmax_prod = nn.Softmax(dim=1)(logits)
        return torch.argmax(softmax_prod, dim=1)

    def zscore(self, logits):
        return -torch.amax(logits, dim=1)


class CustomImageDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.label = torch.from_numpy(self.data.iloc[:, 0].to_numpy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        one_hot_label = self.label[index]
        data = torch.from_numpy(
            self.data[index, :1].to_numpy(dtype=np.float64))
        return data, one_hot_label


class EarlyStoper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.watched_metrics = np.inf

    def early_stop(self, current_value):
        if current_value < self.watched_metrics:
            self.watched_metrics = current_value
            self.counter = 0
        elif current_value > (self.watched_metrics + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False


def train(train_loader, val_loader, model, loss_function, optimizer, epochs, early_stop):
    best_vloss = 1_000_000
    timesamp = datetime.datetime.now().strftime('%d-%m %H:%M')
    for epoch in range(epochs):
        model.train(True)
        running_loss = 0.0
        acc_train = Accuracy(num_classes=len(LIST_LABEL), task="multiclass")
        for batch_number, data in enumerate(train_loader):
            input, label = data
            optimizer.zero_grad()
            pred = model(input)
            loss = loss_function(pred, label)
            loss.backward()
            optimizer.step()
            acc_train.update(model.predict_with_know_class(input), label)
            running_loss += loss.item()

        avg_loss = running_loss/len(train_loader)

        model.train(False)
        running_vloss = 0.0
        acc_val = Accuracy(num_classes=len(LIST_LABEL), task="multiclass")
        for batch_number, val_data in enumerate(val_loader):
            val_input, val_label = val_data
            pred = model(val_input)
            vloss = loss_function(pred, val_label)
            running_vloss += vloss.item()
            acc_val.update(model.predict_with_know_class(input), val_label)

        avg_vloss = running_vloss/len(val_loader)

        print(f"EPOCHS: {epoch}")
        print(f"Accuracy train: {acc_train.compute().item()}, val: {
              acc_val.compute().item()}")
        print(f"Loss train: {avg_loss}, val{avg_vloss}")

        print("Traning vs. Validation loss",
              {"Training": avg_loss, "Validation": avg_vloss}, epoch + 1)
        print("Traning vs. Validation accuracy",
              {"Training": acc_train.compute().item(),
               "Validation": acc_val.compute().item()}, epoch + 1)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model_path = f"./{save_path}/model_{
                timesamp}_{model.__class__.__name__}_best"
            torch.save(model.state_dict(), best_model_path)

        if early_stop.early_stop(avg_vloss):
            print(f"stopping at epoch {epoch}, minimum: {
                  early_stop.watched_metrics}")
            break

    model_path = f'./{save_path}/model_{timesamp}_{model.__class__.__name__}_last'
    torch.save(model._state_dict(), model_path)

    print(acc_val.compute())
    return model, best_model_path


DATA_FOLDER_PATH = "./data2/"
LIST_LABEL = label_dict_from_config_file("hand_gesture.yaml")
train_path = os.path.join(DATA_FOLDER_PATH, "landmark_train.csv")
val_path = os.path.join(DATA_FOLDER_PATH, "landmark_val.csv")
save_path = './models' 
os.makedirs(save_path, exist_ok=True)

trainset = CustomImageDataset(train_path)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=40, shuffle=True)

valset = CustomImageDataset(os.path.join(val_path))
val_loader = torch.utils.data.DataLoader(valset, batch_size=50, shuffle=False)

model = NeuralNetwork()
loss_function = nn.CrossEntropyLoss()
early_stopper = EarlyStoper(patience=30, min_delta=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model, best_model_path = train(
    trainloader, val_loader, model, loss_function, optimizer, 300, early_stopper)
