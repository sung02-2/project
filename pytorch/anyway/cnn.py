import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import random

# ========== 設定隨機種子 ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========== 超參數 ==========
input_dim = 6
seq_length = 64
hidden_dim = 256
batch_size = 64
learning_rate = 1e-4
num_epochs = 100
patience = 10
dropout_rate = 0.3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 自訂資料集 ==========
class MyDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = pd.read_csv(data_path)
        self.labels = pd.read_csv(label_path)
        features = ['Normalized_Tick', 'ViewX', 'ViewY', 'Total_Angular_Velocity', 'Angular_Acceleration', 'Fired']
        self.data[features] = (self.data[features] - self.data[features].mean()) / (self.data[features].std() + 1e-8)

        self.data['DemoID_GroupID'] = list(zip(self.data['DemoID'], self.data['GroupID']))
        self.labels['DemoID_GroupID'] = list(zip(self.labels['DemoID'], self.labels['GroupID']))

        self.groups = self.data.groupby('DemoID_GroupID')
        self.group_labels = self.labels.set_index('DemoID_GroupID')['player?'].to_dict()

        transfer_keys = set(self.groups.groups.keys())
        label_keys = set(self.group_labels.keys())
        valid_keys = list(transfer_keys & label_keys)
        self.group_keys = valid_keys

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        group_key = self.group_keys[idx]
        group_data = self.groups.get_group(group_key)
        X = group_data[['Normalized_Tick', 'ViewX', 'ViewY', 'Total_Angular_Velocity', 'Angular_Acceleration', 'Fired']].values
        y = self.group_labels[group_key]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ========== 強化版 CNN ==========
class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim//2)

        self.dropout = nn.Dropout(dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim//2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(torch.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ========== EarlyStopping ==========
class EarlyStopping:
    def __init__(self, patience=10, save_path="best_cnn_model.pt"):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
        return self.counter >= self.patience

# ========== 資料準備 ==========
data_path = "../output/transfer_all.csv"
label_path = "../output/label_all.csv"

dataset = MyDataset(data_path, label_path)
total_indices = np.arange(len(dataset))

train_idx, temp_idx = train_test_split(total_indices, test_size=0.3, random_state=SEED, shuffle=True)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=SEED, shuffle=True)

train_set = torch.utils.data.Subset(dataset, train_idx)
val_set = torch.utils.data.Subset(dataset, val_idx)
test_set = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

print(f"✅ Train data: {len(train_set)} samples")
print(f"✅ Validation data: {len(val_set)} samples")
print(f"✅ Test data: {len(test_set)} samples")

# ========== 訓練 ==========
model = CNN1DClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
early_stopping = EarlyStopping(patience=patience)

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze(1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    scheduler.step()

    model.eval()
    val_losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)
            val_losses.append(loss.item())
            preds.append(torch.sigmoid(outputs).cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    val_loss = np.mean(val_losses)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    pred_labels = (preds > 0.5).astype(int)

    val_acc = accuracy_score(trues, pred_labels)
    val_precision = precision_score(trues, pred_labels, zero_division=0)
    val_recall = recall_score(trues, pred_labels, zero_division=0)
    val_f1 = f1_score(trues, pred_labels, zero_division=0)

    print(f"Epoch {epoch+1}: "
          f"Train Loss={np.mean(train_losses):.4f}, "
          f"Val Loss={val_loss:.4f}, "
          f"Val Acc={val_acc:.4f}, "
          f"Precision={val_precision:.4f}, "
          f"Recall={val_recall:.4f}, "
          f"F1={val_f1:.4f}")

    if early_stopping(val_loss, model):
        print("⚡ Early stopping triggered")
        break

print("✅ 最佳模型已儲存為 'best_cnn_model.pt'")

# ========== 測試集驗證 ==========
model.load_state_dict(torch.load("best_cnn_model.pt"))
model.eval()

test_preds = []
test_trues = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch).squeeze(1)
        preds = torch.sigmoid(outputs).cpu().numpy()
        trues = y_batch.cpu().numpy()
        test_preds.append(preds)
        test_trues.append(trues)

test_preds = np.concatenate(test_preds)
test_trues = np.concatenate(test_trues)
test_pred_labels = (test_preds > 0.5).astype(int)

test_acc = accuracy_score(test_trues, test_pred_labels)
test_precision = precision_score(test_trues, test_pred_labels, zero_division=0)
test_recall = recall_score(test_trues, test_pred_labels, zero_division=0)
test_f1 = f1_score(test_trues, test_pred_labels, zero_division=0)

print(f"🎯 Test Results on Best Model")
print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1-Score : {test_f1:.4f}")
