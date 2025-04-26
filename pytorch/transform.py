import os
import glob
import pandas as pd
import numpy as np
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

# ========== 模型參數 ========== 
input_dim = 5
hidden_dim = 128
num_heads = 4
num_layers = 2
output_dim = 1
seq_length = 64
learning_rate = 1e-4  # ✅ 學習率設定
num_epochs = 250        # ✅ 訓練輪數設定（Epoch）

# ========== 準備資料夾與訓練記錄檔 ========== 
trainlog_dir = "trainlog"
os.makedirs(trainlog_dir, exist_ok=True)

trained_log = os.path.join(trainlog_dir, "trained_files.txt")
if not os.path.exists(trained_log):
    open(trained_log, "w").close()

with open(trained_log, "r") as f:
    trained_files = set(line.strip() for line in f.readlines())

# ========== Attention Pooling 定義 ========== 
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        attn_weights = self.attn(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        return (x * attn_weights).sum(dim=1)

# ========== Transformer 模型定義 ========== 
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, hidden_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers)
        self.pooling = AttentionPooling(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)
        return self.fc_out(self.pooling(x))

# ========== 收集檔案與過濾已訓練檔案 ========== 
target_players = os.getenv("TARGET_PLAYER", "dev1ce,device").split(",")
target_players = [name.strip().lower() for name in target_players]

all_transfer_files = sorted(glob.glob("output/transfer/transfer*.csv"))
selected_file_list = os.path.join(trainlog_dir, "selected_for_training.txt")
if os.path.exists(selected_file_list):
    with open(selected_file_list, "r") as f:
        all_files = [line.strip() for line in f.readlines()]
    os.remove(selected_file_list)
else:
    all_files = [f for f in all_transfer_files if f.replace("\\", "/") not in trained_files]

if not all_files:
    print("✅ 沒有新的 transferN.csv 檔案需要訓練。")
    exit()

# ========== 將檔案分為訓練與測試 ========== 
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

# ========== 讀取訓練資料並轉成 tensor ========== 
X_all, y_all = [], []
for file in train_files:
    df = pd.read_csv(file)
    label_path = file.replace("transfer", "labels").replace("output/transfer", "output/labels").replace("\\", "/")
    if not os.path.exists(label_path):
        continue
    label_df = pd.read_csv(label_path)
    label_map = {(row["DemoID"], row["GroupID"]): row["player?"] for _, row in label_df.iterrows()}
    for group_id, group in df.groupby("GroupID"):
        if len(group) != seq_length:
            continue
        features = group[["ViewX", "ViewY", "Total_Angular_Velocity", "Angular_Acceleration", "Fired"]].values
        demoid = group["DemoID"].iloc[0] if "DemoID" in group.columns else None
        label = label_map.get((demoid, group_id))
        if label is not None:
            X_all.append(features)
            y_all.append(label)

if not X_all:
    print("⚠️ 沒有足夠資料訓練")
    exit()

X = np.array(X_all)
y = np.array(y_all)

X_train_tensor = torch.tensor(X, dtype=torch.float32)
y_train_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

# ========== 模型初始化與訓練 ========== 
model = TransformerModel(input_dim, output_dim, num_heads, num_layers, hidden_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

print("🚀 訓練資料量：", len(X_all))
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # 每 10 次儲存一次模型檔案
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)
        print(f"💾 已儲存模型至 {checkpoint_path}")

# ========== 讀取測試資料並評估模型 ========== 
X_test_all, y_test_all = [], []
for file in test_files:
    df = pd.read_csv(file)
    label_path = file.replace("transfer", "labels").replace("output/transfer", "output/labels").replace("\\", "/")
    if not os.path.exists(label_path):
        continue
    label_df = pd.read_csv(label_path)
    label_map = {(row["DemoID"], row["GroupID"]): row["player?"] for _, row in label_df.iterrows()}
    for group_id, group in df.groupby("GroupID"):
        if len(group) != seq_length:
            continue
        features = group[["ViewX", "ViewY", "Total_Angular_Velocity", "Angular_Acceleration", "Fired"]].values
        demoid = group["DemoID"].iloc[0] if "DemoID" in group.columns else None
        label = label_map.get((demoid, group_id))
        if label is not None:
            X_test_all.append(features)
            y_test_all.append(label)

X_test_tensor = torch.tensor(np.array(X_test_all), dtype=torch.float32)
y_test_tensor = torch.tensor(np.array(y_test_all), dtype=torch.float32).view(-1, 1)

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (torch.sigmoid(outputs) > 0.5).float()

print("📊 測試結果")
print(classification_report(y_test_tensor, predicted, zero_division=0))