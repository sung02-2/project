# ========== 1. 載入必要套件 ==========
import os
import glob
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ========== 2. 設定全域隨機種子 ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== 3. 設定裝置(GPU或CPU) ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ 使用裝置：{device}")

# ========== 4. 超參數設定 ==========
input_dim = 5
hidden_dim = 256
num_heads = 8
num_layers = 2
output_dim = 1
seq_length = 64
learning_rate = 1e-4
num_epochs = 300
batch_size = 32
patience = 10

# ========== 5. Attention Pooling定義 ==========
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        attn_weights = self.attn(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        return (x * attn_weights).sum(dim=1)

# ========== 6. Transformer模型定義 ==========
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, hidden_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, batch_first=True)
        self.pooling = AttentionPooling(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        return self.fc_out(self.pooling(x))

# ========== 7. 自訂 Dataset ==========
class TransferDataset(Dataset):
    def __init__(self, file_paths, seq_length=64):
        self.file_paths = file_paths
        self.seq_length = seq_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file = self.file_paths[idx]
        df = pd.read_csv(file)
        label_path = file.replace("transfer", "labels").replace("output/transfer", "output/labels").replace("\\", "/")
        label_df = pd.read_csv(label_path)
        label_map = {(row["DemoID"], row["GroupID"]): row["player?"] for _, row in label_df.iterrows()}

        X_all, y_all = [], []
        for group_id, group in df.groupby("GroupID"):
            if len(group) != self.seq_length:
                continue
            features = group[["ViewX", "ViewY", "Total_Angular_Velocity", "Angular_Acceleration", "Fired"]].values
            demoid = group["DemoID"].iloc[0] if "DemoID" in group.columns else None
            label = label_map.get((demoid, group_id))
            if label is not None:
                X_all.append(features)
                y_all.append(label)

        X_tensor = torch.tensor(np.array(X_all), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_all), dtype=torch.float32).view(-1, 1)

        return X_tensor, y_tensor

# ========== 8. 資料準備 ==========
all_transfer_files = sorted(glob.glob("output/transfer/transfer*.csv"))
if not all_transfer_files:
    print("✅ 沒有新的 transferN.csv 檔案需要訓練。")
    exit()

train_files, temp_files = train_test_split(all_transfer_files, test_size=0.3, random_state=SEED)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=SEED)

train_dataset = TransferDataset(train_files)
val_dataset = TransferDataset(val_files)
test_dataset = TransferDataset(test_files)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ========== 9. 初始化模型 ==========
model = TransformerModel(input_dim, output_dim, num_heads, num_layers, hidden_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
epochs_no_improve = 0
start_epoch = 0
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
train_losses, val_losses, epochs_record = [], [], []

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss_total = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.view(-1, seq_length, input_dim).to(device)
        y_batch = y_batch.view(-1, 1).to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss_total += loss.item()

    avg_train_loss = train_loss_total / len(train_loader)

    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.view(-1, seq_length, input_dim).to(device)
            y_batch = y_batch.view(-1, 1).to(device)
            outputs = model(X_batch)
            val_loss_total += criterion(outputs, y_batch).item()

    avg_val_loss = val_loss_total / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    epochs_record.append(epoch + 1)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss
        }, os.path.join(checkpoint_dir, "best_model.pt"))
        print("🌟 新最佳模型已儲存")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"⚡ Early stopping triggered at epoch {epoch+1}!")
        break

# ========== 10. 測試 ==========
print("\n🧪 測試 best model on test set...")
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pt"))['model_state_dict'])
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.view(-1, seq_length, input_dim).to(device)
        y_batch = y_batch.view(-1, 1).to(device)
        outputs = model(X_batch)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.append(preds.cpu())
        all_labels.append(y_batch.cpu())

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
print("\n📊 測試結果:")
print(classification_report(all_labels, all_preds, zero_division=0))
