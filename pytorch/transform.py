import os
import glob
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ========== 全域隨機種子設定（確保每次結果一致） ========== 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== 超參數 ========== 
input_dim = 5
hidden_dim = 128
num_heads = 4
num_layers = 2
output_dim = 1
seq_length = 64
learning_rate = 1e-4
num_epochs = 300
batch_size = 32
patience = 10

# ========== 準備資料夾 ========== 
trainlog_dir = "trainlog"
checkpoint_dir = "checkpoints"
os.makedirs(trainlog_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

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

# ========== 資料載入 function ==========
def load_data(file_list):
    X_all, y_all = [], []
    for file in file_list:
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
    return torch.tensor(np.array(X_all), dtype=torch.float32), torch.tensor(np.array(y_all), dtype=torch.float32).view(-1, 1)

# ========== 讀取 transfer 檔案 ==========
all_transfer_files = sorted(glob.glob("output/transfer/transfer*.csv"))
if not all_transfer_files:
    print("✅ 沒有新的 transferN.csv 檔案需要訓練。")
    exit()

# ========== 三分資料 train/val/test ==========
train_files, temp_files = train_test_split(all_transfer_files, test_size=0.3, random_state=SEED)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=SEED)

X_train_tensor, y_train_tensor = load_data(train_files)
X_val_tensor, y_val_tensor = load_data(val_files)
X_test_tensor, y_test_tensor = load_data(test_files)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

print(f"🚀 訓練資料量：{len(X_train_tensor)}")
print(f"🧪 驗證資料量：{len(X_val_tensor)}")
print(f"🎯 測試資料量：{len(X_test_tensor)}")

# ========== 初始化模型 ==========
model = TransformerModel(input_dim, output_dim, num_heads, num_layers, hidden_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ========== 檢查是否有 checkpoint ==========
start_epoch = 0
best_val_loss = float('inf')
best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

if os.path.exists(best_model_path):
    print(f"🔄 載入 best_model.pt 接續訓練...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['val_loss']
    print(f"✅ 從第 {start_epoch} 個 epoch 繼續")

# ========== 初始化即時繪圖 ==========
plt.ion()
fig, ax = plt.subplots()
train_losses = []
val_losses = []
epochs_record = []
epochs_no_improve = 0

# ========== 訓練 ==========
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()

    # 更新損失紀錄
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    epochs_record.append(epoch + 1)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 即時繪圖更新
    ax.clear()
    ax.plot(epochs_record, train_losses, label='Train Loss')
    ax.plot(epochs_record, val_losses, label='Val Loss')
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    plt.pause(0.01)

    # 每 10 epoch 存一版 checkpoint
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss
        }, checkpoint_path)
        print(f"💾 已儲存模型至 {checkpoint_path}")

    # 如果 val loss 有進步，更新 best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss
        }, best_model_path)
        print(f"🌟 新最佳模型！已儲存至 {best_model_path}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # early stopping 判斷
    if epochs_no_improve >= patience:
        print(f"⚡ Early stopping at epoch {epoch+1}！")
        break

# ========== 測試 ==========
print("\n🧪 測試 best model on test set...")
model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (torch.sigmoid(outputs) > 0.5).float()

print("📊 最終測試結果")
print(classification_report(y_test_tensor, predicted, zero_division=0))

plt.ioff()
plt.show()