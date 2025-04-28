# ========== 1. 載入必要套件 ==========
import os
import glob
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as toptim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import csv
import datetime

# ========== 2. 設定隨機種子（確保結果可重現） ==========
SEED = 47
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== 3. 超參數設定 ==========
input_dim = 6                 # 輸入特徵數量（ViewX, ViewY 等6個特徵）
hidden_dim = 1024             # Transformer隱藏層大小
num_heads = 8                 # 注意力頭數
num_layers = 8                # Transformer層數
output_dim = 1                # 輸出維度（做二分類）
seq_length = 64               # 時序資料長度（64個tick）
learning_rate = 3e-4          # 初始學習率
num_epochs = 100             # 最多訓練輪數（epoch）
batch_size = 64               # 每個batch大小
patience = 20                 # 早停容忍次數（驗證集沒進步）
num_workers = 8               # DataLoader多執行緒數量
resume_from_epoch = None      # 是否從某epoch恢復訓練
weight_decay = 1e-5           # 權重衰減（L2正則化）
dropout_rate = 0.1            # dropout比例
min_lr = 1e-6                 # 最小學習率（cosine annealing最低點）
warmup_epochs = 10            # 前幾個epoch做warmup

# ========== 4. Attention Pooling 定義 ==========
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

# ========== 5. Transformer 主模型定義 ==========
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = AttentionPooling(hidden_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.pooling(x)
        x = self.mlp_head(x)
        return x

# ========== 6. Dataset類別定義 ==========
class TransferDataset(Dataset):
    def __init__(self, file_paths, seq_length=64):
        self.samples = []
        self.seq_length = seq_length

        for file in file_paths:
            df = pd.read_csv(file)
            label_path = file.replace("transfer", "labels").replace("output/transfer", "output/labels").replace("\\", "/")
            label_df = pd.read_csv(label_path)
            label_map = {(row["DemoID"], row["GroupID"]): row["player?"] for _, row in label_df.iterrows()}

            for group_id, group in df.groupby("GroupID"):
                if len(group) != self.seq_length:
                    continue
                features = group[["ViewX", "ViewY", "Total_Angular_Velocity", "Angular_Acceleration", "Fired", "Normalized_Tick"]].values
                demoid = group["DemoID"].iloc[0] if "DemoID" in group.columns else None
                label = label_map.get((demoid, group_id))
                if label is not None:
                    self.samples.append((torch.tensor(features, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ========== 7. 主程式入口點 ==========
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用裝置：{device}")

    # 抓取所有 transferN.csv 檔案
    all_transfer_files = sorted(glob.glob("output/transfer/transfer*.csv"))
    if not all_transfer_files:
        print("✅ 沒有新的 transferN.csv 要訓練")
        exit()

    # 資料集切分：7成訓練，3成驗證+測試
    train_files, temp_files = train_test_split(all_transfer_files, test_size=0.3, random_state=SEED)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=SEED)

    # 計算每個集合有效 group 數量
    def count_valid_groups(file_list):
        count = 0
        for file in file_list:
            df = pd.read_csv(file)
            count += sum(len(group) == seq_length for _, group in df.groupby("GroupID"))
        return count

    print("\n🗖️ 資料分配：")
    print(f"  • 訓練 Train Files：{len(train_files)} 個")
    print(f"  • 驗證 Val Files：{len(val_files)} 個")
    print(f"  • 測試 Test Files：{len(test_files)} 個")

    print("\n📊 標本數：")
    print(f"  • 訓練 Train Groups：{count_valid_groups(train_files)} 組")
    print(f"  • 驗證 Val Groups：{count_valid_groups(val_files)} 組")
    print(f"  • 測試 Test Groups：{count_valid_groups(test_files)} 組\n")

    # 準備 DataLoader
    train_dataset = TransferDataset(train_files)
    val_dataset = TransferDataset(val_files)
    test_dataset = TransferDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 初始化模型、損失函數、優化器
    model = TransformerModel(input_dim, output_dim, num_heads, num_layers, hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    base_optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
    optimizer = toptim.Lookahead(base_optimizer, k=5, alpha=0.5)

    # Learning rate scheduler
    def lr_lambda(current_epoch):
        return (current_epoch + 1) / warmup_epochs if current_epoch < warmup_epochs else 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(base_optimizer, lr_lambda)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=num_epochs-warmup_epochs, eta_min=min_lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("trainlog", exist_ok=True)
    metrics_csv = os.path.join("trainlog", "train_metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])

    # 是否從某個checkpoint繼續
    if resume_from_epoch is not None:
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{resume_from_epoch}.pt")
        if os.path.exists(ckpt_path):
            print(f"🔄 繼續訓練：{ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            warmup_scheduler.last_epoch = start_epoch - 1
            scheduler_cosine.last_epoch = start_epoch - 1

    # ========== 開始訓練 ==========
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss_total = 0
        train_correct, train_total = 0, 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.view(-1, seq_length, input_dim).to(device)
            y_batch = y_batch.view(-1, 1).to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

        avg_train_loss = train_loss_total / len(train_loader)
        train_acc = train_correct / train_total

        # 驗證集評估
        model.eval()
        val_loss_total = 0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.view(-1, seq_length, input_dim).to(device)
                y_batch = y_batch.view(-1, 1).to(device)
                outputs = model(X_batch)
                val_loss_total += criterion(outputs, y_batch).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        avg_val_loss = val_loss_total / len(val_loader)
        val_acc = val_correct / val_total

        # 記錄當前epoch成績
        with open(metrics_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        for param_group in base_optimizer.param_groups:
            print(f"\U0001F4C9 Current Learning Rate: {param_group['lr']:.6f}")

        # ========== 更新 Learning Rate ==========
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler_cosine.step()

        # 每10個epoch存一個checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': base_optimizer.state_dict(),
                'val_loss': avg_val_loss
            }, checkpoint_path)
            print(f"\U0001F515 已儲存 checkpoint：{checkpoint_path}")

        # 更新最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': base_optimizer.state_dict(),
                'val_loss': best_val_loss
            }, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"\U0001F31F 更新了最佳模型：Best Val Loss {best_val_loss:.6f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 提前停止條件
        if epochs_no_improve >= patience:
            print(f"\u26a1\ufe0f Early stopping triggered at epoch {epoch+1}! Best Val Loss: {best_val_loss:.6f}")
            break

        # ========== 測試最佳模型 ==========
    print("\n🧪 測試最佳模型...")
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

    # 印出 & 儲存 classification report
    print("\n📊 測試結果：")
    report_text = classification_report(all_labels, all_preds, zero_division=0)
    print(report_text)

    os.makedirs("trainlog", exist_ok=True)  # 確保 trainlog 資料夾存在

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 取得現在時間，格式：20250428-1632
    report_filename = f"trainlog/test_report_{timestamp}.txt"  # 組成檔名

    with open(report_filename, "w") as f:
        f.write(report_text)

    print(f"\n📄 測試結果也已儲存到 {report_filename}")
