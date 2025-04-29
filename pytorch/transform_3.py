# ✅ transform_3.py（改進版，加上註解）

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
import random

# ========== 超參數 ==========
SEED = 47
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

input_dim = 6
hidden_dim = 1024
num_heads = 8
num_layers = 4
batch_size = 64
learning_rate = 1e-4
num_epochs = 100
patience = 10
lambda_center = 0.001

# ========== 測試集異常率分析 ==========
def test_abnormal_ratio(model, center, threshold, test_csv, device):
    model.eval()
    test_dataset = CustomDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            features = model(X_batch)
            distances = torch.norm(features - center, dim=1)
            preds = (distances > threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    normal_mask = all_labels == 1
    abnormal_mask = all_labels == 0

    normal_abnormal_rate = preds[normal_mask].sum().item() / max(normal_mask.sum().item(), 1)
    abnormal_abnormal_rate = preds[abnormal_mask].sum().item() / max(abnormal_mask.sum().item(), 1)
    normal_correct_rate = (preds[normal_mask] == 0).sum().item() / max(normal_mask.sum().item(), 1)

    print(f"\n📊 測試集中：")
    print(f"• 正常玩家異常率：{normal_abnormal_rate:.2%}")
    print(f"• 正常玩家正常率：{normal_correct_rate:.2%}")
    print(f"• 其他玩家異常率：{abnormal_abnormal_rate:.2%}")
    print(f"→ 差距：{(abnormal_abnormal_rate - normal_abnormal_rate):.2%}")
# ========== Attention Pooling 層 ==========
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=1)
        return (x * attn_weights).sum(dim=1)

# ========== Transformer 特徵萃取器 ==========
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = AttentionPooling(hidden_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.pooling(x)
        return x

# ========== 自適應 Center Loss ==========
class AdaptiveCenterLoss(nn.Module):
    def __init__(self, feature_dim, lambda_center=0.001):
        super().__init__()
        self.center = nn.Parameter(torch.randn(feature_dim))
        self.threshold = nn.Parameter(torch.tensor(1.0))
        self.lambda_center = lambda_center

    def forward(self, features):
        dists = torch.norm(features - self.center, dim=1)
        losses = F.relu(dists - self.threshold) ** 2
        return losses.mean() + self.lambda_center * self.threshold

# ========== 自訂 Dataset ==========
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, dtype=np.float32)  # 強制讀成 float32
        self.features = df.iloc[:, :-1].values.reshape(-1, 64, 6)
        self.labels = df.iloc[:, -1].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# ========== 驗證 ==========
def validate(model, center, threshold, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            features = model(X_batch)
            distances = torch.norm(features - center, dim=1)
            preds = (distances > threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    normal_mask = all_labels == 1
    abnormal_mask = all_labels == 0

    normal_total = normal_mask.sum().item()
    abnormal_total = abnormal_mask.sum().item()

    normal_accuracy = (all_preds[normal_mask] == 0).sum().item() / max(normal_total, 1)
    normal_abnormal_rate = (all_preds[normal_mask] == 1).sum().item() / max(normal_total, 1)
    abnormal_abnormal_rate = (all_preds[abnormal_mask] == 1).sum().item() / max(abnormal_total, 1)
    gap = abnormal_abnormal_rate - normal_abnormal_rate

    return precision, recall, f1, normal_accuracy, normal_abnormal_rate, abnormal_abnormal_rate, gap

# ========== 訓練流程 ==========
def train(model, loss_fn, optimizer, train_loader, val_loader, device, num_epochs, early_stopping_patience=10):
    best_f1 = -1
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            features = model(X_batch)
            loss = loss_fn(features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ======= 做一次驗證 =======
        precision, recall, f1, normal_accuracy, normal_abnormal_rate, abnormal_abnormal_rate, gap = validate(model, loss_fn.center, loss_fn.threshold, val_loader, device)

        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")
        print(f"Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}, Normal Acc: {normal_accuracy:.4f}")
        print(f"• 正常玩家異常率：{normal_abnormal_rate:.2%} | 其他玩家異常率：{abnormal_abnormal_rate:.2%} | 差距：{gap:.2%}")
        print(f"Threshold: {loss_fn.threshold.item():.4f}")

        # ======= (新增) 自動調整 threshold =======
        with torch.no_grad():
            if normal_abnormal_rate > 0.05:
                # 如果正常玩家異常率太高 (>5%)，提高 threshold 讓判定更嚴格
                loss_fn.threshold += 0.01
            elif normal_abnormal_rate < 0.01:
                # 如果正常玩家異常率太低 (<1%)，降低 threshold 讓異常容易被抓到
                loss_fn.threshold -= 0.005

            # 限制 threshold 合理範圍
            loss_fn.threshold.data.clamp_(0.5, 2.0)

        # ======= (原本就有) 保存最佳模型 =======
        if f1 > best_f1:
            best_f1 = f1
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("✅ 儲存新的最佳模型！")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"🛑 早停：Val F1沒進步{early_stopping_patience}個epoch")
                break


# ========== 主程式 ==========
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = CustomDataset("../output/split/train_set.csv")
    val_dataset = CustomDataset("../output/split/val_set.csv")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = CustomDataset("../output/split/test_set.csv")  # 加這行，把測試集讀進來

    print("\n📦 資料量統計")
    print("===================")
    print(f"• 訓練資料總筆數：{len(train_dataset)}")
    print(f"    - 正常玩家 (label=1)：{(train_dataset.labels == 1).sum()} 筆")
    print(f"    - 異常玩家 (label=0)：{(train_dataset.labels == 0).sum()} 筆")
    print(f"• 驗證資料總筆數：{len(val_dataset)}")
    print(f"    - 正常玩家 (label=1)：{(val_dataset.labels == 1).sum()} 筆")
    print(f"    - 異常玩家 (label=0)：{(val_dataset.labels == 0).sum()} 筆")
    print(f"• 測試資料總筆數：{len(test_dataset)}")
    print(f"    - 正常玩家 (label=1)：{(test_dataset.labels == 1).sum()} 筆")
    print(f"    - 異常玩家 (label=0)：{(test_dataset.labels == 0).sum()} 筆")
    print("===================\n")
    model = TransformerFeatureExtractor(input_dim, num_heads, num_layers, hidden_dim).to(device)
    loss_fn = AdaptiveCenterLoss(hidden_dim, lambda_center=lambda_center).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=learning_rate)

    train(model, loss_fn, optimizer, train_loader, val_loader, device, num_epochs=num_epochs, early_stopping_patience=patience)


    print("\n🧪 開始測試集分析...")
    test_abnormal_ratio(model, loss_fn.center, loss_fn.threshold, "../output/split/test_set.csv", device)
