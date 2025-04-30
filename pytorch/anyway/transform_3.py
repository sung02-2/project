# ✅ transform_3_cls_v3.py（正式版，包含 acceptable normal acc + 懲罰機制）

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

# ========== 超參數設定 ========== #
SEED = 47
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 特徵數量、Transformer架構設定
input_dim = 6
hidden_dim = 1024
num_heads = 8
num_layers = 4
batch_size = 64

# 優化器、訓練設定
learning_rate = 1e-4
num_epochs = 100
patience = 10

# 損失函數設定
lambda_center = 0.001
acceptable_normal_acc = 0.80   # 正常玩家正常率最低要求
penalty_weight = 10            # 正常率不夠時的懲罰加權

# ========== 測試集異常率分析 ========== #
def test_abnormal_ratio(model, center, threshold, test_csv, device):
    """在測試集上統計正常玩家與異常玩家的異常判定率"""
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

    # 正常玩家 vs 非正常玩家分開統計
    normal_mask = all_labels == 1
    abnormal_mask = all_labels == 0

    normal_abnormal_rate = all_preds[normal_mask].sum().item() / max(normal_mask.sum().item(), 1)
    abnormal_abnormal_rate = all_preds[abnormal_mask].sum().item() / max(abnormal_mask.sum().item(), 1)
    normal_correct_rate = (all_preds[normal_mask] == 0).sum().item() / max(normal_mask.sum().item(), 1)

    # 顯示結果
    print(f"\n📊 測試集中：")
    print(f"• 正常玩家異常率：{normal_abnormal_rate:.2%}")
    print(f"• 正常玩家正常率：{normal_correct_rate:.2%}")
    print(f"• 其他玩家異常率：{abnormal_abnormal_rate:.2%}")
    print(f"→ 差距：{(abnormal_abnormal_rate - normal_abnormal_rate):.2%}")

# ========== Transformer 特徵萃取器 (加CLS Token) ========== #
class TransformerFeatureExtractor(nn.Module):
    """將輸入序列經過 Transformer 萃取，取 CLS token 作為特徵"""
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # 初始化一個學習到的CLS token

    def forward(self, x):
        x = self.embedding(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # CLS放到最前面
        x = self.transformer(x)
        return x[:, 0, :]  # 取CLS token對應的輸出

# ========== 自適應 Center Loss（含正常率要求） ========== #
class AdaptiveCenterLoss(nn.Module):
    def __init__(self, feature_dim, lambda_center=0.001, margin=0.7, penalty_abnormal_weight=30, fixed_threshold=1.0):
        super().__init__()
        self.center = nn.Parameter(torch.randn(feature_dim))
        self.lambda_center = lambda_center
        self.margin = margin
        self.penalty_abnormal_weight = penalty_abnormal_weight
        self.threshold = fixed_threshold  # 固定 threshold（不是 nn.Parameter）

    def forward(self, features, labels):
        dists = torch.norm(features - self.center, dim=1)
        preds = (dists > self.threshold).float()

        normal_mask = labels == 1
        abnormal_mask = labels == 0
        normal_total = normal_mask.sum().item()
        abnormal_total = abnormal_mask.sum().item()

        # 1. 正常率計算
        if normal_total > 0:
            normal_correct = (preds[normal_mask] == 0).sum().item()
            normal_acc = normal_correct / normal_total
        else:
            normal_acc = 1.0

        # 2. 基本 Loss（越過 threshold 的才有懲罰）
        base_loss = F.relu(dists - self.threshold) ** 2
        center_loss = base_loss.mean() + self.lambda_center * self.threshold

        # 3. 如果正常率不夠，加懲罰
        if normal_acc < acceptable_normal_acc:
            penalty = penalty_weight * (acceptable_normal_acc - normal_acc)
            center_loss += penalty

        # 4. 如果異常樣本離 center 太近，加重懲罰
        if abnormal_total > 0:
            abnormal_close = (dists[abnormal_mask] < (self.threshold - self.margin)).float().mean()
            center_loss += self.penalty_abnormal_weight * abnormal_close

        return center_loss


# ========== 自訂 Dataset 讀取器 ========== #
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, dtype=np.float32)
        self.features = df.iloc[:, :-1].values.reshape(-1, 64, 6)
        self.labels = df.iloc[:, -1].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# ========== 驗證流程 ========== #
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

# ========== 訓練流程 ========== #
def train(model, loss_fn, optimizer, train_loader, val_loader, device, num_epochs, early_stopping_patience=10):
    best_f1 = -1
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            features = model(X_batch)
            loss = loss_fn(features, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        precision, recall, f1, normal_accuracy, normal_abnormal_rate, abnormal_abnormal_rate, gap = validate(model, loss_fn.center, loss_fn.threshold, val_loader, device)

        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")
        print(f"Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}, Normal Acc: {normal_accuracy:.4f}")
        print(f"• 正常玩家異常率：{normal_abnormal_rate:.2%} | 其他玩家異常率：{abnormal_abnormal_rate:.2%} | 差距：{gap:.2%}")
        print(f"Threshold: {loss_fn.threshold:.4f}")

        # 閥值調整：目標是讓正常玩家異常率保持合理
        # 閥值調整

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

# ========== 主程式 ========== #
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = CustomDataset("../output/split/train_set.csv")
    val_dataset = CustomDataset("../output/split/val_set.csv")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = CustomDataset("../output/split/test_set.csv")

    print("\n📦 資料量統計")
    print("===================")
    print(f"• 訓練資料總筆數：{len(train_dataset)}")
    print(f"• 驗證資料總筆數：{len(val_dataset)}")
    print(f"• 測試資料總筆數：{len(test_dataset)}")
    print("===================\n")

    model = TransformerFeatureExtractor(input_dim, num_heads, num_layers, hidden_dim).to(device)
    loss_fn = AdaptiveCenterLoss(
    feature_dim=hidden_dim,
    lambda_center=lambda_center,
    margin=0.7,
    penalty_abnormal_weight=30,
    fixed_threshold=1.0  # ⭐這裡設定你的固定 threshold 值
    ).to(device)

    optimizer = optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=learning_rate)

    train(model, loss_fn, optimizer, train_loader, val_loader, device, num_epochs=num_epochs, early_stopping_patience=patience)

    print("\n🧪 開始測試集分析...")
    test_abnormal_ratio(model, loss_fn.center, loss_fn.threshold, "../output/split/test_set.csv", device)
