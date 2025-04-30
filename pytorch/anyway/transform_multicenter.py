# ✅ transform_4_cls_multicenter.py（多中心正式版，含自適應 threshold）

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

input_dim = 6
hidden_dim = 1024
num_heads = 8
num_layers = 4
batch_size = 64
learning_rate = 1e-4
num_epochs = 100
patience = 10

lambda_center = 0.001
acceptable_normal_acc = 0.80
penalty_weight = 10
n_centers = 5

# ========== Dataset ========== #
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, dtype=np.float32)
        self.features = df.iloc[:, :-1].values.reshape(-1, 64, 6)
        self.labels = df.iloc[:, -1].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# ========== Transformer Feature Extractor ========== #
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x):
        x = self.embedding(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        return x[:, 0, :]

# ========== Adaptive MultiCenter Loss ========== #
class AdaptiveMultiCenterLoss(nn.Module):
    def __init__(self, feature_dim, n_centers=5, lambda_center=0.001):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_centers, feature_dim))
        self.thresholds = nn.Parameter(torch.ones(n_centers))
        self.lambda_center = lambda_center

    def forward(self, features, labels):
        dists = torch.cdist(features.unsqueeze(1), self.centers.unsqueeze(0))
        min_dists, min_indices = dists.min(dim=1)
        chosen_thresholds = self.thresholds[min_indices]
        preds = (min_dists > chosen_thresholds).float()

        normal_mask = labels == 1
        if normal_mask.sum() > 0:
            normal_acc = (preds[normal_mask] == 0).sum() / normal_mask.sum()
            if normal_acc < acceptable_normal_acc:
                penalty = penalty_weight * (acceptable_normal_acc - normal_acc)
            else:
                penalty = 0.0
        else:
            penalty = 0.0

        base_loss = F.relu(min_dists - chosen_thresholds) ** 2
        loss = base_loss.mean() + self.lambda_center * (self.centers.norm(dim=1).mean() + self.thresholds.mean()) + penalty

        self.dynamic_adjust(features, min_dists, min_indices)

        return loss

    def dynamic_adjust(self, features, min_dists, min_indices):
        far_samples = min_dists > (self.thresholds[min_indices] * 2)

        if far_samples.sum() > 0:
            for idx in min_indices[far_samples]:
               self.thresholds.data[idx] += 0.01
               self.thresholds.data.clamp_(0.5, 5.0)

        if far_samples.sum() > 5 and self.centers.size(0) < 10:
            with torch.no_grad():
                new_feature = features[far_samples][0].detach().clone()
                new_center = new_feature.unsqueeze(0)
                new_threshold = torch.tensor([1.0], device=new_center.device)
                self.centers = nn.Parameter(torch.cat([self.centers.data, new_center], dim=0))
                self.thresholds = nn.Parameter(torch.cat([self.thresholds.data, new_threshold], dim=0))


    def predict(self, features):
        dists = torch.cdist(features.unsqueeze(1), self.centers.unsqueeze(0))
        min_dists, min_indices = dists.min(dim=1)
        chosen_thresholds = self.thresholds[min_indices]
        preds = (min_dists > chosen_thresholds).float()
        return preds


# ========== 驗證 ========== #
def validate(model, loss_fn, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            features = model(X_batch)
            preds = loss_fn.predict(features)
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    precision = precision_score(all_labels.int(), all_preds.int(), zero_division=0)
    recall = recall_score(all_labels.int(), all_preds.int(), zero_division=0)
    f1 = f1_score(all_labels.int(), all_preds.int(), zero_division=0)

    return precision, recall, f1

# ========== 訓練 ========== #
def train(model, loss_fn, optimizer, train_loader, val_loader, device, num_epochs, patience):
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
        precision, recall, f1 = validate(model, loss_fn, val_loader, device)

        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("✅ 儲存新的最佳模型")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("🛑 早停！")
                break

# ========== 主程式 ========== #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CustomDataset("../output/split/train_set.csv")
    val_dataset = CustomDataset("../output/split/val_set.csv")
    test_dataset = CustomDataset("../output/split/test_set.csv")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerFeatureExtractor(input_dim, num_heads, num_layers, hidden_dim).to(device)
    loss_fn = AdaptiveMultiCenterLoss(hidden_dim, n_centers=n_centers, lambda_center=lambda_center).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=learning_rate)

    train(model, loss_fn, optimizer, train_loader, val_loader, device, num_epochs, patience)

    model.load_state_dict(torch.load("best_model.pt"))

    print("\n🧪 測試集異常率分析...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    precision, recall, f1 = validate(model, loss_fn, test_loader, device)
    print(f"Test Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
