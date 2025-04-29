# ========== 1. 載入必要套件 ==========
import os
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import csv
import datetime

# ========== 2. 設定隨機種子 ==========
SEED = 47
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== 3. 超參數設定 ==========
input_dim = 6
hidden_dim = 256
num_heads = 8
num_layers = 4
seq_length = 64
learning_rate = 1e-4
num_epochs = 100
batch_size = 64
patience = 15
num_workers = 8
resume_from_epoch = None
weight_decay = 1e-5
dropout_rate = 0.1
min_lr = 1e-6
warmup_epochs = 10
margin = 0.5
feature_bank_path = "feature_bank.pt"

# ========== 4. Attention Pooling ==========
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

# ========== 5. Transformer 特徵提取器 ==========
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = AttentionPooling(hidden_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.pooling(x)
        return x

# ========== 6. Dataset 類別 ==========
class TransferDataset(Dataset):
    def __init__(self, transfer_csv, label_csv, selected_group_ids, seq_length=64):
        self.samples = []
        self.seq_length = seq_length

        df = pd.read_csv(transfer_csv)
        label_df = pd.read_csv(label_csv)
        label_map = {(row["DemoID"], row["GroupID"]): row["player?"] for _, row in label_df.iterrows()}

        grouped = df.groupby(["DemoID", "GroupID"])
        for (demoid, group_id), group in grouped:
            if len(group) != self.seq_length:
                continue
            if (demoid, group_id) not in selected_group_ids:
                continue
            features = group[["ViewX", "ViewY", "Total_Angular_Velocity", "Angular_Acceleration", "Fired", "Normalized_Tick"]].values
            label = label_map.get((demoid, group_id))
            if label is not None:
                self.samples.append((torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ========== 7. 主程式入口點 ==========
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用裝置：{device}")

    transfer_csv = "../output/transfer_all.csv"
    label_csv = "../output/label_all.csv"

    df = pd.read_csv(transfer_csv)
    label_df = pd.read_csv(label_csv)

    valid_groups = []
    grouped = df.groupby(["DemoID", "GroupID"])
    for (demoid, group_id), group in grouped:
        if len(group) == seq_length:
            valid_groups.append((demoid, group_id))

    train_groups, temp_groups = train_test_split(valid_groups, test_size=0.3, random_state=SEED)
    val_groups, test_groups = train_test_split(temp_groups, test_size=0.5, random_state=SEED)

    train_dataset = TransferDataset(transfer_csv, label_csv, train_groups)
    test_dataset = TransferDataset(transfer_csv, label_csv, test_groups)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = TransformerFeatureExtractor(input_dim, num_heads, num_layers, hidden_dim, dropout_rate).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    # ========== 開始訓練 (Triplet Loss) ==========
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            features = model(X_batch)

            anchors, positives, negatives = [], [], []
            for i in range(len(y_batch)):
                anchor_feature = features[i]
                anchor_label = y_batch[i]

                pos_indices = (y_batch == anchor_label).nonzero(as_tuple=True)[0]
                pos_indices = pos_indices[pos_indices != i]

                if len(pos_indices) == 0:
                    continue
                positive_feature = features[random.choice(pos_indices)]

                neg_indices = (y_batch != anchor_label).nonzero(as_tuple=True)[0]
                if len(neg_indices) == 0:
                    continue
                negative_feature = features[random.choice(neg_indices)]

                anchors.append(anchor_feature)
                positives.append(positive_feature)
                negatives.append(negative_feature)

            if anchors and positives and negatives:
                anchors = torch.stack(anchors)
                positives = torch.stack(positives)
                negatives = torch.stack(negatives)
                loss = triplet_loss_fn(anchors, positives, negatives)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            with torch.no_grad():
                for f, true_label in zip(features, y_batch):
                    sims = F.cosine_similarity(f.unsqueeze(0), features)
                    max_sim, idx = sims.max(0)
                    pred_label = y_batch[idx]
                    if pred_label.item() == true_label.item():
                        correct += 1
                    total += 1

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Triplet Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # ========== 儲存 Feature Bank ==========
    print("\n✨ 建立並儲存 Feature Bank...")
    model.eval()
    feature_bank = []
    labels_bank = []
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            features = model(X_batch)
            feature_bank.append(features.cpu())
            labels_bank.append(y_batch.cpu())

    feature_bank = torch.cat(feature_bank)
    labels_bank = torch.cat(labels_bank)
    torch.save({'features': feature_bank, 'labels': labels_bank}, feature_bank_path)
    print(f"✅ Feature Bank 已儲存到 {feature_bank_path}")

    # ========== 測試 ==========
    print("\n🧪 測試資料集...")
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            features = model(X_batch)
            bank = torch.load(feature_bank_path)
            bank_features = bank['features']
            bank_labels = bank['labels']

            for f, true_label in zip(features, y_batch):
                sims = F.cosine_similarity(f.unsqueeze(0), bank_features)
                max_sim, idx = sims.max(0)
                pred_label = bank_labels[idx]
                if pred_label.item() == true_label.item():
                    correct += 1
                total += 1

    acc = correct / total
    print(f"\n🎯 Test Accuracy (with Feature Bank matching): {acc:.4f}")
