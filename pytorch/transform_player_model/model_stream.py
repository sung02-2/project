# ===== 訓練與紀錄：train_autoencoder.py =====
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
from player_style_autoencoder import PlayerStyleTransformerAutoEncoder

# ===== 可調整超參數 =====
SEED = 47
batch_size = 64
num_epochs = 50
learning_rate = 1e-3
output_dir = "checkpoints"

class TransferPTDataset(Dataset):
    def __init__(self, data_tensor):
        self.x_seq = data_tensor['x_seq']
        self.x_group = data_tensor['x_group']

    def __len__(self):
        return len(self.x_seq)

    def __getitem__(self, idx):
        return self.x_seq[idx], self.x_group[idx]

if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ 使用裝置：{device}")

    train_data = torch.load("../../output/pt/train.pt")
    dataset = TransferPTDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PlayerStyleTransformerAutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    metrics_log = os.path.join(output_dir, "autoencoder_metrics.csv")
    with open(metrics_log, 'w', newline='') as f:
        csv.writer(f).writerow(["Epoch", "Loss"])

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for x_seq, x_group in dataloader:
            x_seq, x_group = x_seq.to(device), x_group.to(device)
            recon_seq, recon_group = model(x_seq, x_group)

            loss_seq = criterion(recon_seq, x_seq.mean(dim=1))
            loss_group = criterion(recon_group, x_group)
            loss = loss_seq + loss_group

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_seq.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        with open(metrics_log, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_loss])

    torch.save(model.state_dict(), os.path.join(output_dir, "autoencoder_transformer.pt"))
    print("✅ 模型已儲存")