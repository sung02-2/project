import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as toptim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import csv

from model_transformer import TransformerModelWithGroup

# ===== 全域設定 =====
SEED = 47
random.seed(SEED)
torch.manual_seed(SEED)

# ===== 超參數設定 =====
seq_input_dim = 18
group_input_dim = 5
hidden_dim = 512
num_heads = 8
num_layers = 4
output_dim = 1
learning_rate = 1e-4
num_epochs = 100
batch_size = 64
patience = 15
min_lr = 1e-6
warmup_epochs = 10
checkpoint_dir = "checkpoints"
log_dir = "trainlog"
THRESHOLD = 0.5  # ✅ 可調整門檻值

class TransferPTDataset(Dataset):
    def __init__(self, data_tensor):
        self.x_seq = data_tensor['x_seq']
        self.x_group = data_tensor['x_group']
        self.y = data_tensor['y']
        self.group_ids = data_tensor.get('group_ids', [None] * len(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_seq[idx], self.x_group[idx], self.y[idx]

if __name__ == "__main__":
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用裝置：{device}")

    train_data = torch.load("../../output/pt/train.pt")
    val_data = torch.load("../../output/pt/val.pt")
    test_data = torch.load("../../output/pt/test.pt")

    train_set = TransferPTDataset(train_data)
    val_set = TransferPTDataset(val_data)
    test_set = TransferPTDataset(test_data)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # ===== 加入 pos_weight（處理不平衡） =====
    y_train_tensor = train_data['y']
    num_pos = (y_train_tensor == 1).sum().item()
    num_neg = (y_train_tensor == 0).sum().item()
    pos_weight_value = num_neg / max(num_pos, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    print(f"⚖️ pos_weight 設定為：{pos_weight.item():.4f} (neg/pos = {num_neg}/{num_pos})")

    model = TransformerModelWithGroup(seq_input_dim, group_input_dim, output_dim, num_heads, num_layers, hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    base_optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = toptim.Lookahead(base_optimizer)

    def lr_lambda(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(base_optimizer, lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=num_epochs - warmup_epochs, eta_min=min_lr)

    best_val_f1 = -1  # ✅ 依據 val_f1 儲存最佳模型
    metrics_csv = os.path.join(log_dir, "train_metrics.csv")
    with open(metrics_csv, 'w', newline='') as f:
        csv.writer(f).writerow(["Epoch", "Train Loss", "Val Loss", "Val Acc", "Val Precision", "Val Recall", "Val F1"])

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0
        total_train_samples = 0

        for x_seq, x_group, y in train_loader:
            x_seq, x_group, y = x_seq.to(device), x_group.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_seq, x_group)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * x_seq.size(0)
            total_train_samples += x_seq.size(0)

        model.eval()
        val_loss_total = 0
        total_val_samples = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x_seq, x_group, y in val_loader:
                x_seq, x_group, y = x_seq.to(device), x_group.to(device), y.to(device)
                logits = model(x_seq, x_group)
                val_loss_total += criterion(logits, y).item() * x_seq.size(0)
                total_val_samples += x_seq.size(0)
                probs = torch.sigmoid(logits)
                pred = (probs > THRESHOLD).float()
                val_preds.append(pred.cpu())
                val_labels.append(y.cpu())

        avg_train_loss = train_loss_total / total_train_samples
        avg_val_loss = val_loss_total / total_val_samples

        y_true = torch.cat(val_labels)
        y_pred = torch.cat(val_preds)
        val_acc = (y_true == y_pred).float().mean().item()
        val_precision = precision_score(y_true, y_pred, zero_division=0)
        val_recall = recall_score(y_true, y_pred, zero_division=0)
        val_f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n🎯 使用預測門檻：{THRESHOLD}")
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print("📊 Sigmoid 機率範圍：", probs.min().item(), "~", probs.max().item())
        print("🔍 y_pred 分布：", torch.unique(y_pred, return_counts=True))
        print("🔍 y_true 分布：", torch.unique(y_true, return_counts=True))
        print(classification_report(y_true, y_pred, zero_division=0))

        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, val_acc, val_precision, val_recall, val_f1])

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
            print("🌟 儲存最佳模型（F1 提升）")

    print("\n🧪 測試最佳模型...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pt")))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x_seq, x_group, y in test_loader:
            x_seq, x_group, y = x_seq.to(device), x_group.to(device), y.to(device)
            logits = model(x_seq, x_group)
            pred = (torch.sigmoid(logits) > THRESHOLD).float()
            test_preds.append(pred.cpu())
            test_labels.append(y.cpu())

    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)
    print("\n📋 測試結果 (threshold =", THRESHOLD, "):")
    print(classification_report(test_labels, test_preds, zero_division=0))