import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import csv

from model_transformer_F1 import TransformerModelWithGroup


# ===== 全域設定 =====
SEED = 47
random.seed(SEED)
torch.manual_seed(SEED)

# ===== 超參數設定 =====
seq_input_dim = 18
group_input_dim = 5
hidden_dim = 1024
num_heads = 8
num_layers = 4
output_dim = 1
learning_rate = 1e-5   # 建議更保守
num_epochs = 100
batch_size = 64
patience = 10
min_lr = 1e-6
warmup_epochs = 5
checkpoint_dir = "checkpoints"
log_dir = "trainlog"
sigmoid_csv_path = os.path.join(log_dir, "sigmoid_outputs.csv")


# ===== SoftF1 Loss =====
class SoftF1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        soft_f1 = 2 * tp / (2 * tp + fp + fn + self.epsilon)
        return 1 - soft_f1


# ========== 自動 threshold 搜尋 ==========
def find_best_threshold(y_true, probs, thresholds=None):
    if thresholds is None:
        thresholds = [i / 100 for i in range(10, 90, 5)]  # 0.10 ~ 0.85
    best_f1 = -1
    best_th = 0.5
    for th in thresholds:
        pred = (probs > th).float()
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1


# ===== Dataset =====
class TransferPTDataset(Dataset):
    def __init__(self, data_tensor):
        self.x_seq = data_tensor['x_seq']
        self.x_group = data_tensor['x_group']
        self.y = data_tensor['y']

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

    model = TransformerModelWithGroup(seq_input_dim, group_input_dim, output_dim,
                                      num_heads, num_layers, hidden_dim, dropout_rate=0).to(device)
    criterion = SoftF1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    def lr_lambda(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=min_lr)

    best_val_f1 = -1
    best_threshold = 0.5
    no_improve_count = 0
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
            logits = torch.clamp(logits, -10, 10)  # ✅ 限制範圍
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += loss.item() * x_seq.size(0)
            total_train_samples += x_seq.size(0)

        model.eval()
        val_loss_total = 0
        val_probs, val_labels = [], []
        with torch.no_grad():
            for x_seq, x_group, y in val_loader:
                x_seq, x_group, y = x_seq.to(device), x_group.to(device), y.to(device)
                logits = model(x_seq, x_group)
                logits = torch.clamp(logits, -10, 10)
                loss = criterion(logits, y)
                probs = torch.sigmoid(logits)
                val_loss_total += loss.item() * x_seq.size(0)
                val_probs.append(probs.cpu())
                val_labels.append(y.cpu())

        probs_all = torch.cat(val_probs)
        y_true = torch.cat(val_labels)
        best_threshold, best_f1 = find_best_threshold(y_true, probs_all)
        y_pred = (probs_all > best_threshold).float()

        avg_train_loss = train_loss_total / total_train_samples
        avg_val_loss = val_loss_total / len(val_set)
        val_acc = (y_true == y_pred).float().mean().item()
        val_precision = precision_score(y_true, y_pred, zero_division=0)
        val_recall = recall_score(y_true, y_pred, zero_division=0)

        print(f"\n🎯 最佳門檻：{best_threshold:.2f}")
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {best_f1:.4f}")

        with open(metrics_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, avg_val_loss, val_acc, val_precision, val_recall, best_f1])

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            no_improve_count = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
            print("🌟 儲存最佳模型（F1 提升）")
        else:
            no_improve_count += 1
            print(f"⚠️ 連續 {no_improve_count} 次未提升 F1")
        if no_improve_count >= patience:
            print(f"🛑 Early stopping! 已連續 {patience} 次未提升 F1")
            break

    # 測試階段
    print("\n🧪 測試最佳模型...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pt")))
    model.eval()
    test_preds, test_labels, test_probs = [], [], []
    with torch.no_grad():
        for x_seq, x_group, y in test_loader:
            x_seq, x_group, y = x_seq.to(device), x_group.to(device), y.to(device)
            logits = model(x_seq, x_group)
            logits = torch.clamp(logits, -10, 10)
            probs = torch.sigmoid(logits)
            pred = (probs > best_threshold).float()
            test_preds.append(pred.cpu())
            test_labels.append(y.cpu())
            test_probs.append(probs.cpu())

    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)
    test_probs = torch.cat(test_probs)
    print(f"\n📋 測試結果 (使用最佳門檻 {best_threshold:.2f})")
    print(classification_report(test_labels, test_preds, zero_division=0))

    # 儲存 sigmoid 分數
    with open(sigmoid_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "SigmoidScore"])
        for label, prob in zip(test_labels.tolist(), test_probs.tolist()):
            writer.writerow([label[0], prob[0]])
