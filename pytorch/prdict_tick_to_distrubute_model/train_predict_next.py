import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ========== 參數區塊（可調整超參數與檔案路徑） ==========
INPUT_DIM = 18              # 每個 tick 的特徵維度
HIDDEN_DIM = 256          # Transformer 的隱藏層維度
NUM_HEADS = 4               # 多頭注意力的頭數
NUM_LAYERS = 4              # Transformer 的層數
DROPOUT = 0.1               # dropout 機率
d_model = HIDDEN_DIM        # d_model 參數
LEARNING_RATE = 5e-5        # 初始學習率（會動態調整）
NUM_EPOCHS = 100            # 最多訓練 epoch 數
BATCH_SIZE = 64             # 每批次樣本數
PATIENCE = 100               # early stopping 容忍次數
WARMUP_STEPS = 4000         # step-wise warmup 步數（不是 epoch）
TRAIN_PATH = "../../output/pt/train.pt"      # 訓練資料路徑
VAL_PATH = "../../output/pt/val.pt"          # 驗證資料路徑
MODEL_SAVE_PATH = "encoder_decoder.pt"       # 儲存模型路徑
ERROR_CSV_PATH = "tickwise_prediction_error.csv"  # 輸出誤差記錄

# ===== Positional Encoding（時間序列用） =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

# ===== 模型定義：Encoder-Decoder 架構預測 tick-to-tick 變化 =====
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout, use_pos_encoding=True):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.predictor = nn.Linear(hidden_dim, input_dim)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        if self.use_pos_encoding:
            src_emb = self.pos_encoder(src_emb)
            tgt_emb = self.pos_encoder(tgt_emb)
        memory = self.encoder(src_emb)
        out = self.decoder(tgt_emb, memory)
        return self.predictor(out)

# ===== 分類模型：根據 tick-wise loss 向量預測是否為主角風格 =====
class TickwiseClassifier(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(1)

# ===== 自訂 Dataset：將 [0~62] 做為 src，預測 [1~63] 做為 tgt =====
class SeqToSeqTickDataset(Dataset):
    def __init__(self, data_tensor):
        self.src = data_tensor['x_seq'][:, :-1, :]
        self.tgt = data_tensor['x_seq'][:, 1:, :]
        self.labels = data_tensor['y'].squeeze()  # 加入 .squeeze() 確保為一維 tensor

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.labels[idx]

# ===== 預測誤差（tick-wise loss vector） =====
def evaluate_tickwise_loss(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=1)
    loss_vectors = []
    labels = []
    criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        for src, tgt, label in loader:
            src, tgt = src.to(device), tgt.to(device)
            pred = model(src, tgt)
            loss = criterion(pred, tgt).mean(dim=2)  # 每 tick 的 loss 向量
            loss_vectors.append(loss.squeeze(0).cpu())
            labels.append(label.item())
    return torch.stack(loss_vectors), torch.tensor(labels)

# ===== 主訓練與驗證流程 =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用裝置：{device}")

    train_data = torch.load(TRAIN_PATH)
    val_data = torch.load(VAL_PATH)

    train_dataset = SeqToSeqTickDataset(train_data)
    val_dataset = SeqToSeqTickDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EncoderDecoderTransformer(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_pos_encoding=True
    ).to(device)

    clf = TickwiseClassifier(seq_len=63).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=LEARNING_RATE)

    # Warmup + Cosine Annealing Scheduler (step-wise)
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = WARMUP_STEPS

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss(reduction='none')
    bce_loss = nn.BCELoss()
    best_f1 = -1
    patience_counter = 0
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        clf.train()
        total_loss = 0

        for src, tgt, label in train_loader:
            src, tgt, label = src.to(device), tgt.to(device), label.to(device).float().squeeze()
            pred = model(src, tgt)
            tick_loss = criterion(pred, tgt).mean(dim=2)
            pred_logits = clf(tick_loss)
            loss = bce_loss(pred_logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item() * src.size(0)

        avg_train_loss = total_loss / len(train_dataset)

        model.eval()
        clf.eval()
        val_total_loss = 0
        sigmoid_outputs = []
        with torch.no_grad():
            val_loss_vecs, val_labels = evaluate_tickwise_loss(model, val_dataset, device)
            pred_probs = clf(val_loss_vecs.to(device))
            preds = pred_probs > 0.5

            val_loss = bce_loss(pred_probs, val_labels.float().to(device)).item()
            sigmoid_outputs.extend(pred_probs.cpu().numpy())

            acc = accuracy_score(val_labels.numpy(), preds.cpu().numpy())
            f1 = f1_score(val_labels.numpy(), preds.cpu().numpy())

            num_pred_player = preds.sum().item()
            num_pred_nonplayer = len(preds) - num_pred_player

        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  - Train Loss: {avg_train_loss:.6f}")
        print(f"  - Val Loss: {val_loss:.6f}")
        print(f"  - Val ACC: {acc:.4f}")
        print(f"  - Val F1: {f1:.4f}")
        print(f"  - Current LR: {current_lr:.8f}")
        print(f"  - Sigmoid Avg: {np.mean(sigmoid_outputs):.4f}, Std: {np.std(sigmoid_outputs):.4f}")
        print(f"  - Predicted Players: {num_pred_player}, Non-Players: {num_pred_nonplayer}\n")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break

    print("\n🧪 最終推論分析中...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    val_loss_vecs, val_labels = evaluate_tickwise_loss(model, val_dataset, device)
    preds = clf(val_loss_vecs.to(device)) > 0.5
    print(classification_report(val_labels.numpy(), preds.cpu().numpy(), digits=4))

if __name__ == "__main__":
    main()
