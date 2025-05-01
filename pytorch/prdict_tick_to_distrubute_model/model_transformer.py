import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ===== 模型定義：每 tick 預測下一 tick =====
class StepwisePredictor(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # 預測下一 tick 的所有特徵
        )

    def forward(self, x):  # x: [B, t, input_dim]
        x = self.embedding(x)
        x = self.transformer(x)
        return self.predictor(x[:, -1])  # [B, input_dim] 預測最後一個 tick 的下一個


# ===== 自訂 Dataset：tick 0~t → 預測 tick t+1 =====
class AutoregressiveTickDataset(Dataset):
    def __init__(self, data_tensor):
        self.samples = []
        self.labels = []
        for i in range(len(data_tensor['x_seq'])):
            seq = data_tensor['x_seq'][i]  # [64, 18]
            label = data_tensor['y'][i].item()
            for t in range(1, 64):
                x = seq[:t, :]     # [t, 18]
                y = seq[t, :]      # [18]
                self.samples.append((x, y, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ===== 專用 collate function（支援 padding） =====
def autoregressive_collate_fn(batch):
    x_list, y_list, label_list = zip(*batch)
    x_padded = pad_sequence(x_list, batch_first=True)  # [B, max_t, input_dim]
    y_tensor = torch.stack(y_list)  # [B, input_dim]
    label_tensor = torch.tensor(label_list, dtype=torch.int64)  # [B]
    return x_padded, y_tensor, label_tensor


# ===== 預測誤差計算（支援 Autoregressive 結構） =====
def evaluate_prediction_error(model, data_tensor, device):
    model.eval()
    dataset = AutoregressiveTickDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=1, collate_fn=autoregressive_collate_fn)
    errors = []
    criterion = nn.MSELoss(reduction='mean')

    with torch.no_grad():
        for idx, (x, y, label) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y).item()
            errors.append({"GroupID": idx // 63, "Loss": loss, "Label": label.item()})

    return errors