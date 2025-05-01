import torch
import torch.nn as nn

class TransformerModelWithGroup(nn.Module):
    def __init__(self, seq_input_dim, group_input_dim, output_dim,
                 num_heads, num_layers, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 時序輸入特徵嵌入
        self.embedding = nn.Linear(seq_input_dim, hidden_dim)

        # Transformer 編碼層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 可學習的 CLS token（作為全序列的代表）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # 分類頭：結合 CLS + Group 特徵後分類
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim + group_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x_seq, x_group):
        B = x_seq.size(0)

        # 1. 嵌入每個 tick 的特徵
        x = self.embedding(x_seq)  # [B, 64, H]

        # 2. 加入 CLS token
        cls_token = self.cls_token.expand(B, 1, self.hidden_dim)  # [B, 1, H]
        x = torch.cat([cls_token, x], dim=1)  # [B, 65, H]

        # 3. 通過 Transformer 編碼器
        x = self.transformer(x)  # [B, 65, H]

        # 4. 抽取 CLS token 的輸出作為 pooled 特徵
        pooled = x[:, 0, :]  # [B, H]

        # 5. 拼接 group 特徵並通過 MLP
        combined = torch.cat([pooled, x_group], dim=1)  # [B, H + group_dim]
        return self.mlp_head(combined)  # [B, output_dim]
