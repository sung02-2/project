import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attn_weights = self.attn(x)  # [B, T, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        return (x * attn_weights).sum(dim=1)  # [B, H]

class TransformerModelWithGroup(nn.Module):
    def __init__(self, seq_input_dim, group_input_dim, output_dim=1,
                 num_heads=8, num_layers=4, hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Linear(seq_input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pooling = AttentionPooling(hidden_dim)

        # MLP 接收 pooled [B, H] + group_feature [B, K]
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim + group_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x_seq, x_group):
        x = self.embedding(x_seq)        # [B, 64, H]
        x = self.transformer(x)          # [B, 64, H]
        pooled = self.pooling(x)         # [B, H]
        combined = torch.cat([pooled, x_group], dim=1)  # [B, H+K]
        return self.mlp_head(combined)   # [B, output_dim]
