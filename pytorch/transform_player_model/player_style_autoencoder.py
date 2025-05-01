# ===== 模型定義：player_style_autoencoder.py =====
import torch
import torch.nn as nn

class PlayerStyleTransformerAutoEncoder(nn.Module):
    def __init__(self, seq_input_dim=18, group_input_dim=5, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(seq_input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.group_encoder = nn.Sequential(
            nn.Linear(group_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.merge = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        self.decoder_seq = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_input_dim)
        )

        self.decoder_group = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, group_input_dim)
        )

    def forward(self, x_seq, x_group):
        B, T, D = x_seq.shape
        x_seq_emb = self.embedding(x_seq)
        encoded_seq = self.transformer_encoder(x_seq_emb)
        pooled_seq = encoded_seq.mean(dim=1)

        x_group_encoded = self.group_encoder(x_group)
        merged = torch.cat([pooled_seq, x_group_encoded], dim=1)
        hidden = self.merge(merged)

        recon_seq = self.decoder_seq(hidden)
        recon_group = self.decoder_group(hidden)
        return recon_seq, recon_group