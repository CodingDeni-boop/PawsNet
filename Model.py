
import torch
import torch.nn as nn
import math

# ============================================================================
# Model Architecture (must match trained model file)
# ============================================================================

class ResBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv_block(x))


class CNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int = 512, res_depth: int = 4, dropout: float = 0.3):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 48, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.res_blocks_1 = nn.Sequential(*[ResBlock2D(48) for _ in range(res_depth)])
        self.transition_1 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )
        self.res_blocks_2 = nn.Sequential(*[ResBlock2D(64) for _ in range(res_depth)])
        self.transition_2 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(80),
            nn.Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )
        self.res_blocks_3 = nn.Sequential(*[ResBlock2D(80) for _ in range(res_depth)])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 9 * 4, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks_1(x)
        x = self.transition_1(x)
        x = self.res_blocks_2(x)
        x = self.transition_2(x)
        x = self.res_blocks_3(x)
        return self.fc(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNTransformerClassifier(nn.Module):
    def __init__(self, cnn_feature_dim=512, d_model=512, nhead=8,
                 num_layers=2, num_classes=5, dim_feedforward=2048, dropout=0.3):
        super().__init__()
        self.cnn = CNNFeatureExtractor(feature_dim=cnn_feature_dim, dropout=dropout)
        self.feature_projection = nn.Linear(cnn_feature_dim, d_model) if cnn_feature_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout * 0.5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        x = self.feature_projection(cnn_features)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.classifier(x)
