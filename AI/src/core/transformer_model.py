"""
Time Series Transformer for Motor Fault Detection
Uses self-attention mechanisms to capture temporal dependencies in motor signals.
"""

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MotorFaultTransformer(nn.Module):
    """
    Transformer-based model for time-series classification.
    """
    def __init__(self, num_classes, input_channels=6, d_model=64, nhead=4, 
                 num_layers=3, dim_feedforward=128, dropout=0.1, max_len=200):
        super(MotorFaultTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 1. Input Projection
        # Projects input features (channels) to d_model dimensions
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. Global Average Pooling & Classifier
        # We average over the time dimension
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x shape: [Batch, Channels, Length]
        # Transpose to [Batch, Length, Channels] for Linear layer
        x = x.transpose(1, 2)
        
        # Project to d_model
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add position encoding
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        # Output shape: [Batch, Length, d_model]
        x = self.transformer_encoder(x)
        
        # Global Average Pooling over time dimension (dim=1)
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        return x
