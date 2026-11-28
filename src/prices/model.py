"""Neural sequence model for pooled monthly mandi price forecasting."""
from __future__ import annotations

import torch
from torch import nn


class PriceLSTM(nn.Module):
    """Encode historical monthly prices and arrivals, plus market-series ID.

    Inputs:
        sequence: (batch, lookback_months, n_features)
        series_index: (batch,), integer identity of market × commodity × variety
    Output:
        log1p prediction of next month's modal price.
    """

    def __init__(
        self,
        input_dim: int,
        n_series: int,
        *,
        hidden_dim: int = 64,
        embedding_dim: int = 12,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.series_embedding = nn.Embedding(
            num_embeddings=n_series + 1, embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, sequence: torch.Tensor, series_index: torch.Tensor
    ) -> torch.Tensor:
        """Return a batch of predicted log1p modal prices."""
        _, (hidden, _) = self.encoder(sequence)
        temporal = hidden[-1]
        market = self.series_embedding(series_index)
        return self.regressor(torch.cat([temporal, market], dim=1)).squeeze(1)
