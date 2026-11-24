"""Three small forecasters with one interface: (N, lookback, C) -> (N, horizon)."""

from __future__ import annotations

import torch
from torch import nn


class LinearForecaster(nn.Module):
    """Flatten the window and apply one linear map (attributions are exactly known)."""

    def __init__(self, lookback: int, channels: int, horizon: int):
        super().__init__()
        self.fc = nn.Linear(lookback * channels, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class LSTMForecaster(nn.Module):
    """One-layer LSTM, last hidden state -> linear head."""

    def __init__(self, lookback: int, channels: int, horizon: int, hidden: int = 64):
        super().__init__()
        self.rnn = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int, k: int = 3):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(nn.functional.pad(x, (self.pad, 0)))  # causal left padding
        return self.act(h) + self.skip(x)


class TCNForecaster(nn.Module):
    """Dilated causal convolutions (dilations 1, 2, 4, 8, 16) with residuals, last step -> linear head."""

    def __init__(self, lookback: int, channels: int, horizon: int, width: int = 32):
        super().__init__()
        dims = [channels] + [width] * 5
        self.blocks = nn.Sequential(*[_Block(dims[i], dims[i + 1], 2**i) for i in range(5)])
        self.head = nn.Linear(width, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x.transpose(1, 2))
        return self.head(h[:, :, -1])


ARCHS = {"linear": LinearForecaster, "lstm": LSTMForecaster, "tcn": TCNForecaster}


def build(arch: str, lookback: int, channels: int, horizon: int) -> nn.Module:
    return ARCHS[arch](lookback, channels, horizon)
