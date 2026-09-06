"""DeepLOB - Zhang, Zohren & Roberts (arXiv:1808.03668).

Input is a (1, 100, 40) window: 100 consecutive book states, each 40 numbers
laid out as [ask_p1, ask_v1, bid_p1, bid_v1, ..., ask_p10, ask_v10,
bid_p10, bid_v10] - which is exactly what BookState.as_row(10) produces.

The three convolution blocks are not arbitrary. Each one collapses a specific
axis of that layout:

    (1,2) stride (1,2)   pairs each price with its own volume      40 -> 20
    (1,2) stride (1,2)   pairs the bid side with the ask side      20 -> 10
    (1,10)               spans all ten levels at once              10 -> 1

After that the feature axis is gone and only time remains, so the inception
module looks at several time scales in parallel and the LSTM carries state
across the window.

Output is three classes: down, flat, up.

~140k parameters. Trains on a free Colab GPU, or a laptop CPU given patience.
Compute was never the constraint on this project.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _leaky_block(cin: int, cout: int, kernel, stride=(1, 1)) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride=stride),
        nn.LeakyReLU(0.01),
        nn.BatchNorm2d(cout),
    )


class Inception(nn.Module):
    """Three time scales plus a pooled path, concatenated on the channel axis.

    Short-horizon book dynamics do not live at one frequency: a sweep and a
    slow rebuild of depth are both predictive and look nothing alike in time.
    """

    def __init__(self, cin: int = 32, cout: int = 64) -> None:
        super().__init__()
        self.b1 = nn.Sequential(
            _leaky_block(cin, cout, (1, 1)),
            _leaky_block(cout, cout, (3, 1)),
        )
        self.b2 = nn.Sequential(
            _leaky_block(cin, cout, (1, 1)),
            _leaky_block(cout, cout, (5, 1)),
        )
        self.b3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            _leaky_block(cin, cout, (1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (3,1) and (5,1) shorten the time axis; trim the others to match so
        # the concatenation is aligned rather than padded.
        a, b, c = self.b1(x), self.b2(x), self.b3(x)
        t = min(a.shape[2], b.shape[2], c.shape[2])
        return torch.cat([a[:, :, -t:], b[:, :, -t:], c[:, :, -t:]], dim=1)


class DeepLOB(nn.Module):
    def __init__(self, levels: int = 10, n_classes: int = 3,
                 lstm_hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.levels = levels

        # price paired with its own volume
        self.conv1 = nn.Sequential(
            _leaky_block(1, 32, (1, 2), stride=(1, 2)),
            _leaky_block(32, 32, (4, 1)),
            _leaky_block(32, 32, (4, 1)),
        )
        # bid side paired with ask side
        self.conv2 = nn.Sequential(
            _leaky_block(32, 32, (1, 2), stride=(1, 2)),
            _leaky_block(32, 32, (4, 1)),
            _leaky_block(32, 32, (4, 1)),
        )
        # across all levels at once
        self.conv3 = nn.Sequential(
            _leaky_block(32, 32, (1, levels)),
            _leaky_block(32, 32, (4, 1)),
            _leaky_block(32, 32, (4, 1)),
        )
        self.inception = Inception(32, 64)
        self.lstm = nn.LSTM(192, lstm_hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:                      # (B, T, F) -> (B, 1, T, F)
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.inception(x)                 # (B, 192, T', 1)
        x = x.squeeze(-1).permute(0, 2, 1)    # (B, T', 192)
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))   # last step only

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LogisticBaseline(nn.Module):
    """Multinomial logistic regression on engineered features.

    Present on purpose. If this matches DeepLOB, that IS the finding: it says
    the predictive content is in a handful of microstructure features and the
    deep model is not adding anything. Reporting that honestly is worth more
    than a deep model with no reference point.
    """

    def __init__(self, n_features: int, n_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:                      # accept (B, T, F), use last step
            x = x[:, -1, :]
        return self.fc(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
