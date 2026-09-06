"""Training and evaluation for LOBForge models.

The metrics here are deliberately unflattering. Bare accuracy on a label set
that is 60-80% flat scores 60-80% by predicting flat forever, so it is never
reported alone: per-class precision, recall and F1 plus the confusion matrix
are the minimum, and a "always predict majority" baseline is printed beside
every result so the number has a floor to be compared against.

Nothing in this module knows about walk-forward splitting. That belongs to the
evaluation tier, which is built next and which owns purging and embargo. Keep
it out of here: a training loop that also decides its own splits is a training
loop that will eventually decide a convenient one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CLASSES = ("down", "flat", "up")


class WindowDataset(Dataset):
    """Windows are sliced from a flat (T, F) array at access time.

    Materialising them would be absurd: 13M frames x 100 x 40 x 4 bytes is
    ~208 TB, while the flat array is ~2 GB. `starts` is the index of valid
    window start offsets, and it is where the trust index does its work - a
    window that straddles a gap simply never appears in it.
    """

    def __init__(self, frames: np.ndarray, labels: np.ndarray,
                 starts: np.ndarray, window: int = 100) -> None:
        self.frames = frames
        self.labels = labels
        self.starts = starts
        self.window = window

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = int(self.starts[i])
        x = self.frames[s: s + self.window]
        return torch.from_numpy(np.ascontiguousarray(x)).float(), int(self.labels[s + self.window - 1])


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    per_class: dict
    confusion: list
    majority_accuracy: float
    n: int

    def summary(self) -> str:
        lift = self.accuracy - self.majority_accuracy
        out = [f"  accuracy   {self.accuracy:.4f}   "
               f"(majority-class baseline {self.majority_accuracy:.4f}, "
               f"lift {lift:+.4f})",
               f"  macro F1   {self.macro_f1:.4f}   n={self.n:,}",
               "  class      prec   recall     f1  support"]
        for c in CLASSES:
            p = self.per_class[c]
            out.append(f"  {c:<9}{p['precision']:7.3f}{p['recall']:7.3f}"
                       f"{p['f1']:7.3f}{p['support']:9,}")
        out.append("  confusion (rows=true, cols=pred)  " + " ".join(CLASSES))
        for c, row in zip(CLASSES, self.confusion):
            out.append(f"    {c:<7}" + "".join(f"{v:>9,}" for v in row))
        return "\n".join(out)

    def as_dict(self) -> dict:
        return {"accuracy": self.accuracy, "macro_f1": self.macro_f1,
                "majority_accuracy": self.majority_accuracy,
                "per_class": self.per_class, "confusion": self.confusion,
                "n": self.n}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    k = len(CLASSES)
    conf = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    per, f1s = {}, []
    for i, name in enumerate(CLASSES):
        tp = conf[i, i]
        prec = tp / conf[:, i].sum() if conf[:, i].sum() else 0.0
        rec = tp / conf[i, :].sum() if conf[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[name] = {"precision": float(prec), "recall": float(rec),
                     "f1": float(f1), "support": int(conf[i, :].sum())}
        f1s.append(f1)
    counts = np.bincount(y_true, minlength=k)
    return Metrics(
        accuracy=float((y_true == y_pred).mean()),
        macro_f1=float(np.mean(f1s)),
        per_class=per,
        confusion=conf.tolist(),
        majority_accuracy=float(counts.max() / counts.sum()),
        n=int(len(y_true)),
    )


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 4
    seed: int = 0
    class_weights: bool = True
    device: str = "auto"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Determinism is not optional: without it a code change cannot be told
    apart from run-to-run noise."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train(model: nn.Module, train_ds: Dataset, val_ds: Dataset,
          cfg: TrainConfig = TrainConfig(), log=print) -> dict:
    set_seed(cfg.seed)
    device = cfg.resolved_device()
    model = model.to(device)

    tl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                    drop_last=True)
    vl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    weight = None
    if cfg.class_weights:
        # Labels are heavily flat-dominated. Without reweighting the model
        # collapses to predicting flat, which scores well and says nothing.
        y = np.array([train_ds[i][1] for i in range(min(len(train_ds), 20000))])
        counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
        counts[counts == 0] = 1
        weight = torch.tensor((counts.sum() / (len(CLASSES) * counts)),
                              dtype=torch.float32, device=device)

    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    best, best_state, bad, history = -1.0, None, 0, []
    log(f"device {device}  params {sum(p.numel() for p in model.parameters()):,}"
        f"  train {len(train_ds):,}  val {len(val_ds):,}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n, t0 = 0.0, 0, time.time()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y); n += len(y)

        yt, yp = predict(model, vl, device)
        m = evaluate(yt, yp)
        history.append({"epoch": epoch, "train_loss": total / max(1, n),
                        "val_accuracy": m.accuracy, "val_macro_f1": m.macro_f1})
        log(f"epoch {epoch:>3}  loss {total / max(1, n):.4f}  "
            f"val acc {m.accuracy:.4f}  macro F1 {m.macro_f1:.4f}  "
            f"{time.time() - t0:.1f}s")

        # Early stopping on macro F1, not accuracy - accuracy rewards the
        # majority-class collapse this dataset invites.
        if m.macro_f1 > best:
            best, bad = m.macro_f1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"early stop: no macro-F1 gain in {cfg.patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_macro_f1": best, "history": history, "device": device}


def log_trial(path: Path, record: dict) -> None:
    """Append-only trial ledger, written BEFORE the run rather than after.

    Failed and abandoned runs must be in here too, or the deflated Sharpe
    denominator is understated and the final number overstates itself.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps({"ts": time.time(), **record}) + "\n")
