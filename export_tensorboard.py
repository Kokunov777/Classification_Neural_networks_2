# -*- coding: utf-8 -*-
"""Экспорт скалярных метрик TensorBoard в PNG и CSV (папка logs/)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

ROOT = Path("logs_tb_titanic")
OUT = Path("logs")
OUT.mkdir(exist_ok=True)

TAGS = {
    "epoch_loss": "loss",
    "epoch_accuracy": "accuracy",
    "epoch_auc": "auc",
    "epoch_learning_rate": "lr",
}


def scalar_value(v) -> float:
    if len(v.tensor.float_val):
        return float(v.tensor.float_val[0])
    if len(v.tensor.tensor_content):
        return float(np.frombuffer(v.tensor.tensor_content, dtype="<f4")[0])
    return float(v.simple_value)


def read_split(directory: Path) -> pd.DataFrame:
    rows = {}
    for f in sorted(directory.glob("events*")):
        for event in EventFileLoader(str(f)).Load():
            for v in event.summary.value:
                if v.tag in TAGS:
                    rows.setdefault(event.step, {})[TAGS[v.tag]] = scalar_value(v)
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return df


# ---------------------------------------------------------------------------
records = []
for run_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    for split in ("train", "validation"):
        d = run_dir / split
        if not d.exists():
            continue
        df = read_split(d)
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"index": "step"})
        df["epoch"] = df["step"] + 1
        df["run"] = run_dir.name
        df["split"] = split
        records.append(df)

data = pd.concat(records, ignore_index=True)
data = data[["run", "split", "epoch", "loss", "accuracy", "auc", "lr"]]
data.to_csv(OUT / "tensorboard_scalars.csv", index=False)
print("runs:", data["run"].nunique(), "| rows:", len(data))


def series(run: str, split: str) -> pd.DataFrame:
    return data[(data["run"] == run) & (data["split"] == split)].sort_values("epoch")


# ---------------------------------------------------------------------------
# 1. Baseline: train vs validation
tr, va = series("baseline", "train"), series("baseline", "validation")
best_epoch = int(va.loc[va["loss"].idxmin(), "epoch"])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, metric, title in zip(
    axes,
    ["loss", "accuracy", "auc"],
    ["Loss (binary crossentropy)", "Accuracy", "AUC-ROC"],
):
    ax.plot(tr["epoch"], tr[metric], color="#1f77b4", lw=1.8, label="train")
    ax.plot(va["epoch"], va[metric], color="#d62728", lw=1.8, label="validation")
    ax.axvline(best_epoch, color="gray", ls=":", lw=1.2,
               label=f"min val_loss (эпоха {best_epoch})")
    ax.set_title(title)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("TensorBoard / baseline: обучение и валидация (32 → 16)", fontsize=13, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "tensorboard_baseline_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Learning rate comparison (validation)
lr_runs = [("baseline", "lr = 1e-3"), ("exp_lr_0.01", "lr = 1e-2"),
           ("exp_lr_0.0001", "lr = 1e-4")]
colors = ["#1f77b4", "#d62728", "#2ca02c"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for (run, label), color in zip(lr_runs, colors):
    d = series(run, "validation")
    axes[0].plot(d["epoch"], d["loss"], lw=1.8, color=color, label=label)
    axes[1].plot(d["epoch"], d["accuracy"], lw=1.8, color=color, label=label)

axes[0].set_title("Validation loss")
axes[0].set_xlabel("Эпоха")
axes[0].set_ylabel("val_loss")
axes[0].legend(title="learning rate")
axes[0].grid(alpha=0.3)

axes[1].set_title("Validation accuracy")
axes[1].set_xlabel("Эпоха")
axes[1].set_ylabel("val_accuracy")
axes[1].legend(title="learning rate")
axes[1].grid(alpha=0.3)

fig.suptitle("TensorBoard / влияние learning rate на сходимость", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "tensorboard_learning_rate.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Сравнение конфигураций (validation)
selected = [
    ("baseline", "baseline (32,16)"),
    ("exp_wide_64_32", "(64,32)"),
    ("exp_dropout_0.5", "dropout 0.5"),
    ("exp_batch_128", "batch 128"),
    ("exp_sgd", "SGD momentum"),
]
palette = plt.get_cmap("tab10").colors

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for (run, label), color in zip(selected, palette):
    d = series(run, "validation")
    axes[0].plot(d["epoch"], d["loss"], lw=1.7, color=color, label=label)
    axes[1].plot(d["epoch"], d["auc"], lw=1.7, color=color, label=label)

axes[0].set_title("Validation loss по конфигурациям")
axes[0].set_xlabel("Эпоха")
axes[0].set_ylabel("val_loss")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8)

axes[1].set_title("Validation AUC-ROC по конфигурациям")
axes[1].set_xlabel("Эпоха")
axes[1].set_ylabel("val_auc")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8)

fig.suptitle("TensorBoard / сравнение запусков (Scalars)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "tensorboard_runs_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("saved:", [p.name for p in sorted(OUT.glob("*.png"))])
