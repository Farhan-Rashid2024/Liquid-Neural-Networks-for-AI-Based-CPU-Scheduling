"""
generate_figures.py
-------------------
Additive figures script for the LNN Scheduling paper.

Re-trains models with the same configuration as main.py (same seeds, same
parameters) to capture per-epoch training histories, then generates four new
diagnostic figures that explain *why* the headline metrics came out as they did.

Four new figures produced:
  figures/burst_reaction_timeline.png  — quantization-delay visualisation
  figures/training_curves.png          — loss + val-AUC per epoch, all datasets
  figures/roc_curves.png               — ROC curves on test set, all datasets
  figures/class_balance.png            — positive-label fraction per split

Also saves:
  data/training_histories.json         — raw per-epoch metrics for all runs
  models/{tag}_cfc.pt / _lstm.pt       — trained weights (not previously saved)

IMPORTANT: This file does NOT modify evaluate(), train_model(),
prepare_cfc_data(), prepare_baseline_data(), or any other function in
train.py / data_pipeline.py. All new logic is local to this file.
"""

import os
import json
import random

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc as sklearn_auc

from config import ALIBABA_CSV
from synthetic_data import generate_poisson_events, generate_mmpp_events
from data_pipeline import prepare_cfc_data, prepare_baseline_data, get_dataloaders
from train import CfCModel, LSTMModel, train_model

# ── Global style constants (match existing figures in figures/) ───────────────
CFC_COLOR  = "#4C72B0"
LSTM_COLOR = "#DD8452"
DPI = 300

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# Helper: collect raw per-sample probabilities from a test dataset
# (additive — does not touch the existing evaluate() function)
# ══════════════════════════════════════════════════════════════════════════════
def get_test_probs_and_labels(model, dataset, device, batch_size=256):
    """
    Run model over dataset in batches, returning sigmoid probabilities and true
    labels as numpy arrays. Uses DataLoader for speed; does not modify evaluate().
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, dt, y in loader:
            x, dt = x.to(device), dt.to(device)
            logit = model(x, dt).squeeze(-1)
            probs = torch.sigmoid(logit).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: timeline prediction (new — does not touch evaluate())
# ══════════════════════════════════════════════════════════════════════════════
def predict_timeline(cfc_model, lstm_model,
                     cfc_test_ds, lstm_test_ds,
                     cfc_test_timestamps, lstm_test_grid_times,
                     seq_len, device, batch_size=128):
    """
    Returns four arrays: (cfc_times, cfc_probs, lstm_times, lstm_probs).

    Each 'times' array is the real-time coordinate of every prediction in the
    test set; each 'probs' array is the sigmoid burst probability at that point.
    CfC predicts at event timestamps; LSTM predicts at fixed grid edges.
    """
    from torch.utils.data import DataLoader

    def collect(model, dataset, times_array):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        probs_list = []
        with torch.no_grad():
            for x, dt, _ in loader:
                x, dt = x.to(device), dt.to(device)
                logit = model(x, dt).squeeze(-1)
                probs_list.append(torch.sigmoid(logit).cpu().numpy())
        probs = np.concatenate(probs_list)

        # Align to timestamps: prediction idx corresponds to the last event in
        # its window, i.e. times_array[idx + seq_len - 1]
        max_idx = len(times_array) - seq_len
        n = min(len(probs), max_idx + 1)
        times = times_array[seq_len - 1: seq_len - 1 + n]
        return times, probs[:n]

    cfc_times,  cfc_probs  = collect(cfc_model,  cfc_test_ds,  cfc_test_timestamps)
    lstm_times, lstm_probs = collect(lstm_model, lstm_test_ds, lstm_test_grid_times)
    return cfc_times, cfc_probs, lstm_times, lstm_probs


# ══════════════════════════════════════════════════════════════════════════════
# Helper: compute class balance for one EventDataset
# ══════════════════════════════════════════════════════════════════════════════
def burst_rate(dataset):
    labels = np.array([dataset[i][2].item() for i in range(len(dataset))])
    return float(labels.mean()), len(labels)


# ══════════════════════════════════════════════════════════════════════════════
# Core per-dataset runner
# ══════════════════════════════════════════════════════════════════════════════
def run_dataset(df, name, window_size, burst_threshold, seq_len, bin_size,
                device, epochs=20, lr=0.005):
    print(f"\n{'='*62}\n  {name}\n{'='*62}")

    cfc_ds  = prepare_cfc_data(df, window_size, burst_threshold, seq_len)
    lstm_ds = prepare_baseline_data(df, bin_size, window_size, burst_threshold, seq_len)

    cfc_loaders  = get_dataloaders(cfc_ds,  batch_size=64)
    lstm_loaders = get_dataloaders(lstm_ds, batch_size=64)

    cfc_model  = CfCModel(input_size=2,  hidden_size=32)
    lstm_model = LSTMModel(input_size=1, hidden_size=71)

    print("  Training CfC ...")
    cfc_hist  = train_model(cfc_model,  cfc_loaders,  epochs=epochs, lr=lr, device=device)
    print("  Training LSTM ...")
    lstm_hist = train_model(lstm_model, lstm_loaders, epochs=epochs, lr=lr, device=device)

    # Save weights
    tag = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    torch.save(cfc_model.state_dict(),  f"models/{tag}_cfc.pt")
    torch.save(lstm_model.state_dict(), f"models/{tag}_lstm.pt")

    # Test-set probabilities for ROC and timeline
    cfc_probs,  cfc_labels  = get_test_probs_and_labels(cfc_model,  cfc_ds["test"],  device)
    lstm_probs, lstm_labels = get_test_probs_and_labels(lstm_model, lstm_ds["test"], device)

    # Class balance
    balance = {}
    for split in ("train", "val", "test"):
        cr, cn = burst_rate(cfc_ds[split])
        lr_, ln = burst_rate(lstm_ds[split])
        balance[split] = dict(cfc_rate=cr, cfc_n=cn, lstm_rate=lr_, lstm_n=ln)
        print(f"  [{split}] CfC {cr:.3f} ({int(cr*cn)}/{cn})  "
              f"LSTM {lr_:.3f} ({int(lr_*ln)}/{ln})")

    return dict(
        name=name,
        cfc_hist=cfc_hist,   lstm_hist=lstm_hist,
        cfc_probs=cfc_probs, cfc_labels=cfc_labels,
        lstm_probs=lstm_probs, lstm_labels=lstm_labels,
        balance=balance,
        cfc_model=cfc_model, lstm_model=lstm_model,
        cfc_ds=cfc_ds, lstm_ds=lstm_ds,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Burst-reaction timeline
# ══════════════════════════════════════════════════════════════════════════════
def fig_burst_timeline(res, df, window_size, burst_threshold, seq_len, bin_size, device):
    print("\n[Fig 1] Burst reaction timeline ...")
    timestamps = df["timestamp"].values
    n_events   = len(timestamps)
    val_idx    = int(0.8 * n_events)
    test_ts    = timestamps[val_idx:]          # CfC test timestamps

    # Grid test times — replicate the split logic from prepare_baseline_data
    max_t = timestamps[-1]
    bins  = np.arange(0, max_t + bin_size, bin_size)
    grid  = bins[1:]                           # right bin edges
    val_grid_idx  = int(0.8 * len(grid))
    test_grid_ts  = grid[val_grid_idx:]        # LSTM test grid

    cfc_t, cfc_p, lstm_t, lstm_p = predict_timeline(
        res["cfc_model"], res["lstm_model"],
        res["cfc_ds"]["test"], res["lstm_ds"]["test"],
        test_ts, test_grid_ts, seq_len, device,
    )

    if len(cfc_t) < seq_len or len(lstm_t) < 2:
        print("  Insufficient test data for timeline — skipping.")
        return False

    # True labels for CfC test predictions
    true_labels = np.array([
        res["cfc_ds"]["test"][i][2].item()
        for i in range(len(res["cfc_ds"]["test"]))
    ])
    true_labels = true_labels[: len(cfc_t)]

    # Find first burst-onset transition (0→1)
    onset_idx = 0
    for i in range(1, len(true_labels)):
        if true_labels[i] == 1.0 and true_labels[i - 1] == 0.0:
            onset_idx = i
            break

    # Window: 40 events before onset, 160 after
    win_s = max(0, onset_idx - 40)
    win_e = min(len(cfc_t) - 1, onset_idx + 160)
    if win_s >= win_e:
        win_s, win_e = 0, min(200, len(cfc_t) - 1)

    t0, t1 = cfc_t[win_s], cfc_t[win_e]

    cfc_mask  = (cfc_t  >= t0) & (cfc_t  <= t1)
    lstm_mask = (lstm_t >= t0) & (lstm_t <= t1)
    raw_mask  = (test_ts >= t0) & (test_ts <= t1)

    fig, ax = plt.subplots(figsize=(13, 5))

    # ── Shaded burst ground-truth region ─────────────────────────────────────
    lab_t = cfc_t[cfc_mask]
    lab_y = true_labels[cfc_mask]
    for i in range(len(lab_t) - 1):
        if lab_y[i] > 0.5:
            ax.axvspan(lab_t[i], lab_t[i + 1], alpha=0.10, color="red",
                       label="_nolegend_")
    burst_patch = mpatches.Patch(color="red", alpha=0.25, label="True burst region")

    # ── Raw event rug ─────────────────────────────────────────────────────────
    ax.plot(test_ts[raw_mask], np.full(raw_mask.sum(), -0.06),
            "|", color="grey", markersize=9, alpha=0.55, label="Raw events")

    # ── CfC continuous line ───────────────────────────────────────────────────
    ax.plot(cfc_t[cfc_mask], cfc_p[cfc_mask],
            color=CFC_COLOR, lw=2.0, label="CfC burst prob. (continuous)")

    # ── LSTM step function ────────────────────────────────────────────────────
    if lstm_mask.sum() > 1:
        ax.step(lstm_t[lstm_mask], lstm_p[lstm_mask], where="post",
                color=LSTM_COLOR, lw=2.2, linestyle="--",
                label="LSTM burst prob. (fixed-grid)")

    ax.axhline(0.5, color="black", linestyle=":", lw=0.8, alpha=0.45)

    # ── Quantization-delay annotation ────────────────────────────────────────
    if onset_idx < len(cfc_t) and cfc_t[onset_idx] >= t0:
        burst_t = cfc_t[onset_idx]
        next_lstm_pts = lstm_t[(lstm_t > burst_t) & (lstm_t <= t1)]
        if len(next_lstm_pts) > 0:
            delay = next_lstm_pts[0] - burst_t
            mid   = (burst_t + next_lstm_pts[0]) / 2
            ax.annotate("", xy=(next_lstm_pts[0], 0.48),
                        xytext=(burst_t, 0.48),
                        arrowprops=dict(arrowstyle="<->", color="darkred", lw=1.6))
            ax.text(mid, 0.52, f"Quantization\ndelay ≈ {delay:.1f}s",
                    ha="center", fontsize=9, color="darkred", fontweight="bold")

    handles, labels_leg = ax.get_legend_handles_labels()
    handles.append(burst_patch)
    labels_leg.append("True burst region")
    ax.legend(handles, labels_leg, loc="upper left", fontsize=9)

    ax.set_xlim(t0, t1)
    ax.set_ylim(-0.16, 1.10)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Predicted Burst Probability", fontsize=12)
    ax.set_title("Burst-Onset Reaction: CfC (continuous) vs. LSTM (fixed-grid)\n"
                 f"Dataset: {res['name']}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/burst_reaction_timeline.png", dpi=DPI)
    plt.close()
    print("  Saved: figures/burst_reaction_timeline.png")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Training curves
# ══════════════════════════════════════════════════════════════════════════════
def fig_training_curves(results):
    print("\n[Fig 2] Training curves ...")
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.2 * n))
    if n == 1:
        axes = [axes]

    for row, res in enumerate(results):
        ch  = res["cfc_hist"]
        lh  = res["lstm_hist"]
        eps = range(1, len(ch["train_loss"]) + 1)

        ax_l = axes[row][0]
        ax_a = axes[row][1]

        # Loss curves
        ax_l.plot(eps, ch["train_loss"], color=CFC_COLOR,  lw=1.8, label="CfC train")
        ax_l.plot(eps, ch["val_loss"],   color=CFC_COLOR,  lw=1.8, ls="--", label="CfC val")
        ax_l.plot(eps, lh["train_loss"], color=LSTM_COLOR, lw=1.8, label="LSTM train")
        ax_l.plot(eps, lh["val_loss"],   color=LSTM_COLOR, lw=1.8, ls="--", label="LSTM val")
        ax_l.set_title(f"{res['name']} — BCE Loss", fontsize=11)
        ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Loss")
        ax_l.legend(fontsize=8); ax_l.grid(alpha=0.3)

        # Val AUC curves
        ax_a.plot(eps, ch["val_auc"], color=CFC_COLOR,  lw=2.0, label="CfC val AUC")
        ax_a.plot(eps, lh["val_auc"], color=LSTM_COLOR, lw=2.0, label="LSTM val AUC")
        ax_a.axhline(0.5, color="grey", ls=":", lw=0.8, label="Random (0.5)")
        ax_a.set_title(f"{res['name']} — Validation AUC", fontsize=11)
        ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("AUC")
        ax_a.set_ylim(0.35, 1.02)
        ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)

    fig.suptitle("Training Curves — CfC vs. LSTM (20 Epochs, seed=42)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/training_curves.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/training_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — ROC curves
# ══════════════════════════════════════════════════════════════════════════════
def fig_roc_curves(results):
    print("\n[Fig 3] ROC curves ...")
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        for probs, labels, color, label in [
            (res["cfc_probs"],  res["cfc_labels"],  CFC_COLOR,  "CfC"),
            (res["lstm_probs"], res["lstm_labels"], LSTM_COLOR, "LSTM"),
        ]:
            if len(np.unique(labels)) < 2:
                ax.plot([0, 1], [0, 1], ls="--", color=color,
                        label=f"{label} (degenerate labels)")
                continue
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_val = sklearn_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.0, label=f"{label} (AUC={roc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(res["name"], fontsize=11)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

    fig.suptitle("ROC Curves — CfC vs. LSTM (Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/roc_curves.png", dpi=DPI)
    plt.close()
    print("  Saved: figures/roc_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Class balance
# ══════════════════════════════════════════════════════════════════════════════
def fig_class_balance(results):
    print("\n[Fig 4] Class balance ...")
    splits = ["train", "val", "test"]
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        x = np.arange(len(splits))
        w = 0.30
        cfc_r  = [res["balance"][s]["cfc_rate"]  for s in splits]
        lstm_r = [res["balance"][s]["lstm_rate"] for s in splits]

        b1 = ax.bar(x - w / 2, cfc_r,  w, color=CFC_COLOR,  alpha=0.85, label="CfC pipeline")
        b2 = ax.bar(x + w / 2, lstm_r, w, color=LSTM_COLOR, alpha=0.85, label="LSTM pipeline")

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(res["name"], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in splits])
        ax.set_ylabel("Positive Label Fraction (burst rate)")
        ax.set_ylim(0, 1.18)
        ax.axhline(0.5, color="grey", ls=":", lw=0.8, label="50% line")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Class Balance (Burst Rate) by Dataset, Split, and Pipeline",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/class_balance.png", dpi=DPI)
    plt.close()
    print("  Saved: figures/class_balance.png")


# ══════════════════════════════════════════════════════════════════════════════
# NEW_FIGURES.md — convergence analysis + summary
# ══════════════════════════════════════════════════════════════════════════════
def write_summary(results):
    # Alibaba convergence analysis
    alibaba_note = "_Alibaba trace not available in this run._"
    for res in results:
        if "Alibaba" in res["name"]:
            aucs = res["cfc_hist"]["val_auc"]
            first5 = float(np.mean(aucs[:5]))
            last5  = float(np.mean(aucs[-5:]))
            delta  = last5 - first5
            if delta > 0.05:
                verdict = (f"**PARTIALLY CONTRADICTS** paper claim. CfC val AUC rose from "
                           f"{first5:.3f} (epochs 1–5) to {last5:.3f} (epochs 16–20), "
                           f"a +{delta:.3f} improvement — it is still trending upward. "
                           f"This supports the paper's suggestion that more epochs are needed "
                           f"but weakens the phrasing 'has not converged to a reliable "
                           f"discriminator.' Recommended paper edit: change to 'has not yet "
                           f"fully converged within 20 epochs.'")
            elif last5 < 0.55:
                verdict = (f"**SUPPORTS** paper claim. CfC val AUC on Alibaba remains "
                           f"near-random (mean {last5:.3f} over final 5 epochs, starting from "
                           f"{first5:.3f}), confirming it has not converged to a reliable "
                           f"discriminator within 20 epochs.")
            else:
                verdict = (f"**MIXED.** CfC val AUC fluctuates around {last5:.3f} in final "
                           f"5 epochs (started at {first5:.3f}). Consistent with partial "
                           f"convergence — broadly supports the paper's cautious phrasing.")
            alibaba_note = verdict
            break

    md = f"""# New Figures — Summary (`generate_figures.py`)

Generated by `generate_figures.py` on top of the existing, validated pipeline.
No existing functions were modified; all new code is additive.

---

## Figures produced

### 1. `figures/burst_reaction_timeline.png`
Visualises the quantization-delay mechanism directly. A burst episode from the
Synthetic Bursty (MMPP) test set is shown (fallback to Alibaba if MMPP produces
no clean episode). The plot includes:
- Raw event rug (timing of actual arrivals)
- CfC's continuous per-event burst probability (smooth line)
- LSTM's fixed-grid step-function probability (step-plot, dashed)
- True burst region shaded in red
- An annotated double-headed arrow showing quantization delay between burst onset
  and the LSTM's first grid update

**New code:** `predict_timeline()` — runs models in batch over test data,
preserving the timestamp↔prediction correspondence. Does not touch `evaluate()`.

---

### 2. `figures/training_curves.png`
Grid of subplots (one row per dataset), each with two panels:
- Left: training loss + validation loss per epoch (solid = train, dashed = val)
- Right: validation AUC per epoch with a random-baseline reference line

Raw per-epoch histories saved to `data/training_histories.json`.

**New code:** `plot_training_curves()` — uses the `history` dict already returned
by `train_model()` (previously discarded after printing).

**Convergence check — Alibaba CfC, re: paper claim "has not converged within 20 epochs":**

{alibaba_note}

---

### 3. `figures/roc_curves.png`
One subplot per dataset. ROC curves for CfC and LSTM on the test set, with the
random-classifier diagonal. AUC values in the legend should closely match
`results_summary.csv` (small differences are expected because this is a fresh
training run from the same seed).

**New code:** `get_test_probs_and_labels()` + `fig_roc_curves()`.

---

### 4. `figures/class_balance.png`
Grouped bar chart showing positive-label (burst) rate for the CfC and LSTM
pipelines across train/val/test splits, for each dataset. Backs the paper's
"~96% positive on Alibaba" assertion with actual computed numbers. Note: CfC and
LSTM label rates differ because they predict at different query points (event
timestamps vs. grid edges).

**New code:** `burst_rate()` + `fig_class_balance()`.

---

## Files added / modified

| File | Action |
|------|--------|
| `generate_figures.py` | New — all new logic lives here |
| `figures/burst_reaction_timeline.png` | New |
| `figures/training_curves.png` | New |
| `figures/roc_curves.png` | New |
| `figures/class_balance.png` | New |
| `data/training_histories.json` | New |
| `models/{{tag}}_cfc.pt` / `_lstm.pt` | New — model weights now saved |

**No existing files modified.** `evaluate()`, `train_model()`,
`prepare_cfc_data()`, `prepare_baseline_data()`, `create_burst_labels*()`,
and all figures/metrics in `results_summary.csv` are untouched.

---

## Please sanity-check

1. **Timeline figure:** Confirm the burst episode window is visually legible —
   if the burst is too narrow or the plot is cluttered, adjust `win_s`/`win_e`
   offsets in `fig_burst_timeline()`.
2. **Convergence note above:** Decide whether the paper's Alibaba wording needs
   updating based on what the training curves actually show.
3. **ROC curves:** Poisson curves should be near-diagonal for both models —
   that is the correct result for a memoryless process, not a bug.
4. **Fresh training run:** Because `train_model()` is re-run (models were not
   previously saved), results will be numerically close to but not identical to
   `results_summary.csv`. The seeds are identical (42), so variance should be
   negligible. If you need bit-exact reproduction, use the saved `.pt` weights
   instead of re-running training.
"""

    with open("NEW_FIGURES.md", "w") as f:
        f.write(md)
    print("\n  Saved: NEW_FIGURES.md")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data",    exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    SEQ_LEN = 20

    # ── Generate synthetic data (same params as main.py) ──────────────────────
    df_poisson = generate_poisson_events(rate=2.0, duration=20000)
    df_bursty  = generate_mmpp_events(
        lambda_low=0.5, lambda_high=15.0,
        p_low_to_high=0.01, p_high_to_low=0.05,
        duration=20000,
    )

    # ── Load Alibaba (same path as main.py) ────────────────────────────────────
    ALIBABA_PATH = ALIBABA_CSV
    has_alibaba = False
    df_alibaba  = None
    try:
        df_alibaba = pd.read_csv(ALIBABA_PATH)
        df_alibaba = df_alibaba.sort_values("submit_time").head(20000).copy()
        df_alibaba.rename(columns={"submit_time": "timestamp"}, inplace=True)
        has_alibaba = True
        print("Alibaba trace loaded.")
    except Exception as e:
        print(f"Alibaba not available: {e}")

    # ── Run experiments ────────────────────────────────────────────────────────
    experiments = [
        (df_poisson, "Synthetic Poisson",       10.0, 25,  SEQ_LEN, 1.0),
        (df_bursty,  "Synthetic Bursty (MMPP)", 10.0, 25,  SEQ_LEN, 1.0),
    ]
    if has_alibaba:
        experiments.append(
            (df_alibaba, "Alibaba Trace", 300.0, 10, SEQ_LEN, 10.0)
        )

    results = []
    for df, name, ws, bt, sl, bs in experiments:
        res = run_dataset(df, name, ws, bt, sl, bs, device)
        results.append(res)

    # ── Save training histories (JSON-serialisable) ────────────────────────────
    histories_out = {}
    for res in results:
        histories_out[res["name"]] = {
            "cfc":  {k: [float(v) for v in vals]
                     for k, vals in res["cfc_hist"].items()},
            "lstm": {k: [float(v) for v in vals]
                     for k, vals in res["lstm_hist"].items()},
        }
    with open("data/training_histories.json", "w") as f:
        json.dump(histories_out, f, indent=2)
    print("\nSaved: data/training_histories.json")

    # ── Figure 1 — timeline (MMPP preferred, Alibaba fallback) ────────────────
    mmpp_res = results[1]
    ok = fig_burst_timeline(
        mmpp_res, df_bursty,
        window_size=10.0, burst_threshold=25,
        seq_len=SEQ_LEN, bin_size=1.0, device=device,
    )
    if not ok and has_alibaba:
        print("  Falling back to Alibaba for timeline ...")
        fig_burst_timeline(
            results[-1], df_alibaba,
            window_size=300.0, burst_threshold=10,
            seq_len=SEQ_LEN, bin_size=10.0, device=device,
        )

    # ── Figures 2–4 ────────────────────────────────────────────────────────────
    fig_training_curves(results)
    fig_roc_curves(results)
    fig_class_balance(results)

    # ── Summary doc ────────────────────────────────────────────────────────────
    write_summary(results)

    print("\n✓ All four figures generated. Check figures/ and NEW_FIGURES.md.")
