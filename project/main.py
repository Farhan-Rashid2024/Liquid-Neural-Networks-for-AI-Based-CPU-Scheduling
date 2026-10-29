import os
import random
import pandas as pd
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns

from config import ALIBABA_CSV
from synthetic_data import generate_poisson_events, generate_mmpp_events
from data_pipeline import prepare_cfc_data, prepare_baseline_data, get_dataloaders
from train import CfCModel, LSTMModel, train_model, evaluate


def measure_inference(model, x, dt, warmup=10, reps=100):
    """
    Measure mean per-batch inference latency.
    - warmup: number of un-timed warm-up passes to allow PyTorch to settle.
    - reps: number of timed repetitions to average over.
    - Uses time.perf_counter() for sub-millisecond accuracy (important on Windows
      where time.time() has only ~15 ms resolution).
    Note: the CfC uses a Python step loop while PyTorch LSTM uses a native fused
    kernel, so the raw timing gap understates any architectural difference.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x, dt)
    times = []
    with torch.no_grad():
        for _ in range(reps):
            start = time.perf_counter()
            _ = model(x, dt)
            times.append(time.perf_counter() - start)
    return float(np.mean(times))


def run_experiment(df, dataset_name, window_size=10.0, burst_threshold=5, seq_len=20, bin_size=1.0):
    print(f"\n--- Running experiment on {dataset_name} ---")

    # Prepare CfC Data (Continuous — irregular timestamps + explicit Δt)
    cfc_datasets = prepare_cfc_data(df, window_size, burst_threshold, seq_len)
    cfc_loaders = get_dataloaders(cfc_datasets, batch_size=64)

    # Prepare Baseline Data (Discrete Grid — fixed bin resampling)
    baseline_datasets = prepare_baseline_data(df, bin_size, window_size, burst_threshold, seq_len)
    baseline_loaders = get_dataloaders(baseline_datasets, batch_size=64)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Features for CfC: [Δt, event_indicator] → 2 features
    # Features for LSTM: [count_in_bin] → 1 feature
    # hidden_size=71 gives LSTM 21,088 params ≈ CfC's 21,025 params (parameter-matched).
    cfc_model  = CfCModel(input_size=2, hidden_size=32)
    lstm_model = LSTMModel(input_size=1, hidden_size=71)  # 21,088 params ≈ CfC's 21,025 (fixed: was 32 → 4,513 params)

    cfc_params  = sum(p.numel() for p in cfc_model.parameters()  if p.requires_grad)
    lstm_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
    print(f"CfC Parameters: {cfc_params:,}, LSTM Parameters: {lstm_params:,}")

    # Report label class balance so we can detect degenerate splits
    for split in ('train', 'val', 'test'):
        cfc_y  = np.array([cfc_datasets[split][i][2].item()  for i in range(len(cfc_datasets[split]))])
        lstm_y = np.array([baseline_datasets[split][i][2].item() for i in range(len(baseline_datasets[split]))])
        print(f"  [{split}] CfC  burst rate: {cfc_y.mean():.3f} ({int(cfc_y.sum())}/{len(cfc_y)})")
        print(f"  [{split}] LSTM burst rate: {lstm_y.mean():.3f} ({int(lstm_y.sum())}/{len(lstm_y)})")

    print("Training CfC Model...")
    cfc_history = train_model(cfc_model, cfc_loaders, epochs=20, lr=0.005, device=device)

    print("Training LSTM Baseline...")
    lstm_history = train_model(lstm_model, baseline_loaders, epochs=20, lr=0.005, device=device)

    # Evaluate on Test Set
    criterion = torch.nn.BCEWithLogitsLoss()
    print("Evaluating CfC on Test Set...")
    cfc_loss, cfc_acc, cfc_auc, cfc_f1 = evaluate(cfc_model, cfc_loaders['test'], criterion, device)

    print("Evaluating LSTM on Test Set...")
    lstm_loss, lstm_acc, lstm_auc, lstm_f1 = evaluate(lstm_model, baseline_loaders['test'], criterion, device)

    print(f"\n[Results - {dataset_name}]")
    print(f"CfC  - AUC: {cfc_auc:.4f}, F1: {cfc_f1:.4f}, Acc: {cfc_acc:.4f}")
    print(f"LSTM - AUC: {lstm_auc:.4f}, F1: {lstm_f1:.4f}, Acc: {lstm_acc:.4f}")

    # Inference Speed Test (CPU, warmup-corrected, high-precision timer)
    cfc_model.to('cpu')
    lstm_model.to('cpu')

    x_cfc,  dt_cfc,  _ = next(iter(cfc_loaders['test']))
    x_lstm, dt_lstm, _ = next(iter(baseline_loaders['test']))
    x_cfc,  dt_cfc  = x_cfc.to('cpu'),  dt_cfc.to('cpu')
    x_lstm           = x_lstm.to('cpu')

    cfc_inf_time  = measure_inference(cfc_model,  x_cfc,  dt_cfc)
    lstm_inf_time = measure_inference(lstm_model, x_lstm, dt_lstm)

    print(f"CfC  Inference Time (batch, CPU): {cfc_inf_time  * 1000:.3f} ms")
    print(f"LSTM Inference Time (batch, CPU): {lstm_inf_time * 1000:.3f} ms")

    return {
        'dataset': dataset_name,
        'cfc_auc':  cfc_auc,  'cfc_f1':  cfc_f1,  'cfc_acc':  cfc_acc,  'cfc_time':  cfc_inf_time,
        'lstm_auc': lstm_auc, 'lstm_f1': lstm_f1, 'lstm_acc': lstm_acc, 'lstm_time': lstm_inf_time,
        'cfc_params': cfc_params, 'lstm_params': lstm_params,
    }


if __name__ == "__main__":
    # ── Reproducibility seeds ──────────────────────────────────────────────────
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ──────────────────────────────────────────────────────────────────────────

    os.makedirs("figures", exist_ok=True)
    os.makedirs("data",    exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    # 1. Generate Synthetic Data
    df_poisson = generate_poisson_events(rate=2.0, duration=20000)
    df_bursty  = generate_mmpp_events(lambda_low=0.5, lambda_high=15.0,
                                      p_low_to_high=0.01, p_high_to_low=0.05,
                                      duration=20000)

    # 2. Load Alibaba Data
    alibaba_path = ALIBABA_CSV
    try:
        df_alibaba = pd.read_csv(alibaba_path)
        df_alibaba = df_alibaba.sort_values('submit_time').head(20000).copy()
        df_alibaba.rename(columns={'submit_time': 'timestamp'}, inplace=True)
        has_alibaba = True
    except Exception as e:
        print(f"Could not load Alibaba dataset: {e}")
        has_alibaba = False

    results = []

    results.append(run_experiment(df_poisson, "Synthetic Poisson",
                                  window_size=10.0, burst_threshold=25))
    results.append(run_experiment(df_bursty, "Synthetic Bursty (MMPP)",
                                  window_size=10.0, burst_threshold=25))

    if has_alibaba:
        results.append(run_experiment(df_alibaba, "Alibaba Trace",
                                      window_size=300.0, burst_threshold=10, bin_size=10.0))

    # ── Summarise ──────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(results)
    print("\n--- Final Results ---")
    print(df_res.to_string(index=False))
    df_res.to_csv("data/results_summary.csv", index=False)

    # ── Plots ──────────────────────────────────────────────────────────────────
    x_pos = np.arange(len(df_res))
    width = 0.35

    # AUC comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos - width / 2, df_res['cfc_auc'],  width, label='CfC',  color='#4C72B0')
    ax.bar(x_pos + width / 2, df_res['lstm_auc'], width, label='LSTM', color='#DD8452')
    ax.axhline(0.5, linestyle='--', color='grey', linewidth=0.8, label='Random baseline (AUC=0.5)')
    ax.set_ylabel('AUC Score')
    ax.set_title('Burst Prediction AUC by Dataset')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_res['dataset'])
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/auc_comparison.png', dpi=300)
    plt.close(fig)

    # Inference time comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos - width / 2, df_res['cfc_time']  * 1000, width, label='CfC',  color='#4C72B0')
    ax.bar(x_pos + width / 2, df_res['lstm_time'] * 1000, width, label='LSTM', color='#DD8452')
    ax.set_ylabel('Inference Time per Batch (ms)')
    ax.set_title('Inference Speed Comparison (CPU, 100 reps, 10 warmup)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_res['dataset'])
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/inference_time.png', dpi=300)
    plt.close(fig)

    print("\nExperiment complete. Figures saved to 'figures/', results to 'data/results_summary.csv'")
