import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os


def create_burst_labels(timestamps, window_size, burst_threshold):
    """
    For each timestamp, check if the number of arrivals in (t, t + window_size]
    exceeds burst_threshold. Uses the provided timestamps array as BOTH the query
    points AND the event pool — appropriate when query_times == event_timestamps
    (i.e., for the CfC pipeline where we predict at each event).
    """
    timestamps = np.array(timestamps)
    labels = np.zeros(len(timestamps), dtype=np.float32)
    for i, t in enumerate(timestamps):
        count = np.sum((timestamps > t) & (timestamps <= t + window_size))
        labels[i] = 1.0 if count > burst_threshold else 0.0
    return labels


def create_burst_labels_from_events(event_timestamps, query_times, window_size, burst_threshold):
    """
    For each query_time t, count raw EVENT arrivals in (t, t + window_size].
    Use this for the LSTM baseline pipeline where query_times are grid bin edges
    but the burst signal must be measured against the original event stream,
    not against the grid-edge density (which is always window_size/bin_size and
    therefore always below any reasonable burst_threshold).
    """
    event_timestamps = np.array(event_timestamps)
    query_times = np.array(query_times)
    labels = np.zeros(len(query_times), dtype=np.float32)
    for i, t in enumerate(query_times):
        count = np.sum((event_timestamps > t) & (event_timestamps <= t + window_size))
        labels[i] = 1.0 if count > burst_threshold else 0.0
    return labels


def _labels_for_split(split_ts, window_size, burst_threshold):
    """
    Compute burst labels using ONLY the events within this split so that no
    future-split events leak into a split's labels at the boundary.
    """
    split_ts = np.array(split_ts)
    labels = np.zeros(len(split_ts), dtype=np.float32)
    for i, t in enumerate(split_ts):
        count = np.sum((split_ts > t) & (split_ts <= t + window_size))
        labels[i] = 1.0 if count > burst_threshold else 0.0
    return labels


def _labels_for_split_from_events(event_timestamps, query_times, window_size, burst_threshold):
    """
    Same as _labels_for_split but for a grid pipeline: counts raw events from
    event_timestamps (restricted to the same temporal slice) for each query_time.
    """
    event_timestamps = np.array(event_timestamps)
    query_times = np.array(query_times)
    labels = np.zeros(len(query_times), dtype=np.float32)
    for i, t in enumerate(query_times):
        count = np.sum((event_timestamps > t) & (event_timestamps <= t + window_size))
        labels[i] = 1.0 if count > burst_threshold else 0.0
    return labels


class EventDataset(Dataset):
    def __init__(self, X, dt, y, seq_len):
        """
        X: [N, num_features]
        dt: [N]   — per-step elapsed time (Δt)
        y: [N]    — binary burst label
        """
        self.X = X
        self.dt = dt
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_len]
        dt_seq = self.dt[idx:idx + self.seq_len]
        y_val = self.y[idx + self.seq_len - 1]   # target at the last step of the sequence

        return (torch.tensor(x_seq, dtype=torch.float32),
                torch.tensor(dt_seq, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32))


def prepare_cfc_data(df, window_size=10.0, burst_threshold=5, seq_len=20):
    """
    Prepare data for the CfC model.
    - Preserves real, irregular event timestamps.
    - Passes Δt explicitly as the timespans argument.
    - Computes burst labels per-split to avoid boundary leakage.
    """
    timestamps = df['timestamp'].values

    dt = np.diff(timestamps, prepend=timestamps[0])
    # Features: [Δt, constant event-indicator]
    # NOTE: Δt also flows through the separate timespans channel to the CfC cell.
    # The constant 1.0 simply signals "an event occurred".
    X = np.stack([dt, np.ones_like(dt)], axis=-1)

    # Strict chronological split (70 / 10 / 20) — no shuffling across the time axis.
    n = len(X)
    train_idx = int(0.7 * n)
    val_idx = int(0.8 * n)

    # Compute labels per-split using only that split's own events to prevent
    # future-event leakage at the train/val and val/test boundaries.
    ts_train = timestamps[:train_idx]
    ts_val   = timestamps[train_idx:val_idx]
    ts_test  = timestamps[val_idx:]

    y_train = _labels_for_split(ts_train, window_size, burst_threshold)
    y_val   = _labels_for_split(ts_val,   window_size, burst_threshold)
    y_test  = _labels_for_split(ts_test,  window_size, burst_threshold)

    datasets = {
        'train': EventDataset(X[:train_idx],          dt[:train_idx],          y_train, seq_len),
        'val':   EventDataset(X[train_idx:val_idx],   dt[train_idx:val_idx],   y_val,   seq_len),
        'test':  EventDataset(X[val_idx:],            dt[val_idx:],            y_test,  seq_len),
    }
    return datasets


def prepare_baseline_data(df, bin_size=1.0, window_size=10.0, burst_threshold=5, seq_len=20):
    """
    Prepare data for the discrete-time LSTM baseline.
    - Resamples the event stream onto a fixed time grid (bin_size-second bins).
    - LSTM feature: event count per bin.
    - Burst labels are computed by counting raw events in the forward window from
      each grid point — NOT by counting grid points (which always gives the same
      density regardless of burstiness and would make all labels 0 for any
      reasonable threshold).
    """
    timestamps = df['timestamp'].values

    max_time = timestamps[-1]
    bins = np.arange(0, max_time + bin_size, bin_size)
    counts, _ = np.histogram(timestamps, bins=bins)

    # The grid query times are the right edges of the bins.
    grid_times = bins[1:]

    # Strict chronological split on the grid.
    n = len(counts)
    train_idx = int(0.7 * n)
    val_idx   = int(0.8 * n)

    # Compute burst labels per-split: for each grid point, count raw events in
    # the forward window using only events from the same temporal slice.
    gt_train = grid_times[:train_idx]
    gt_val   = grid_times[train_idx:val_idx]
    gt_test  = grid_times[val_idx:]

    ts_train = timestamps[timestamps <= gt_train[-1]] if len(gt_train) > 0 else timestamps[:0]
    ts_val   = timestamps[(timestamps > gt_train[-1]) & (timestamps <= gt_val[-1])] if len(gt_val) > 0 else timestamps[:0]
    ts_test  = timestamps[timestamps > gt_val[-1]] if len(gt_test) > 0 else timestamps[:0]

    y_train = _labels_for_split_from_events(ts_train, gt_train, window_size, burst_threshold)
    y_val   = _labels_for_split_from_events(ts_val,   gt_val,   window_size, burst_threshold)
    y_test  = _labels_for_split_from_events(ts_test,  gt_test,  window_size, burst_threshold)

    X = counts.reshape(-1, 1).astype(np.float32)
    dt = np.full(len(counts), bin_size, dtype=np.float32)

    datasets = {
        'train': EventDataset(X[:train_idx],          dt[:train_idx],          y_train, seq_len),
        'val':   EventDataset(X[train_idx:val_idx],   dt[train_idx:val_idx],   y_val,   seq_len),
        'test':  EventDataset(X[val_idx:],            dt[val_idx:],            y_test,  seq_len),
    }
    return datasets


def get_dataloaders(datasets, batch_size=64):
    return {
        split: DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'))
        for split, ds in datasets.items()
    }
