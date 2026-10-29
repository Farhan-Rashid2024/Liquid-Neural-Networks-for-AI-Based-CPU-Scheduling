# Architecture and methodology

## Data and labels

`main.py` generates two synthetic workloads and optionally loads the derived Alibaba PAI CSV. It sorts Alibaba records by `submit_time`, takes 20,000 rows, and renames that field to `timestamp`. Other CSV columns do not enter the models.

At query time t, label 1 means the raw arrival count in `(t, t + W]` is strictly greater than K. This is future high-arrival-window classification, rather than explicit detection of the exact burst-start transition.

| Setting | Poisson | MMPP | Alibaba |
| --- | --- | --- | --- |
| Source duration / selection | 20,000 seconds | 20,000 seconds | First 20,000 sorted records |
| Arrival rate | 2/s | 0.5/s or 15/s | Observed submissions |
| Future horizon W | 10 s | 10 s | 300 s |
| Threshold K | 25 | 25 | 10 |
| LSTM bin width | 1 s | 1 s | 10 s |

MMPP state transitions have rates 0.01/s (quiet to busy) and 0.05/s (busy to quiet). Despite their `p_` parameter names, these values serve as rates in exponential waiting-time sampling, not per-step transition probabilities.

## CfC branch

1. `prepare_cfc_data()` computes inter-arrival gaps with `np.diff`; the first gap is zero.
2. Each event supplies `[dt, 1]`. The same dt also enters the explicit `timespans` channel.
3. `EventDataset` provides 20-event windows. For batch size B, X has shape `[B, 20, 2]`, dt has shape `[B, 20]`, and labels have shape `[B]`.
4. `CfCModel` initializes a zero `[B, 32]` hidden state for each window. It runs the library CfCCell for 20 steps, with each timespan reshaped to `[B, 1]`.
5. A linear layer converts the final hidden output to one logit. Total trainable parameters: 21,025.

The library default gated update takes the form `g = sigmoid(a * dt + b)` and `h_new = (1-g) * u + g * v`. Learned functions of the input and previous hidden state supply a, b and the tanh candidates u and v. Products are elementwise. CfC avoids numerical ODE integration at each step.

## LSTM branch

1. `prepare_baseline_data()` builds a histogram on a fixed grid starting at zero.
2. Each bin supplies its arrival count. Query times are bin right edges.
3. X has shape `[B, 20, 1]`. Each sequence spans 20 s for synthetic data or 200 s for Alibaba.
4. PyTorch LSTM uses hidden size 71. A linear layer maps the final output to one logit. Total parameters: 21,088.
5. The LSTM ignores the timespans argument. Labels count raw future events, not the number of bin edges.

## Training and evaluation

Each branch splits chronologically at 70% and 80% of its own input representation. Labels use events restricted to the relevant split. Training windows are shuffled, but event order inside each window remains intact. The two branches therefore do not necessarily share temporal split boundaries or query points.

Both use Adam, learning rate 0.005, 20 epochs, batch size 64, and BCEWithLogitsLoss. Gradients update model weights during training. Evaluation uses `eval()` and `no_grad()`, sigmoid probabilities and a strict `> 0.5` decision threshold. ROC AUC uses scores, not binary predictions. Single-class targets trigger fallback AUC 0.5 and F1 0 in the current implementation.

Each dataset trains a separate pair of models. Hidden states reset for every sequence. The code does not maintain an online state across an entire workload or take scheduling actions.

## Outputs and provenance

`main.py` saves the results summary plus AUC and latency plots. `generate_figures.py` performs another training run to save histories, ROC curves, class balance, training curves, a timeline and model weights. These outputs are related evidence, not guaranteed to originate from identical model weights.

The repository README lists remaining evaluation limitations. In particular, complete future horizons and common query timestamps are necessary before a stronger controlled comparison.
