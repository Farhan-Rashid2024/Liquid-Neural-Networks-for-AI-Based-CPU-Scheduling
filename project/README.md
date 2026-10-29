# Experiment implementation

Start with the [repository README](../README.md) for setup, dataset acquisition, saved results and limitations.

Run scripts from this directory so relative `data/`, `figures/` and `models/` outputs go to the expected locations.

| Command | Behavior |
| --- | --- |
| `python main.py` | Generates 20,000-second synthetic workloads, optionally loads Alibaba, trains models and overwrites summary CSV and two headline plots |
| `python generate_figures.py` | Retrains models, writes four additional plots, training histories and six weights when Alibaba is available |
| `python synthetic_data.py` | Writes synthetic CSVs using the standalone script's 10,000-second configuration, which differs from the main experiment |

`config.py` resolves the repository-relative Alibaba path or the `ALIBABA_CSV` environment override. The synthetic DataFrames in `main.py` remain in memory; that entry point does not write synthetic CSVs.

Saved `.pt` files contain state dictionaries. Recreate the matching `CfCModel(2, 32)` or `LSTMModel(1, 71)` before loading weights. The `.pt` files do not include preprocessing or a serialized complete application.
