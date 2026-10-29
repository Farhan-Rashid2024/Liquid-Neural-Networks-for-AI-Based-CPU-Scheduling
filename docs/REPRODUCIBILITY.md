# Reproduction and output provenance

## Environment

Install `project/requirements.txt` in an isolated environment. `ncps` is pinned to 0.0.2; most other dependencies use minimum versions, so this is not a full environment lock. CPU/GPU kernels and package versions can affect training results and timing. Random seed 42 and deterministic cuDNN settings improve repeatability but do not guarantee identical results across machines.

## External input

Download the derived `pai_job_duration_estimate_100K.csv` from the [Alibaba clusterdata v2020 simulator traces](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020/simulator/traces/pai). Follow upstream dataset terms and attribution. No third-party dataset is included in this repository.

Either reproduce the directory structure documented in the root README or set `ALIBABA_CSV` to an absolute file path. A readable CSV with `submit_time` is required. The loaders sort and retain 20,000 records. They catch load errors and continue with synthetic data, so inspect console output to confirm that all three datasets ran.

## Main run

From `project/`, run `python main.py`. This trains six models when Alibaba is available, prints class balance and metrics, and writes:

- `data/results_summary.csv`
- `figures/auc_comparison.png`
- `figures/inference_time.png`

The synthetic workloads span 20,000 simulated seconds; generation does not wait in real time. `main.py` does not save synthetic CSVs or model weights. The standalone generator writes CSVs with different 10,000-second settings and is not an exact substitute for the main configuration.

## Diagnostic run

Run `python generate_figures.py` from `project/`. This retrains, writes training histories and diagnostic figures, and saves weights under `models/`. It also generates a local `NEW_FIGURES.md` report. Re-running may overwrite previous artifacts. Preserve original evidence before reproducing experiments.

## Timing

`main.py` moves models and example test batches to CPU and times repeated forward passes using `time.perf_counter()`. It averages 100 repetitions after 10 warmups. Batch size is up to 64 sequences. It excludes training, preprocessing, file access and scheduler integration. The Python CfC loop versus optimized LSTM implementation is a confound.

## Saved evidence

The initial repository upload includes existing local result artifacts. This publication did not rerun the full training pipeline. The saved values document the earlier run, not results from the publication environment. The updated PDF reproduces the CSV's headline values. Diagnostic curves and weights come from a separate figure-generation run.

The earlier paper remains a draft, with stronger claims than the current evidence supports. The root README and architecture documentation state the implemented scope and known limitations.
