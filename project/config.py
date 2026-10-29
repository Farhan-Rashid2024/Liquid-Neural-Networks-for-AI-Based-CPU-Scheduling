"""Dataset location shared by both experiment entry points."""
import os
from pathlib import Path

DEFAULT_ALIBABA_CSV = (
    Path(__file__).resolve().parents[1]
    / "datasets/alibaba_cluster/cluster-trace-gpu-v2020/simulator/traces/pai"
    / "pai_job_duration_estimate_100K.csv"
)
ALIBABA_CSV = Path(os.environ.get("ALIBABA_CSV", str(DEFAULT_ALIBABA_CSV))).expanduser()
