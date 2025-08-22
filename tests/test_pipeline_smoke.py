#test_pipeline_smoke.py
import os
import sys
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def run(cmd):
    print("RUN:", cmd)
    completed = subprocess.run(cmd, cwd=REPO, shell=True, capture_output=True, text=True)
    print(completed.stdout)
    print(completed.stderr)
    assert completed.returncode == 0, f"Command failed: {cmd}\nSTDERR:\n{completed.stderr}"

def test_pipeline_filemode_smoke(synthetic_parquet):
    # Ensure python can import the repo modules
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)

    # Run the pipeline (file-mode = no network)
    cmd = "python -m scripts.run_pipeline --config configs/base.yaml --file-mode --k 3"
    completed = subprocess.run(cmd, cwd=REPO, shell=True, capture_output=True, text=True, env=env)
    print(completed.stdout)
    print(completed.stderr)
    assert completed.returncode == 0

    # Check that outputs exist
    vec_any = list((REPO / "outputs" / "backtests").glob("vec_k3_thr*.parquet"))
    ex_any = list((REPO / "outputs" / "backtests").glob("exact_k3_thr*.parquet"))
    assert len(vec_any) >= 1, "Vectorized backtest parquet not found"
    assert len(ex_any) >= 1, "Exact backtest parquet not found"
