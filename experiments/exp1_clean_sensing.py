"""Exp1 -- clean-sensing parity check. Both agents should perform ~equally."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.run_eval import evaluate_condition, load_yaml

array_cfg = load_yaml("configs/array_default.yaml")
fault_cfg = load_yaml("configs/fault_profiles.yaml")
hp = load_yaml("configs/training_baseline.yaml")

rows_out: list[dict] = []
evaluate_condition(array_cfg, fault_cfg["profiles"]["clean"],
                   "results/checkpoints/baseline_faulty_best.pt",
                   "results/checkpoints/aware_faulty_best.pt",
                   hp["eval"]["episodes_per_condition"], hp, rows_out, "clean")

import csv
os.makedirs("results", exist_ok=True)
with open("results/exp1_clean_sensing.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)
print("wrote results/exp1_clean_sensing.csv")
