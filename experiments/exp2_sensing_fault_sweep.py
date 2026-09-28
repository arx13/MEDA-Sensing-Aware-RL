"""Exp2 -- sensing-fault density sweep 0-50%. Core result: degradation curves."""
from __future__ import annotations

import copy
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.run_eval import evaluate_condition, load_yaml

array_cfg = load_yaml("configs/array_default.yaml")
fault_cfg = load_yaml("configs/fault_profiles.yaml")
hp = load_yaml("configs/training_baseline.yaml")

base = fault_cfg["profiles"]["sensing_medium"]
rows_out: list[dict] = []
for density in fault_cfg["sweep"]["densities"]:
    profile = copy.deepcopy(base)
    profile["sensing_fault_density"] = density
    evaluate_condition(array_cfg, profile,
                       "results/checkpoints/baseline_faulty_best.pt",
                       "results/checkpoints/aware_faulty_best.pt",
                       hp["eval"]["episodes_per_condition"], hp, rows_out,
                       f"density_{density:.2f}")

os.makedirs("results", exist_ok=True)
with open("results/exp2_sensing_fault_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)
print("wrote results/exp2_sensing_fault_sweep.csv")

from evaluation.plots import plot_degradation_curve, plot_false_trust
plot_degradation_curve("results/exp2_sensing_fault_sweep.csv",
                       "results/exp2_degradation_curve.png")
plot_false_trust("results/exp2_sensing_fault_sweep.csv",
                 "results/exp2_false_trust.png")
print("wrote figures")
