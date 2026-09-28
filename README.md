# Sensing-Fault-Aware RL Routing for MEDA Biochips

Shows that an RL router which **distrusts noisy/spoofed sensing** outperforms a
router that treats the sensing map as ground truth (the implicit assumption in
Elfar et al. 2023 and follow-ups), under realistic sensing-fault conditions.
Fully digital: array, droplets, faults, and sensors are all simulated.

## Claim

> Prior RL routing work is fault-aware for the fluidic layer but assumes a
> fault-free sensing layer; we relax that assumption with a learned
> residual-based confidence signal.

| Baseline (Elfar-style) | This work |
|---|---|
| Sensing map = ground truth | Explicit confidence signal, learned trust |
| Degrades silently under sensing faults | Graceful degradation, measured |
| Fluidic-fault-aware only | Cross-layer: fluidic + sensing, incl. co-occurrence |

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -q                                   # unit tests
python training/train_baseline.py \                  # phase A+B, baseline
  --episodes-clean 50 --episodes-faulty 50
python training/train_confidence_aware.py \          # phase A+B, aware
  --episodes-clean 50 --episodes-faulty 50
python evaluation/run_eval.py --episodes 50          # summary CSV
python experiments/exp2_sensing_fault_sweep.py       # headline figure
```

> Full runs use the defaults in `configs/training_baseline.yaml`
> (1500 clean + 1500 faulty episodes per agent). The flags above are smoke-test
> sized.

## Repo contents (what's on GitHub)

`REPORT.pdf` is the compiled interim report. Everything else below is code;
run-generated outputs (`results/`, checkpoints) are git-ignored and rebuilt
via the Quickstart.

```
meda-sensing-aware-rl/
├── README.md                     # this file: claim, quickstart, physics notes
├── REPORT.pdf                    # compiled interim report
├── requirements.txt              # numpy, gymnasium, torch, pyyaml, pandas,
│                                 # matplotlib, seaborn, pytest, tensorboard (+ optional SB3)
├── .gitignore                    # excludes checkpoints, TB logs, pycache, report sources
├── configs/
│   ├── array_default.yaml        # grid size, actuation threshold/wear, droplet volume,
│   │                             # rewards (+10/-0.05/-1.0/-0.1), PBRS shaping (k=0.2), windows
│   ├── fault_profiles.yaml       # clean/low/medium/high/co-occurrence profiles + 0–50% sweep
│   └── training_baseline.yaml    # shared DQN hyperparams, phase budgets, eval cadence,
│                                 # buffer-clear flag, best-checkpoint selection
├── meda_env/                     # fully digital simulator
│   ├── array.py                  # MicroCell grid: ground-truth health/wear/fault/occupancy
│   ├── droplet.py                # droplet position, volume, target
│   ├── actuation.py              # 8-directional moves (Discrete(8)); silent-fail on dead cells
│   ├── gym_env.py                # Gymnasium wrapper: reported-position obs (3ch/4ch),
│   │                             # PBRS shaping, confidence gating, per-episode RNG seeding
│   ├── renderer.py               # matplotlib debug view (truth vs reported vs confidence)
│   ├── fault_injection/
│   │   ├── fluidic_faults.py     # breakdown / stuck-at / charge-trapping (ground truth only)
│   │   ├── sensing_faults.py     # noise / spoof / dropout / delay (reported readings only)
│   │   └── fault_scheduler.py    # per-episode fault sets; exact fluidic↔sensing overlap
│   └── sensing/
│       ├── capacitive_model.py   # ground truth → reported health grid + reported position
│       └── confidence_signal.py  # residual-based reliability score, rolling window
├── agents/                       # one shared Double-DQN class (Huber loss); hyperparams identical
│   ├── dqn.py                    # replay buffer, epsilon-greedy, target net, save/load
│   ├── networks.py               # CNN encoder + MLP head (MLP fallback included)
│   ├── baseline_dqn.py           # trusts sensing map: 3ch obs, no confidence
│   └── confidence_aware_dqn.py   # learns trust: 4ch obs, +confidence signal
├── training/
│   ├── common.py                 # shared loop: rollout → replay → updates; greedy probes;
│   │                             # continuous epsilon via start_episode; best-checkpoint saving
│   ├── callbacks.py              # TensorBoard logging helper
│   ├── train_baseline.py         # phase A (clean) → buffer-clear → phase B (faulty), baseline
│   └── train_confidence_aware.py # same curriculum, confidence-aware agent
├── evaluation/
│   ├── run_eval.py               # greedy rollouts per fault profile → summary CSV;
│   │                             # fail-closed checkpoints; warmed-up decision latency
│   ├── metrics.py                # success, degradation, false-trust, steps, latency
│   └── plots.py                  # degradation curves, false-trust bars
├── experiments/
│   ├── exp1_clean_sensing.py         # parity check: both agents ~equal on clean sensing
│   ├── exp2_sensing_fault_sweep.py   # HEADLINE: success vs fault density 0–50%
│   ├── exp3_ablation_confidence_weight.py  # zero confidence at test time (reliance check)
│   └── exp4_multi_fault_cooccurrence.py    # fluidic+sensing fault on same cell
└── tests/                        # 36 tests: array legality, fault separation, env rewards,
    ├── test_array.py             #   reported-position obs, RNG lockstep, tracker gating,
    ├── test_fault_injection.py   #   exact overlap, epsilon continuity, best-ckpt, fail-closed
    ├── test_env.py
    ├── test_sensing_aware.py
    └── test_training.py
```

## Physics grounding

Array size (10×10 default), actuation voltages (~20–40 V), and droplet volumes
(0.5–4× unit) are chosen within ranges cited in MEDA literature (Elfar et al.
2023 + MEDA reviews); see `configs/array_default.yaml`. Charge trapping is a
cumulative, monotonic wear that can progress into dielectric breakdown.

## Notes

- A move into a ground-truth dead cell **silently fails** (droplet stays put),
  modelling insufficient EWOD force — not a crash.
- `stable-baselines3` is listed as optional; the default agents are hand-rolled
  PyTorch DQNs so results reproduce without it.

## Results (local only, not on GitHub)

`results/` holds checkpoints, CSVs, and figures generated by training,
`evaluation/run_eval.py`, and `experiments/exp*.py` (git-ignored except `.gitkeep`).
