"""Plotting helpers: degradation curves, parity bars, co-occurrence."""
from __future__ import annotations

import pandas as pd


def plot_degradation_curve(csv_path: str, out_path: str) -> str:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=df, x="condition", y="success_rate", hue="agent",
                 marker="o", ax=ax)
    ax.set_title("Routing success vs. sensing-fault condition")
    ax.set_ylabel("Success rate")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_false_trust(csv_path: str, out_path: str) -> str:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=df, x="condition", y="false_trust_rate", hue="agent", ax=ax)
    ax.set_title("False-trust rate by condition")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
