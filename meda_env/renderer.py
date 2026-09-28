"""Optional matplotlib debug view of ground truth vs. reported sensing."""
from __future__ import annotations

import numpy as np


def render_state(array, reported_health, confidence, droplet, target=None, ax=None, title="MEDA state"):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    true_h = array.health_grid()
    ax.clear()
    ax.imshow(true_h, vmin=0.0, vmax=1.0, cmap="Greys", alpha=0.35)
    ax.imshow(reported_health, vmin=0.0, vmax=1.0, cmap="Blues", alpha=0.35)
    if droplet is not None:
        ax.scatter([droplet.position[1]], [droplet.position[0]], c="red", s=120, marker="o",
                   label="droplet")
    tgt = target if target is not None else (droplet.target if droplet else None)
    if tgt is not None:
        ax.scatter([tgt[1]], [tgt[0]], c="green", s=120, marker="*",
                   label="target")
    # low-confidence overlay
    if confidence is not None:
        low = np.argwhere(confidence < 0.5)
        if len(low):
            ax.scatter(low[:, 1], low[:, 0], c="orange", s=40, marker="x", label="low-conf")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax
