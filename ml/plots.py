import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#1f1f1e"
MUTED = "#6b6a63"
GRID = "#e4e3dc"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

TITLES = {"centralized": "Centralized", "fedavg": "FL (FedAvg)", "fedavg_he": "FL (FedAvg) + HE"}


def confusion_panels(mats, path):
    fig, axes = plt.subplots(1, len(mats), figsize=(7.16, 2.7), constrained_layout=True)
    vmax = max(m.max() for m in mats.values())
    for ax, (name, m) in zip(axes, mats.items()):
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j]:
                    ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if m[i, j] > 0.6 * vmax else INK)
        ax.set_title(TITLES.get(name, name), fontsize=9, color=INK)
        ax.set_xlabel("Predicted user")
        ax.set_ylabel("True user")
        ax.set_xticks(range(m.shape[0]))
        ax.set_yticks(range(m.shape[0]))
        ax.tick_params(length=0, labelsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)


def bar_with_error(summary, path):
    metrics = [("accuracy", "Accuracy"), ("macro_recall", "Recall"), ("macro_precision", "Precision")]
    names = list(summary)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    for i, (key, label) in enumerate(metrics):
        means = [100 * summary[n][key]["mean"] for n in names]
        stds = [100 * summary[n][key]["std"] for n in names]
        ax.errorbar(x + (i - 1) * 0.18, means, yerr=stds, fmt="o", markersize=5, color=SERIES[i], ecolor=SERIES[i],
                    elinewidth=1.2, capsize=2.5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([TITLES[n].replace(" + ", "\n+ ") for n in names])
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel("Score (%)", color=INK, fontsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)


def curves(series, ylabel, path):
    fig, ax = plt.subplots(figsize=(3.5, 2.3), constrained_layout=True)
    for i, (label, ys) in enumerate(series.items()):
        ax.plot(np.arange(1, len(ys) + 1), ys, color=SERIES[i % len(SERIES)], linewidth=1.5, label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)


def roc(curves_by_model, path):
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    for i, (label, (far, frr)) in enumerate(curves_by_model.items()):
        ax.plot(np.maximum(far, 1e-5), 1 - frr, color=SERIES[i], linewidth=1.5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("False acceptance rate")
    ax.set_ylabel("Genuine acceptance rate")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
