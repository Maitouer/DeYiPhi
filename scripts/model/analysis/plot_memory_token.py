"""Figures for Analysis III/IV/V (TIGER + Product, full view, K = 1/2/4/8)."""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

ROOT = Path(os.environ.get("MT_ROOT", "output/model/analysis/memory_token/single_channel_product"))
OUT = Path(os.environ.get("MT_OUT", "output/model/analysis/memory_token/figures"))
KS = tuple(int(x) for x in os.environ.get("MT_KS", "1 2 4 8").split())
SPACES = [("code_level0", "Level-0 semantic code", "$8{,}192$ classes"),
          ("item_head", "Head items", "top $1{,}000$"),
          ("item_full", "Full catalog", "$1.13$M items")]
SAME, DIFF = "#D55E00", "#0072B2"
KCOL = {1: "#332288", 2: "#88CCEE", 4: "#DDCC77", 8: "#CC6677"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.linewidth": 0.8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.top": True, "ytick.right": True, "legend.frameon": False,
})


def pairs(k, space):
    return np.load(ROOT / ("k%d" % k) / space / "pairs.npz")


def token_map(k, space):
    return pd.read_parquet(ROOT / ("k%d" % k) / space / "token_map.parquet")


def fig1():
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), height_ratios=[1.3, 1.0])
    for col, (space, title, sub) in enumerate(SPACES):
        ax = axes[0, col]
        width = 0.36
        for ci, (key, colour, label) in enumerate((("same_cosine", SAME, "same token"),
                                                   ("diff_cosine", DIFF, "different token"))):
            means, errs = [], []
            for k in KS:
                v = pairs(k, space)[key]
                means.append(v.mean())
                errs.append(v.std(ddof=1) / np.sqrt(len(v)))
            xs = np.arange(len(KS)) + (ci - 0.5) * (width + 0.04)
            ax.bar(xs, means, width=width, color=colour, label=label, yerr=errs,
                   error_kw=dict(linewidth=0.7, capsize=1.6))
        for gi, k in enumerate(KS):
            s_m = pairs(k, space)["same_cosine"].mean()
            d_m = pairs(k, space)["diff_cosine"].mean()
            ax.annotate("%.1f$\\times$" % (s_m / max(d_m, 1e-12)), xy=(gi, s_m), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=6.5, color="0.25")
        ax.set_xticks(range(len(KS)))
        ax.set_xticklabels(["$K{=}%d$" % k for k in KS])
        ax.set_title("%s (%s)" % (title, sub))
        ax.set_ylim(0, max(pairs(k, space)["same_cosine"].mean() for k in KS) * 1.42)
        if col == 0:
            ax.set_ylabel("mean future-behaviour\ncosine over sampled pairs")
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))
        ax = axes[1, col]
        for k, colour in ((KS[0], KCOL.get(KS[0], SAME)), (KS[-1], KCOL.get(KS[-1], DIFF))):
            z = pairs(k, space)
            for key, style, name in (("same_cosine", "-", "same"), ("diff_cosine", "--", "diff")):
                v = np.sort(z[key])
                ax.plot(v, 1.0 - np.arange(1, len(v) + 1) / len(v), style, color=colour,
                        linewidth=1.0, label="$K{=}%d$, %s" % (k, name))
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.3)
        ax.set_xlim(0, max(np.percentile(pairs(KS[0], space)["same_cosine"], 95), 1e-4))
        ax.set_xlabel("future-behaviour cosine")
        if col == 0:
            ax.set_ylabel("1 $-$ ECDF")
            ax.legend(loc="upper right")
    fig.tight_layout(pad=0.4, h_pad=0.7)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / ("fig1_token_consistency.%s" % suffix), dpi=300)


def fig2(space="code_level0", perplexity=15):
    ncols = min(2, len(KS))
    nrows = int(np.ceil(len(KS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.7 * nrows), squeeze=False)
    for ax, k in zip(axes.ravel(), KS):
        tm = token_map(k, space)
        dist = np.load(ROOT / ("k%d" % k) / space / "js_distance.npy")
        if not np.isfinite(dist).all():
            dist = np.nan_to_num(dist, nan=0.0, posinf=dist[np.isfinite(dist)].max())
        emb = TSNE(n_components=2, metric="precomputed", perplexity=perplexity,
                   init="random", learning_rate="auto", random_state=0).fit_transform(dist)
        sizes = 6 + 26 * np.sqrt(tm["users"].values / tm["users"].max())
        lo_h, hi_h = np.percentile(tm["entropy"], [5, 95])
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=tm["entropy"], s=sizes, cmap="viridis",
                        vmin=lo_h, vmax=hi_h, edgecolor="black", linewidth=0.25, alpha=0.9)
        ax.set_title("$K{=}%d$  (%d tokens, %.0f users/token)" % (k, len(tm), tm["users"].mean()))
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label("future entropy", fontsize=7)
    fig.tight_layout(pad=0.4)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / ("fig2_predictive_memory_map.%s" % suffix), dpi=300)


def fig3(space="code_level0"):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for k in KS:
        tm = token_map(k, space)
        counts = np.sort(tm["users"].values)[::-1]
        axes[0].plot(np.arange(1, len(counts) + 1), counts, color=KCOL[k], linewidth=1.1,
                     label="$K{=}%d$  (%d tokens)" % (k, len(counts)))
        v = tm.dropna(subset=["mean_pass@32"])
        axes[1].scatter(v["users"], v["mean_pass@32"], s=7, color=KCOL[k], alpha=0.55,
                        edgecolor="none", label="$K{=}%d$" % k)
        if len(v) > 10:
            bins = np.linspace(v["users"].min(), v["users"].max(), 8)
            idx = np.digitize(v["users"], bins)
            xs = [v["users"][idx == b].mean() for b in range(1, len(bins)) if (idx == b).any()]
            ys = [v["mean_pass@32"][idx == b].mean() for b in range(1, len(bins)) if (idx == b).any()]
            axes[1].plot(xs, ys, color=KCOL[k], linewidth=1.3)
    axes[0].set_xlabel("token rank"); axes[0].set_ylabel("users per token")
    axes[0].set_yscale("log"); axes[0].legend(loc="upper right")
    axes[0].set_title("Vocabulary utilisation")
    axes[1].set_xlabel("users per token"); axes[1].set_ylabel("Pass@32 (per user)")
    axes[1].legend(loc="lower right")
    axes[1].set_title("Reuse vs accuracy")
    fig.tight_layout(pad=0.4)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / ("fig3_token_reuse.%s" % suffix), dpi=300)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3()
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
