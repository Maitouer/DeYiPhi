"""Export the Analysis III ("predictive equivalence") plotting package for one task.

Mirrors the shared data-package template:

    README.txt              what each file is, how it maps to the panels
    token_consistency.csv   panel (a): one row per memory token,
                            x = inter-token future similarity, y = intra-token future similarity
    pair_similarity.csv     panel (b): same-token vs different-token similarity samples
    plot_config.json        plot semantics / style

The underlying statistic is the one behind Figure 1 of the memory-token analysis: every row's
future window becomes a distribution over a comparison space (level-0 TIGER code / head items /
full catalog), and two rows are compared with cosine or ``1 - JS``.

  inter-token  similarity of a token's mean future profile to the *other* tokens' profiles
               (low = the token occupies a distinctive region of the space)
  intra-token  similarity among the future profiles of the *users of that same token*
               (high = the token groups behaviourally coherent users)

Points above the ``y = x`` diagonal are tokens whose members resemble each other more than they
resemble the rest of the population, i.e. tokens that are predictively equivalent.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.phi.token_semantics import (PAIRS, SCORED, SEED, cosine, js_similarity, load_targets,
                                           load_tokens, profile_map, resolve, sample_pairs, weights)

METRICS = {"cosine": cosine, "js": js_similarity}
PER_TOKEN_PAIRS = 2000


def inter_token(matrix, metric):
    """Mean similarity of each token's mean profile to every other token's profile."""
    if metric == "js":
        out = np.full(len(matrix), np.nan)
        for i in range(len(matrix)):
            vals = [js_similarity(dict(enumerate(matrix[i])), dict(enumerate(matrix[j])))
                    for j in range(len(matrix)) if j != i]
            out[i] = float(np.mean(vals)) if vals else np.nan
        return out
    norm = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    gram = norm @ norm.T
    np.fill_diagonal(gram, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(gram, axis=1)


def intra_token(primary, rows, w, metric_fn, rng, cap=PER_TOKEN_PAIRS):
    """Per token: sampled mean similarity among the future profiles of its own users."""
    members = defaultdict(list)
    for i, g in enumerate(primary):
        if g >= 0:
            members[int(g)].append(i)
    out = {}
    for g, idx in members.items():
        n = len(idx)
        if n < 2:
            out[g] = (float("nan"), 0, n)
            continue
        total = n * (n - 1) // 2
        if total <= cap:
            pairs = [(idx[a], idx[b]) for a in range(n) for b in range(a + 1, n)]
        else:
            seen = set()
            while len(seen) < cap:
                a, b = int(rng.integers(n)), int(rng.integers(n))
                if a == b:
                    continue
                seen.add((min(a, b), max(a, b)))
            pairs = [(idx[a], idx[b]) for a, b in seen]
        vals = []
        for i, j in pairs:
            a, b = w.get(int(rows[i])), w.get(int(rows[j]))
            if not a or not b:
                continue
            vals.append(metric_fn(a, b))
        out[g] = (float(np.mean(vals)) if vals else float("nan"), len(vals), n)
    return out


def export(dataset, task, k, recent, spaces, metric, out_root):
    metric_fn = METRICS[metric]
    tokens = load_tokens(resolve("output/model/phi/%s/%s/k%d_r%d/tokens/test.pt"
                                 % (dataset, task, k, recent)))
    targets = load_targets(resolve("output/model/tiger/data/%s/%s/full/test.pt" % (dataset, task)))
    rows = tokens["rows"]
    primary = tokens["primary"]

    head = {item for item, _ in
            Counter(i for v in targets.values() for i in v).most_common(1000)}
    table = {
        "item_full": {r: weights(items) for r, items in targets.items()},
        "item_head": {r: weights([i for i in items if i in head]) for r, items in targets.items()},
        "code_level0": {r: weights([(i[0],) for i in items]) for r, items in targets.items()},
    }
    try:
        sample = pd.read_parquet(resolve(
            "output/model/tiger/%s/%s/phi_full_r%d_k%d/seed2026/eval/per_sample.parquet"
            % (dataset, task, recent, k)))
        per_row = sample.set_index("source_row_idx")["sid/pass@32"]
    except Exception:
        per_row = None
    token_perf = defaultdict(list)
    if per_row is not None:
        for i, row in enumerate(rows):
            g = int(primary[i])
            if g >= 0 and int(row) in per_row.index:
                token_perf[g].append(float(per_row.loc[int(row)]))

    consistency, pairs_rows = [], []
    for space in spaces:
        w = table[space]
        rng = np.random.default_rng(SEED)
        for same in (True, False):
            tag = "same_token" if same else "different_token"
            drawn = sample_pairs(primary, rng, same, PAIRS)[:SCORED]
            vals = []
            for i, j in drawn:
                a, b = w.get(int(rows[i])), w.get(int(rows[j]))
                if not a or not b:
                    continue
                vals.append(metric_fn(a, b))
            pairs_rows += [{"pair_type": tag, "future_similarity": float(v),
                            "k": k, "space": space} for v in vals]
        token_map, _, matrix = profile_map(primary, rows, w, space)
        inter = inter_token(matrix, metric)
        intra = intra_token(primary, rows, w, metric_fn, np.random.default_rng(SEED + k))
        for pos, tok in enumerate(token_map["token"].astype(int)):
            i_mean, i_n, i_users = intra.get(tok, (float("nan"), 0, 0))
            consistency.append({
                "token_id": tok,
                "inter_token_future_similarity": float(inter[pos]),
                "intra_token_future_similarity": i_mean,
                "k": k, "space": space,
                "token_users": i_users,
                "intra_pairs_scored": i_n,
                "mean_pass@32": (float(np.mean(token_perf[tok])) if token_perf.get(tok) else
                                 float("nan")),
            })
        print("[%-12s] K=%-2d tokens=%d inter=%.4f intra=%.4f" % (
            space, k, len(token_map), np.nanmean(inter), np.nanmean([v[0] for v in intra.values()])),
            flush=True)

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(consistency).to_csv(out / "token_consistency.csv", index=False)
    pd.DataFrame(pairs_rows).to_csv(out / "pair_similarity.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="single_channel")
    ap.add_argument("--task", default="short_video")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--recent", type=int, default=32)
    ap.add_argument("--spaces", nargs="+", default=["code_level0", "item_head", "item_full"])
    ap.add_argument("--metric", choices=sorted(METRICS), default="cosine")
    ap.add_argument("--out", default="output/model/analysis/memory_token/analysisIII_package")
    args = ap.parse_args()
    for k in args.ks:
        print(export(args.dataset, args.task, k, args.recent, args.spaces, args.metric, args.out))


if __name__ == "__main__":
    main()
