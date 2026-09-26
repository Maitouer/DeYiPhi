"""Per-token future-behaviour similarity: the raw material of the *upper panel* (token level).

The global ``pairs.npz`` shipped by ``token_semantics.py`` only holds similarity *values*
(no user/token ids), so it can never be disaggregated into per-token points.  This script
re-derives them from the frozen artifacts:

    tokens/test.pt   (phi vocabulary)      -> which token each test row belongs to
    tiger .../full/test.pt (future window) -> p_u, the future distribution of that row

For every token ``g`` (token = its most confident valid slot, the same rule as the inference-time
route pre-fill), in one comparison space:

    intra_mean   mean cosine similarity over sampled pairs of *members* of g
    inter_mean   mean cosine similarity over sampled pairs (u in g, v NOT in g)

Both use up to ``--pairs`` sampled pairs per token (all pairs when the token is small).  The
reference line of the upper panel is ``y = x``: tokens above it group their members more tightly
than the outside population resembles them.

Usage (from the project root, CPU-only):

    python -m scripts.model.analysis.export_token_level_similarity \
        --task product --ks 1 2 4 8 --recent 16 --space code_level0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.phi.token_semantics import (SEED, cosine, js_similarity, load_targets,
                                           load_tokens, resolve, weights)

METRICS = {"cosine": cosine, "js": js_similarity}
HEAD = 1000


def sample_index_pairs(rng, n, cap):
    """Up to ``cap`` distinct index pairs (a < b) out of C(n, 2); exhaustive when small."""
    if n < 2:
        return []
    if n * (n - 1) // 2 <= cap:
        return [(a, b) for a in range(n) for b in range(a + 1, n)]
    seen = set()
    while len(seen) < cap:
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b:
            seen.add((min(a, b), max(a, b)))
    return sorted(seen)


def mean_similarity(metric_fn, table, rows, pairs):
    vals = []
    for i, j in pairs:
        a, b = table.get(int(rows[i])), table.get(int(rows[j]))
        if a and b:
            vals.append(metric_fn(a, b))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def build_table(targets, head, space):
    if space == "item_full":
        return {r: weights(items) for r, items in targets.items()}
    if space == "item_head":
        return {r: weights([i for i in items if i in head]) for r, items in targets.items()}
    if space == "code_level0":
        return {r: weights([(i[0],) for i in items]) for r, items in targets.items()}
    raise ValueError("unknown space %s" % space)


def token_profile_inter(table, rows, primary, metric):
    """Variant of ``inter_mean`` computed on token *mean profiles* instead of user pairs."""
    sums, counts = defaultdict(lambda: defaultdict(float)), Counter()
    keys = set()
    for i, g in enumerate(primary):
        if g < 0:
            continue
        w = table.get(int(rows[i]))
        if not w:
            continue
        for key, value in w.items():
            sums[int(g)][key] += value
            keys.add(key)
        counts[int(g)] += 1
    tokens = sorted(counts)
    index = {k: n for n, k in enumerate(sorted(keys))}
    matrix = np.zeros((len(tokens), len(index)), dtype=np.float32)
    for r, g in enumerate(tokens):
        for key, value in sums[g].items():
            matrix[r, index[key]] = value / counts[g]
    norm = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    gram = norm @ norm.T
    np.fill_diagonal(gram, np.nan)
    with np.errstate(invalid="ignore"):
        return {g: float(v) for g, v in zip(tokens, np.nanmean(gram, axis=1))}


def export(dataset, task, k, recent, space, metric, pairs_cap, out_root):
    metric_fn = METRICS[metric]
    tokens = load_tokens(resolve("output/model/phi/%s/%s/k%d_r%d/tokens/test.pt"
                                 % (dataset, task, k, recent)))
    targets = load_targets(resolve("output/model/tiger/data/%s/%s/full/test.pt" % (dataset, task)))
    rows, primary = tokens["rows"], tokens["primary"]
    positions = np.flatnonzero(primary >= 0)

    head = {item for item, _ in Counter(i for v in targets.values() for i in v).most_common(HEAD)}
    table = build_table(targets, head, space)
    members = defaultdict(list)
    for i in positions:
        members[int(primary[i])].append(int(i))

    try:
        per_sample = pd.read_parquet(resolve(
            "output/model/tiger/%s/%s/phi_full_r%d_k%d/seed2026/eval/per_sample.parquet"
            % (dataset, task, recent, k)))
        pass_by_row = per_sample.set_index("source_row_idx")["sid/pass@32"]
    except Exception:
        pass_by_row = None

    profile_inter = token_profile_inter(table, rows, primary, metric)
    rng = np.random.default_rng(SEED)
    out_rows = []
    for g in sorted(members):
        idx = np.asarray(members[g], dtype=int)
        intra_pairs = [(idx[a], idx[b]) for a, b in sample_index_pairs(rng, len(idx), pairs_cap)]
        intra_mean, intra_n = mean_similarity(metric_fn, table, rows, intra_pairs)
        outside = positions[primary[positions] != g] if len(members) > 1 else np.array([], dtype=int)
        inter_pairs = []
        if len(outside) and len(idx):
            inside_draw = idx[rng.integers(0, len(idx), pairs_cap)]
            outside_draw = outside[rng.integers(0, len(outside), pairs_cap)]
            inter_pairs = list(zip(inside_draw.tolist(), outside_draw.tolist()))
        inter_mean, inter_n = mean_similarity(metric_fn, table, rows, inter_pairs)
        users = np.asarray([int(rows[i]) for i in idx])
        perf = (float(pass_by_row.reindex(users).mean()) if pass_by_row is not None else float("nan"))
        out_rows.append({
            "token_id": g, "users": len(idx),
            "intra_mean": intra_mean, "inter_mean": inter_mean,
            "intra_pairs_scored": intra_n, "inter_pairs_scored": inter_n,
            "inter_token_profile_mean": profile_inter.get(g, float("nan")),
            "mean_pass@32": perf,
            "task": task, "K": k, "space": space, "metric": metric,
        })
    frame = pd.DataFrame(out_rows).sort_values("token_id")
    frame["difference"] = frame["intra_mean"] - frame["inter_mean"]
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["ratio"] = frame["intra_mean"] / frame["inter_mean"].replace(0.0, np.nan)

    out = Path(out_root) / ("%s_%s" % (dataset, task)) / ("k%d" % k) / space
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "token_level.csv", index=False)

    valid = frame.dropna(subset=["intra_mean", "inter_mean"])
    summary = {
        "task": task, "K": int(k), "space": space, "metric": metric,
        "tokens_with_users": int(len(frame)),
        "tokens_scored": int(len(valid)),
        "tokens_singleton": int((frame["users"] == 1).sum()),
        "pairs_cap_per_token": int(pairs_cap),
        "intra_mean_over_tokens": float(valid["intra_mean"].mean()),
        "inter_mean_over_tokens": float(valid["inter_mean"].mean()),
        "ratio_over_tokens": float(valid["intra_mean"].mean() / valid["inter_mean"].mean()),
        "users": int(frame["users"].sum()),
        "fraction_tokens_above_diagonal": float((valid["intra_mean"] > valid["inter_mean"]).mean()),
        "fraction_users_above_diagonal": float(
            valid.loc[valid["intra_mean"] > valid["inter_mean"], "users"].sum() / valid["users"].sum()),
        "user_weighted_intra_mean": float(np.average(valid["intra_mean"], weights=valid["users"])),
        "user_weighted_inter_mean": float(np.average(valid["inter_mean"], weights=valid["users"])),
    }
    (out / "token_level_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                                 encoding="utf-8")
    print("[%s K=%d %s] tokens=%d scored=%d intra=%.4f inter=%.4f ratio=%.2f above=%.1f%%" % (
        task, k, space, len(frame), len(valid), summary["intra_mean_over_tokens"],
        summary["inter_mean_over_tokens"], summary["ratio_over_tokens"],
        100 * summary["fraction_tokens_above_diagonal"]), flush=True)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="single_channel")
    ap.add_argument("--task", default="product")
    ap.add_argument("--ks", type=int, nargs="+", default=[1])
    ap.add_argument("--recent", type=int, default=None)
    ap.add_argument("--space", default="code_level0",
                    choices=["code_level0", "item_head", "item_full"])
    ap.add_argument("--metric", choices=sorted(METRICS), default="cosine")
    ap.add_argument("--pairs", type=int, default=2000, help="max sampled pairs per token per side")
    ap.add_argument("--out", default="output/model/analysis/memory_token")
    args = ap.parse_args()
    recent = args.recent if args.recent is not None else (16 if args.task == "product" else 32)
    for k in args.ks:
        export(args.dataset, args.task, k, recent, args.space, args.metric, args.pairs, args.out)


if __name__ == "__main__":
    main()
