"""Analysis III/IV/V on TIGER + Product (full view): what does a memory token remember?

Post-hoc, CPU-only.  Reads three frozen artifacts and writes, per K and per *comparison space*:

  (a) Analysis III -- same-token vs different-token future-behaviour similarity,
  (b) Analysis IV  -- token-level future profiles projected to two dimensions,
  (c) Analysis V   -- token re-use statistics and their relation to accuracy.

Inputs
  per-row memory token   output/model/phi/<ds>/<task>/k<K>_r<R>/tokens/test.pt
  per-row future window  output/model/tiger/data/<ds>/<task>/full/test.pt      (target_width = 10)
  per-row evaluation     output/model/tiger/<ds>/<task>/phi_full_r<R>_k<K>/seed2026/eval/per_sample.parquet
  (every path resolves inside the project; set ``DEYIPHI_EXTRA_ROOT`` to also look in a second
  tree, e.g. a scratch copy of ``output/`` that lives elsewhere)

Why three spaces: with a 1.1M-item catalog and only ten future items per user, the item-level
distributions almost never overlap, so the raw JS/cosine is degenerate (we keep it for
transparency).  The head-restricted item space and the level-0 semantic code give the comparison
support to be meaningful; the level-0 space is also the coarse "category" proxy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SEED = 2026
PAIRS = 100_000
SCORED = 20_000
HEAD = 1000
# Optional second tree to resolve artifacts against, e.g. a scratch copy of ``output/`` that was
# kept elsewhere.  Unset by default: everything resolves under the current working directory.
EXTRA_ROOT = Path(os.environ["DEYIPHI_EXTRA_ROOT"]) if os.environ.get("DEYIPHI_EXTRA_ROOT") else None


def resolve(relative):
    candidates = [Path(relative)]
    if EXTRA_ROOT is not None:
        stripped = relative.split("output/", 1)[1] if relative.startswith("output/") else relative
        candidates.append(EXTRA_ROOT / stripped)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("%s (looked in %s)" % (relative, ", ".join(str(p) for p in candidates)))


def js_distance(matrix):
    """Pairwise Jensen-Shannon *distance* between token-level profiles (rows of ``matrix``)."""
    n = matrix.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        a = matrix[i]
        for j in range(i + 1, n):
            b = matrix[j]
            m = 0.5 * (a + b)
            mask = m > 0
            pa, pb, pm = a[mask], b[mask], m[mask]
            kl_a = np.sum(np.where(pa > 0, pa * np.log2(pa / pm), 0.0))
            kl_b = np.sum(np.where(pb > 0, pb * np.log2(pb / pm), 0.0))
            out[i, j] = out[j, i] = float(np.sqrt(max(0.5 * (kl_a + kl_b), 0.0)))
    return out


def load_tokens(path):
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    codes, mask, conf = d["codes"].numpy(), d["mask"].numpy().astype(bool), d["confidence"].numpy()
    rows = d["source_row_idx"].numpy()
    primary = np.full(len(rows), -1, dtype=np.int64)
    for i in range(len(rows)):
        valid = np.flatnonzero(mask[i])
        if valid.size:                      # most confident valid slot == the pre-fill rule
            primary[i] = int(codes[i][valid[np.argmax(conf[i][valid])]])
    return {"rows": rows, "primary": primary, "vocabulary": int(d["vocabulary"]),
            "slots": int(codes.shape[1])}


def load_targets(path):
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    digits = d["target_digits"].numpy()
    keep = (d["target_mask"].numpy().astype(bool) if "target_mask" in d
            else np.ones(digits.shape[:2], dtype=bool))
    return {int(row): [tuple(int(x) for x in digits[i][j])
                       for j in range(digits.shape[1]) if keep[i][j]]
            for i, row in enumerate(d["source_row_idx"].numpy())}


def weights(items):
    counts = Counter(items)
    total = float(sum(counts.values())) or 1.0
    return {k: v / total for k, v in counts.items()}


def cosine(a, b):
    shared = set(a) & set(b)
    num = sum(a[k] * b[k] for k in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def js_similarity(a, b):
    keys = set(a) | set(b)
    m = {k: 0.5 * (a.get(k, 0.0) + b.get(k, 0.0)) for k in keys}
    def kl(p, q):
        return sum(v * math.log(v / q[k], 2) for k, v in p.items() if v > 0 and q.get(k, 0) > 0)
    return 1.0 - 0.5 * (kl(a, m) + kl(b, m))


def sample_pairs(primary, rng, same, n_pairs):
    groups = defaultdict(list)
    for i in np.flatnonzero(primary >= 0):
        groups[primary[i]].append(i)
    if same:
        groups = {g: np.asarray(v) for g, v in groups.items() if len(v) >= 2}
    else:
        groups = {g: np.asarray(v) for g, v in groups.items()}
    keys = list(groups)
    w = np.array([len(groups[g]) * (len(groups[g]) - 1) if same else len(groups[g]) for g in keys], float)
    w /= w.sum()
    out = []
    while len(out) < n_pairs:
        g = keys[rng.choice(len(keys), p=w)]
        pool = groups[g]
        i = int(pool[rng.integers(len(pool))])
        if same:
            j = int(pool[rng.integers(len(pool))])
            if j == i:
                continue
        else:
            h = g
            while h == g:
                h = keys[rng.choice(len(keys), p=w)]
            other = groups[h]
            j = int(other[rng.integers(len(other))])
        out.append((i, j))
    return out


def profile_map(primary, rows, weights_by_row, keyspace):
    sums, counts = {}, Counter()
    for i, g in enumerate(primary):
        row = int(rows[i])
        if g < 0 or row not in weights_by_row:
            continue
        bucket = sums.setdefault(int(g), defaultdict(float))
        for key, value in weights_by_row[row].items():
            bucket[key] += value
        counts[int(g)] += 1
    tokens = sorted(counts)
    items = sorted({k for g in tokens for k in sums[g]})
    index = {k: n for n, k in enumerate(items)}
    matrix = np.zeros((len(tokens), len(items)), dtype=np.float32)
    for r, g in enumerate(tokens):
        for key, value in sums[g].items():
            matrix[r, index[key]] = value / counts[g]
    gram = matrix @ matrix.T
    eig, vec = np.linalg.eigh(gram.astype(np.float64))
    order = np.argsort(eig)[::-1][:2]
    coords = vec[:, order] * np.sqrt(np.maximum(eig[order], 0))
    explained = [float(eig[o] / max(eig.sum(), 1e-12)) for o in order]
    out = []
    for r, g in enumerate(tokens):
        p = matrix[r]
        nz = p[p > 0]
        out.append({"token": int(g), "users": int(counts[g]),
                    "entropy": float(-(nz * np.log(nz)).sum()),
                    "top_key": repr(max(index, key=lambda k: p[index[k]])),
                    "x": float(coords[r, 0]), "y": float(coords[r, 1])})
    return pd.DataFrame(out), explained, matrix


def run(dataset, task, k, recent, out_root, seed=SEED):
    tokens = load_tokens(resolve("output/model/phi/%s/%s/k%d_r%d/tokens/test.pt" % (dataset, task, k, recent)))
    targets = load_targets(resolve("output/model/tiger/data/%s/%s/full/test.pt" % (dataset, task)))
    sample = pd.read_parquet(resolve("output/model/tiger/%s/%s/phi_full_r%d_k%d/seed2026/eval/per_sample.parquet"
                                     % (dataset, task, recent, k)))
    rows = tokens["rows"]

    freq = Counter(item for items in targets.values() for item in items)
    head = {item for item, _ in freq.most_common(HEAD)}
    spaces = {
        "item_full": {r: weights(items) for r, items in targets.items()},
        "item_head": {r: weights([i for i in items if i in head]) for r, items in targets.items()},
        "code_level0": {r: weights([(i[0],) for i in items]) for r, items in targets.items()},
    }
    per_user = sample.set_index("source_row_idx")
    token_perf = defaultdict(list)
    for i, row in enumerate(rows):
        g = int(tokens["primary"][i])
        if g >= 0 and int(row) in per_user.index:
            token_perf[g].append(float(per_user.loc[int(row), "sid/pass@32"]))

    out = Path(out_root) / ("%s_%s" % (dataset, task)) / ("k%d" % k)
    report = {"dataset": dataset, "task": task, "mode": "full", "K": int(k), "recent": int(recent),
              "vocabulary": tokens["vocabulary"], "slots": tokens["slots"], "rows": int(len(rows)),
              "rows_with_token": int((tokens["primary"] >= 0).sum()), "future_window": 10,
              "metric_basis": "sid", "spaces": {}}
    for name, w in spaces.items():
        rng = np.random.default_rng(seed)
        stats, pairs_out = {}, {}
        for same in (True, False):
            pairs = sample_pairs(tokens["primary"], rng, same, PAIRS)
            cos, js = [], []
            for i, j in pairs[:SCORED]:
                a, b = w.get(int(rows[i])), w.get(int(rows[j]))
                if not a or not b:
                    continue
                cos.append(cosine(a, b)); js.append(js_similarity(a, b))
            tag = "same" if same else "diff"
            stats[tag] = {"cosine_mean": float(np.mean(cos)), "cosine_std": float(np.std(cos)),
                          "js_mean": float(np.mean(js)), "js_std": float(np.std(js)),
                          "scored": len(cos)}
            pairs_out[tag] = {"cosine": np.asarray(cos, np.float32), "js": np.asarray(js, np.float32)}
        stats["gap"] = {"cosine": stats["same"]["cosine_mean"] - stats["diff"]["cosine_mean"],
                        "js": stats["same"]["js_mean"] - stats["diff"]["js_mean"]}
        token_map, explained, matrix = profile_map(tokens["primary"], rows, w, name)
        token_map["mean_pass@32"] = token_map["token"].map(
            lambda g: float(np.mean(token_perf[g])) if token_perf.get(g) else float("nan"))
        (out / name).mkdir(parents=True, exist_ok=True)
        token_map.to_parquet(out / name / "token_map.parquet")
        np.save(out / name / "js_distance.npy", js_distance(matrix))
        np.savez_compressed(out / name / "pairs.npz",
                            **{("%s_%s" % (t, m)): v for t, d in pairs_out.items() for m, v in d.items()})
        report["spaces"][name] = {"similarity": stats, "explained_variance": explained,
                                  "tokens_with_users": int(len(token_map)),
                                  "entropy_mean": float(token_map["entropy"].mean()),
                                  "mean_users_per_token": float(token_map["users"].mean()),
                                  "tokens_with_one_user": int((token_map["users"] == 1).sum())}
        print("[%-12s] K=%-2d same=%.4f diff=%.4f gap=%.4f | JS same=%.4f diff=%.4f gap=%.4f | tokens=%d"
              % (name, k, stats["same"]["cosine_mean"], stats["diff"]["cosine_mean"],
                 stats["gap"]["cosine"], stats["same"]["js_mean"], stats["diff"]["js_mean"],
                 stats["gap"]["js"], len(token_map)), flush=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description="Analysis III/IV/V on a frozen phi vocabulary")
    ap.add_argument("--dataset", default="single_channel")
    ap.add_argument("--task", default="product")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--recent", type=int, default=16)
    ap.add_argument("--out", default="output/model/analysis/memory_token")
    args = ap.parse_args()
    for k in args.ks:
        run(args.dataset, args.task, k, args.recent, args.out)


if __name__ == "__main__":
    main()
