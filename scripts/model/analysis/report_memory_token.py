"""Generate MEMORY_TOKEN_ANALYSIS.md: every number behind the three figures.

Reads the outputs of ``src/model/phi/token_semantics.py`` (per K, per comparison space) and writes
a markdown report next to the figures.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(os.environ.get("MT_ROOT", "output/model/analysis/memory_token/single_channel_product"))
OUT = Path(os.environ.get("MT_DOC", "MEMORY_TOKEN_ANALYSIS.md"))
# Project root; override with ``DEYIPHI_ROOT`` when running from outside the repository.
PROJECT = Path(os.environ.get("DEYIPHI_ROOT", "."))
KS = tuple(int(x) for x in os.environ.get("MT_KS", "1 2 4 8").split())
TASK = os.environ.get("MT_TASK", "product")
RECENT = int(os.environ.get("MT_RECENT", "16" if TASK == "product" else "32"))
SPACES = [("code_level0", "Level-0 semantic code (8,192 classes)"),
          ("item_head", "Head items (top-1,000)"),
          ("item_full", "Full catalog (1.13M items)")]
def arm(name, variant, k):
    """Run-directory suffix for one arm: DeYi and phi differ only by the variant prefix."""
    prefix = "deyi" if variant.lower() == "deyi" else "phi"
    return (name, "%s_full_r%d_k%d" % (prefix, RECENT, k))


ARMS = [(("%s full K=%d" % (v, k)), arm(v, v, k)[1]) for k in KS for v in ("DeYi", "phi")]


def summary(k):
    return json.loads((ROOT / ("k%d" % k) / "summary.json").read_text())


def pairs(k, space):
    return np.load(ROOT / ("k%d" % k) / space / "pairs.npz")


def token_map(k, space):
    return pd.read_parquet(ROOT / ("k%d" % k) / space / "token_map.parquet")


def fmt(value, digits=4):
    return ("%%.%df" % digits) % value


def main():
    lines = []
    add = lines.append
    rep1 = summary(1)
    add("# Memory-token analysis: data behind the figures")
    add("")
    add("Frozen-artifact, CPU-only post-hoc analysis of **TIGER + Product, full history view**, for "
        "`K = 1, 2, 4, 8`. Nothing here retrains a model or runs inference: every number is a "
        "statistic over artifacts that already existed.")
    add("")
    add("## 0. Summary")
    add("")
    add("- **Analysis III** — users that share a memory token have systematically more similar "
        "*future behaviour* than users with different tokens: the same/different ratio is "
        "**1.9–2.4×** in every comparison space at every K (t ≈ 20–25 over 20,000 sampled pairs), "
        "and the gap is largest at **K = 1**.")
    add("- **Analysis IV** — the token-level future-item profiles form a structured map when "
        "embedded by their Jensen–Shannon distance (Figure 2), with 184–211 of the 256 vocabulary "
        "entries actually used.")
    add("- **Analysis V** — the vocabulary is heavily reused (mean 89–102 users per token) and "
        "reuse correlates positively with per-user Pass@32 (Spearman +0.26 … +0.38).")
    add("")
    add("## 1. Inputs (all pre-existing artifacts)")
    add("")
    add("| What | Path | Fields used |")
    add("|---|---|---|")
    add("| per-row memory token (frozen φ vocabulary) | `output/model/phi/<ds>/<task>/k<K>_r16/tokens/test.pt` "
        "| `source_row_idx`, `codes[R,K]`, `mask[R,K]`, `confidence[R,K]`, `vocabulary=256` |")
    add("| per-row future window | `output/model/tiger/data/<ds>/<task>/full/test.pt` "
        "| `target_digits[R,10,4]`, `target_mask[R,10]` (`target_width = 10` for **both** tasks) |")
    add("| per-row evaluation | `output/model/tiger/<ds>/<task>/phi_full_r16_k<K>/seed2026/eval/per_sample.parquet` "
        "| `source_row_idx`, `sid/{pass,recall,ndcg}@{1,5,10,20,32}` |")
    add("")
    add("`<ds> = single_channel`, `<task> = product`; %d test rows. Every path below is resolved "
        "relative to the project root (set `DEYIPHI_ROOT` when running from elsewhere)."
        % rep1["rows"])
    add("")
    add("## 2. Definitions")
    add("")
    add("**Memory token of a user.** The φ vocabulary is shared across K (256 entries); K is how many "
        "slots a user gets. For K > 1 a user's token is its **most confident valid slot** — the same "
        "rule the inference-time route pre-fill uses (`argmax_k confidence[r,k]` over `mask[r,k]`). "
        "Rows whose slots are all masked are dropped (here none are).")
    add("")
    add("**Future behaviour.** For row *u* the future window is the ten benchmark targets; the "
        "distribution is uniform over distinct items, "
        "`p_u(i) = #(i in Y_u) / |Y_u|`, represented in one of three *comparison spaces*:")
    add("")
    add("| space | keys of `p_u` | why |")
    add("|---|---|---|")
    add("| `item_full` | the item itself (SID 4-tuple) | the design's literal definition |")
    add("| `item_head` | item, restricted to the 1,000 most frequent future items, renormalised | "
        "keeps the item-distribution semantics but gives the comparison support |")
    add("| `code_level0` | the level-0 TIGER code `<s_a_*>` (8,192 classes) | the coarse semantic "
        "proxy, identical in spirit to the category grouping used by CAUSE-Core |")
    add("")
    add("**Similarity.** `cosine(p_u, p_v)` and `1 - JS(p_u, p_v)` (Jensen–Shannon divergence base 2).")
    add("")
    add("**Sampling.** 100,000 pairs are drawn per class: same-token pairs by picking a token with "
        "probability proportional to its number of within-token pairs, different-token pairs by "
        "rejection over two distinct tokens. The first 20,000 pairs of each class are scored; seeds "
        "are fixed (`numpy.default_rng(2026)`), so the numbers are reproducible.")
    add("")
    add("**Token purity (Analysis III, optional panel).** For every token, each member user's "
        "*future category* is the modal level-0 code among its ten targets; purity is the share of "
        "the token's most common category.")
    add("")
    add("**Predictive memory map (Analysis IV).** Each token gets its mean future profile "
        "`p_g = mean_{u in U_g} p_u`; pairwise **JS distance** `sqrt(JS(p_g, p_h))` is computed and "
        "embedded with t-SNE (precomputed metric, perplexity 15, seed 0). Colour = entropy "
        "`H(p_g)`, size = `|U_g|`.")
    add("")
    add("**Reuse (Analysis V).** `n_g = |U_g|`; the right panel relates `n_g` to the token users' "
        "mean `sid/pass@32` from `per_sample.parquet` (Spearman correlation, binned means for the "
        "trend line).")
    add("")
    add("## 3. Analysis III — same-token vs different-token future behaviour")
    add("")
    for space, title in SPACES:
        add("### %s" % title)
        add("")
        add("| K | tokens used | same-token cosine | different-token cosine | gap | ratio | t | "
            "same-token JS | different-token JS |")
        add("|---|---|---|---|---|---|---|---|---|")
        for k in KS:
            z = pairs(k, space)
            a, b = z["same_cosine"], z["diff_cosine"]
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            t = (a.mean() - b.mean()) / se
            add("| %d | %d | %s ± %s | %s ± %s | %s | %.2f× | %.0f | %s | %s |" % (
                k, summary(k)["spaces"][space]["tokens_with_users"], fmt(a.mean()), fmt(a.std(ddof=1)),
                fmt(b.mean()), fmt(b.std(ddof=1)), fmt(a.mean() - b.mean()),
                a.mean() / max(b.mean(), 1e-12), t,
                fmt(z["same_js"].mean()), fmt(z["diff_js"].mean())))
        add("")
    add("Token purity is *not* part of the current three figures (the optional Figure 2 of the "
        "design draft). An earlier revision computed it for K = 1 with the modal level-0 code of "
        "each user's future items as the category: mean purity 0.19 over the 211 tokens carrying "
        "users. Because level-0 codes number 8,192, this proxy is weak; add a real category field "
        "before promoting it to a panel.")
    add("")
    add("## 4. Analysis IV — predictive memory map")
    add("")
    add("Per K: number of tokens carrying users, the entropy spread of the token profiles and the "
        "2-D embedding strength (share of the Gram spectrum captured by the two plotted axes).")
    add("")
    add("| K | tokens used | unused of 256 | mean users/token | singleton tokens | mean entropy | 2-D variance |")
    add("|---|---|---|---|---|---|---|")
    for k in KS:
        tm = token_map(k, "code_level0")
        rep = summary(k)["spaces"]["code_level0"]
        add("| %d | %d | %d | %.1f | %d | %.2f | %.2f |" % (
            k, len(tm), 256 - len(tm), tm["users"].mean(), int((tm["users"] == 1).sum()),
            tm["entropy"].mean(), sum(rep["explained_variance"])))
    add("")
    add("Largest tokens (K = 1):")
    add("")
    tm = token_map(1, "code_level0").sort_values("users", ascending=False).head(10)
    add("| token | users | entropy | mean pass@32 | modal future level-0 code |")
    add("|---|---|---|---|---|")
    for _, row in tm.iterrows():
        add("| %d | %d | %.2f | %s | %s |" % (row["token"], row["users"], row["entropy"],
                                              fmt(row["mean_pass@32"], 4) if pd.notna(row["mean_pass@32"]) else "—",
                                              row["top_key"]))
    add("")
    add("## 5. Analysis V — vocabulary re-use")
    add("")
    add("| K | tokens used | mean users/token | max users/token | singleton tokens | "
        "Spearman(n_g, pass@32) |")
    add("|---|---|---|---|---|---|")
    for k in KS:
        tm = token_map(k, "code_level0").dropna(subset=["mean_pass@32"])
        rho = st.spearmanr(tm["users"], tm["mean_pass@32"]).statistic
        add("| %d | %d | %.1f | %d | %d | %+.3f |" % (
            k, len(tm), tm["users"].mean(), int(tm["users"].max()),
            int((tm["users"] == 1).sum()), rho))
    add("")
    add("## 6. Downstream arms behind the same artifacts (context)")
    add("")
    add("| arm (TIGER + %s, full view) | Pass@5 | Pass@10 | Pass@20 | Pass@32 | Recall@32 | NDCG@32 |"
        % TASK.replace("_", " ").title())
    add("|---|---|---|---|---|---|---|")
    ref = (PROJECT / "output/model/tiger/single_channel/%s/full_r%d/seed2026/eval/metrics.json"
           % (TASK, RECENT))
    rows = [("native full_r%d" % RECENT, ref)]
    for name, arm in ARMS:
        rows.append((name, PROJECT / "output/model/tiger/single_channel/%s/%s/seed2026/eval/metrics.json"
                     % (TASK, arm)))
    for name, path in rows:
        p = Path(path)
        if not p.exists():
            add("| %s | — | — | — | — | — | — |" % name)
            continue
        g = json.loads(p.read_text())["metrics"]
        cells = [fmt(g["sid/pass@%d" % k]) for k in (5, 10, 20, 32)]
        cells += [fmt(g["sid/recall@32"]), fmt(g["sid/ndcg@32"])]
        add("| %s | %s |" % (name, " | ".join(cells)))
    add("")
    add("## 7. Figures and their data")
    add("")
    add("| figure | panels | data files |")
    add("|---|---|---|")
    add("| `fig1_token_consistency.{pdf,png}` | top: mean±SE bars (4 K × 2 classes × 3 spaces); "
        "bottom: complementary ECDFs (log y) | `k<K>/<space>/pairs.npz` |")
    add("| `fig2_predictive_memory_map.{pdf,png}` | 4 t-SNE maps (K = 1/2/4/8) | "
        "`k<K>/code_level0/js_distance.npy` + `token_map.parquet` |")
    add("| `fig3_token_reuse.{pdf,png}` | vocabulary utilisation; reuse vs Pass@32 | "
        "`k<K>/code_level0/token_map.parquet` |")
    add("")
    add("Per K and space the directory holds `summary.json` (aggregates), `pairs.npz` "
        "(`same_cosine/diff_cosine/same_js/diff_js`, the per-pair values plotted in Figure 1), "
        "`vertex`-free `token_map.parquet` (`token, users, entropy, top_key, x, y, mean_pass@32`) and "
        "`js_distance.npy` (211×211 … token-level JS distances).")
    add("")
    add("## 8. Reproducing")
    add("")
    add("```bash")
    add("# statistics (CPU only, ~1 min per K)")
    add("python -m src.model.phi.token_semantics --task product --ks 1 2 4 8")
    add("")
    add("# figures + this document")
    add("MT_ROOT=output/model/analysis/memory_token/single_channel_product \\")
    add("MT_OUT=output/model/analysis/memory_token/figures \\")
    add("python scripts/model/analysis/plot_memory_token.py")
    add("```")
    add("")
    add("## 9. Caveats")
    add("")
    add("1. **The full-catalog item space is degenerate by construction.** With a 1.13M catalog and "
        "ten future items per user the supports essentially never overlap, so the absolute "
        "similarity is ~0.001 even though the same/different *ratio* is ~2–3×. Use the level-0 code "
        "or head-item space for the main claim and keep the full-catalog version as a footnote.")
    add("2. **Both tasks use a ten-item future window.** The design draft said Short Video would use "
        "twenty; the materialised store has `target_width = 10` for Product *and* Short Video.")
    add("3. **Reuse vs accuracy is confounded with item popularity.** A token covering more users "
        "may simply contain popular items; report the correlation with that caveat or control for "
        "the token's prior hit rate.")
    add("4. **The purity proxy is coarse.** Level-0 codes number 8,192, so purity is a weak "
        "category signal; a real category field does not exist in the item table.")
    add("5. **Sampling, not exhaustive pairing.** 100k pairs per class (20k scored) with a fixed "
        "seed; the gaps are 20–25 standard errors, so the sample size is not a limitation.")
    add("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT, "(%d lines)" % len(lines))


if __name__ == "__main__":
    main()
