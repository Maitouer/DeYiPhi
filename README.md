# A Generative Recommender Also Needs a Discrete Language for Memory

Code for the paper's inference-time memory study on **TIGER**. A generative recommender already
has a discrete language for *items* (Semantic IDs). This repository adds a second, smaller
language for *memory*: recurrent predictive states of the interaction history get their own
symbols, and one symbol is enough to improve generation from the full history.

Two components are released here.

* **DeYi** distils the interaction history into a small number of user-specific continuous
  predictive states, trained to preserve what the history implies about the future rather than
  what it literally contains.
* **Φ** maps those private continuous states into a *shared* discrete vocabulary. Its codebook is
  fitted on the item predictions the states induce, not on latent-space proximity, so the same
  symbol means the same thing for different users.

The downstream arm consumes the memory token together with the unchanged TIGER input: the memory
token is prepended to the full interaction sequence and also defines a context-dependent
predictive route ahead of the target item's native SID. All numbers in the paper's main tables
come from this repository; both history views (`full` and `recent`) and both memory types
(continuous `DeYi` and discrete `Φ`) are produced by the same pipeline.

---

## 1. Repository layout

```
config/
  data.yaml                 # data pipeline: sources, windows, datasets, task ownership
  model/deyi.yaml           # DeYi encoder, objective, item semantic space
  model/phi.yaml            # Φ vocabulary, router, routes, audit
  model/tiger.yaml          # TIGER backbone, SID codebook, training and evaluation
scripts/
  data/prepare.sbatch       # build output/data from the released assets
  model/common_logging.sh   # per-stage log directory, scratch/TMPDIR policy, disk preflight
  model/deyi/               # item semantic space, teacher training, state export, K sweep
  model/phi/                # the Φ chain and its single stages
  model/tiger/              # TIGER asset build, arm drivers, sft/eval stages
  model/baseline/           # controlled continuous-memory baselines (pooling, CAUSE, …)
  model/analysis/           # memory-token analysis (Analysis III/IV/V) and held-out equivalence
src/
  data/                     # the data pipeline (sources, item table, splits, self-check)
  model/deyi/               # DeYi: model, objective, encoder, artifacts
  model/phi/                # Φ: reservoir, estimator, codebook, router, tokenizer, routes, audit
  model/tiger/              # TIGER: prepare, materialise, train, evaluate, marginal beam
  model/baseline/           # alternative state organisations feeding the same TIGER head
  model/tasks.py            # task-neutral row reader shared by every consumer
tests/                      # unit tests for the data layout, DeYi/Φ configs and TIGER components
```

Only the TIGER line is included: the item catalog, DeYi, Φ, the TIGER arms, and the analyses that
ablate memory. Artefacts that need the released corpus (the item table, the codebooks, checkpoints,
predictions) are **not** in the repository; the pipeline below regenerates them.

---

## 2. Environment

Everything runs through Slurm. The stages only assume:

* a Python environment on `PATH` (or `DEYIPHI_PYTHON` pointing at its interpreter) with the
  packages in `requirements.txt`, **plus a CUDA build of FAISS installed separately** — FAISS is
  distributed outside PyPI here, so `pip install -r requirements.txt` does not cover it;
* `DEYIPHI_ROOT` set to the repository root (every script defaults to the working directory, so
  running from the root works as well);
* a writable scratch directory. `scripts/model/common_logging.sh` resolves one in this order:
  `DEYIPHI_TMPDIR`, `SLURM_TMPDIR`, `<DEYIPHI_SCRATCH_ROOT>/<user>/slurm-scratch/<jobid>` (only
  when that root is exported), `/dev/shm`, `/tmp`, `/var/tmp`. Set `DEYIPHI_TMPDIR` when your site
  forbids the rest.

The partition and QOS names in the `#SBATCH` headers (`debug`/`debug_normal`,
`gpu`/`gpu_normal`, `gpu_heavy`/`gpu_priv`) describe the shape each stage needs — a 1-GPU
preparation job, a 4-GPU training job, or a whole 8-GPU node. Adapt them to your own cluster;
the drivers also accept `KSWEEP_PARTITION` / `KSWEEP_QOS` and `--limit` for the same purpose.

---

## 3. Data

### 3.1 Source

All recommendation data and the item catalogue come from the released **OpenOneRec RecIF** assets:

<https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF>

Mirror the following files under `asset/data/` (the paths below are what `config/data.yaml`
expects):

| File | Size | Role |
|---|---|---|
| `OpenOneRec-RecIF/onerec_bench_release.parquet` | 162,074 rows × 25 cols | training pool (`split == 0`): multi-channel histories, targets, behaviour flags, `uid` |
| `OpenOneRec-RecIF/benchmark_data/video/video_test.parquet` | 38,781 rows | short-video test split (`metadata` carries `uid`, `hist_pid`, `answer_pid`) |
| `OpenOneRec-RecIF/benchmark_data/product/product_test.parquet` | 27,910 rows | product test split (`hist_goods`, `answer_iid`) |
| `OpenOneRec-RecIF/video_ad_pid2sid.parquet` | 15,885,203 | official SID mapping for video/ad pids |
| `OpenOneRec-RecIF/product_pid2sid.parquet` | 2,066,115 | official SID mapping for product pids |
| `OpenOneRec-RecIF/benchmark_data/sid2pid.json`, `sid2iid.json` | — | official SID → item maps |
| `openonerec_multimodal_embedding/` | 1,500 shards, 17,433,569 pids | item content: `text_emb` (4,096-d) and `vision_emb` |

The text embedding behind the item table is Qwen3-Embedding-8B (4,096-d), as required by the
released tokenizer card. The two tasks used in the paper are the **single-channel** views of
short video and product; the released assets also carry an ad task, which the pipeline supports
but the paper does not report.

### 3.2 Processing

One builder produces everything (`config/data.yaml` + `src/data/`, run as
`sbatch scripts/data/prepare.sbatch`). Every artefact is written with `.tmp` + atomic rename, and
the five stages are resumable:

1. **plan** — read the config, the released tables and the embedding shard footers; print the plan.
2. **item-table** — two passes over the 1,500 embedding shards. Pass A reads only `pid` and
   whether `text_emb` is valid, so final row numbers are known before any vector is decoded and
   the output matrix can be pre-allocated; pass B decodes contiguous shard ranges in parallel and
   writes straight into the matrix. Row 0 is a zero sentinel: a PID owns the row of its first
   occurrence that actually carries a text vector, and a PID with no text vector keeps row 0.
   Output: `pids.npy` (int64) and `text.npy` (float32, 4,096-d).
3. **splits** — per dataset × task × split:
   * rows: train = `split == 0` of the wide table, test = the released benchmark file;
   * keep rows with at least one target;
   * right-truncate each history channel to its declared window (`tail`, most recent items kept):
     video 512, product 100, ad 200;
   * drop a row whose *primary* channel is shorter than `min_primary_history = 32`;
   * map PIDs to physical item rows and drop a row if any history or target entry has no vector;
   * write the channel order inside the file header, so a reader can never recover it by scanning
     tensor keys.
4. **manifest** — `dataset.json` per dataset (rules, tasks, per-split row counts) and
   `catalog_rows.npy` per task and dataset; the task → dataset ownership table is `tasks.json`.
5. **selfcheck** — invariants (padding, monotone catalogues, row ranges) plus a sampled parity
   check against the released tables.

Two datasets come out of the same tables:

* `single_channel` — only the task's own channel; **used in the paper**;
* `multi_channel` — the cross-domain form (video + ad, video + product), video first.

### 3.3 What the two paper datasets look like afterwards

| | Product | Short Video |
|---|---|---|
| Dataset variant | `single_channel` | `single_channel` |
| Train rows | 75,987 | 151,510 |
| Test rows | 18,744 | 37,464 |
| History window used | 100 | 512 |
| Future window (targets per row) | 10 (8.1 avg) | 10 (8.9 avg) |
| Task catalogue | 1,132,066 | 12,881,955 |

Shared item table (`output/data/items/item_table.json`): 17,433,569 scanned embedding rows,
13,839,796 requested PIDs, **13,819,843 stored rows** including the sentinel, 19,954 PIDs with no
text vector.

---

## 4. Method pipeline and its artefacts

All paths below are relative to the repository root and are deterministic given the seed
(`seed: 2026` everywhere).

| Stage | Produces | Built by |
|---|---|---|
| Data | `output/data/{single_channel,multi_channel}/<task>/*.bin` | `scripts/data/prepare.sbatch` |
| Item semantic space | `output/model/item/semantic_space/` (PCA of the 4,096-d text vectors → 1,024-d) | `scripts/model/deyi/prepare.sbatch` |
| TIGER SID codebook | `output/model/item/tiger_codebook/` | `scripts/model/tiger/prepare.sbatch` |
| DeYi teacher + states | `output/model/deyi/<ds>/<task>/k<K>_r<R>/{checkpoint.pt,states/{train,test}.pt}` | `scripts/model/deyi/{train,encode}.sbatch` |
| Φ vocabulary + routes | `output/model/phi/<ds>/<task>/k<K>_r<R>/{codebook.pt,router.pt,tokens/,routes/,audit.json}` | `scripts/model/phi/chain.sbatch` |
| TIGER arms | `output/model/tiger/<ds>/<task>/<arm>/seed2026/{train_summary.json,model.pt,eval/metrics.json}` | `scripts/model/tiger/*.sbatch` |

`<ds>` is `single_channel`. `<arm>` is `full_r<R>`, `recent_r<R>`, `deyi_full_r<R>_k<K>` or
`phi_full_r<R>_k<K>`, where `R` is the recent window (`32` for short video, `16` for product) and
`K` is the number of memory states or memory tokens. The DeYi and Φ arms always read a `full`
history; the subscript only names the teacher's recent cut.

Evaluation writes `sid/pass@{1,5,10,20,32}`, `sid/recall@{…}`, `sid/ndcg@{…}` with beam 32 over
the whole test split, plus per-sample scores for the analyses.

---

## 5. Running the main experiments

Run every command from the repository root (or export `DEYIPHI_ROOT`). `sbatch` echoes a job id;
the drivers are idempotent — they skip a stage whose artefact exists, submit the next missing
stages while submission slots are free, and stop. Call them again later to advance.

### 5.1 Data

```bash
sbatch scripts/data/prepare.sbatch                    # add --resume or --max-rows N while testing
```

### 5.2 Shared assets (once)

```bash
sbatch scripts/model/deyi/prepare.sbatch              # item semantic space (1 GPU)
sbatch scripts/model/tiger/prepare.sbatch             # SID codebook (1 GPU, several hours)
```

### 5.3 DeYi teacher and Φ vocabulary

The upstream sweep trains the DeYi teacher, exports its states and fits the Φ vocabulary for
every K of one task. It uses only 1-GPU jobs, so it does not compete with the 4-GPU arms.

```bash
bash scripts/model/deyi/chain_ksweep.sh short_video --ks "1 2 4 8" --recent 32
bash scripts/model/deyi/chain_ksweep.sh product     --ks "1 2 4 8" --recent 16
```

Both accept `--dry-run` and `--retry`. Single stages are available too:

```bash
sbatch scripts/model/deyi/encode.sbatch short_video 4 32
sbatch scripts/model/phi/chain.sbatch short_video 4 32
sbatch scripts/model/phi/stage.sbatch audit short_video 4 32    # one stage only
```

### 5.4 TIGER arms

`chain_arms.sh` runs one task's four arm families in order: `full` and `recent` first (each
materialise → sft → eval), then the DeYi arm and the Φ arm for one `K`.

```bash
bash scripts/model/tiger/chain_arms.sh short_video --k 1     # full, recent, DeYi K=1, Φ K=1
bash scripts/model/tiger/chain_arms.sh product     --k 1
bash scripts/model/tiger/chain_arms.sh short_video --k 4     # then K=4, and so on
```

Add `--dry-run` to print the plan, `--retry` after fixing a failed stage, `--limit N` to cap how
many jobs the driver keeps queued. Individual stages:

```bash
sbatch scripts/model/tiger/materialize.sbatch short_video full   # once per task and mode
sbatch scripts/model/tiger/sft.sbatch        short_video full    # native, full history
sbatch scripts/model/tiger/deyi_sft.sbatch   short_video recent 4
sbatch scripts/model/tiger/phi_sft.sbatch    short_video recent 4
sbatch scripts/model/tiger/eval.sbatch       short_video full
```

### 5.5 Controlled memory baselines

The analysis table keeps the TIGER interface fixed and varies only how the auxiliary memory is
built. `chain_sv_analysis.sh` / `chain_analysis_arms.sh` drive the short-video and product
versions; the preparation stages run on one GPU each and the arms on four.

```bash
bash scripts/model/tiger/chain_sv_analysis.sh            # short video, K=4
bash scripts/model/tiger/chain_analysis_arms.sh          # product, K=4
sbatch scripts/model/baseline/pooling.sbatch short_video 4 32
sbatch scripts/model/baseline/cause.sbatch   short_video 32 4
sbatch scripts/model/phi/variant.sbatch euclidean short_video 4 32
```

### 5.6 Memory-token analyses

Analysis III–V read the frozen Φ tokens, the materialised future windows and the per-sample
evaluation:

```bash
python -m src.model.phi.token_semantics --help          # per K and per comparison space
python scripts/model/analysis/export_analysis3_data.py  # tables behind the figures
python scripts/model/analysis/plot_memory_token.py      # figures
python scripts/model/analysis/export_token_level_similarity.py
```

Set `DEYIPHI_EXTRA_ROOT` if a copy of `output/` lives outside the repository.
Held-out predictive equivalence is a separate 1-GPU job:

```bash
sbatch scripts/model/analysis/equivalence.sbatch product 4 16
```

---

## 6. Which stage produces which paper result

| Paper result | Artefact |
|---|---|
| Main table, `FULL` / `RECENT` rows | `output/model/tiger/<ds>/<task>/{full,recent}_r<R>/seed2026/eval/metrics.json` |
| Main table, `DeYi` and `Φ` rows | `…/deyi_full_r<R>_k<K>/…` and `…/phi_full_r<R>_k<K>/…` |
| Memory-capacity study (K = 1…8) | the same two families, one directory per K |
| Controlled analysis (`+Dummy`, `+Pooling`, CAUSE/HiCoGen/Chronicle/… ) | `…/full_r<R>_dummy<K>/…`, `…/deyi_full_r<R>_k<K>/{pooling,cause,hicogen,chronicle}/…`, `…/phi_full_r<R>_k<K>_{euclidean,cosine}/…` |
| Token-level and pair-level memory analyses | `scripts/model/analysis/` + `src/model/phi/token_semantics.py` |
| Vocabulary quality | `output/model/phi/<ds>/<task>/k<K>_r<R>/audit.json` (`dead codes`, predictive/validation KL, route confidence, router top-1, regret) |

Each arm directory carries `train_summary.json` (`training_complete`, steps, rows, wall-clock) and
`eval/metrics.json`; an evaluation refuses to run unless the training summary says the full
budget was consumed, so a preempted run can never be scored by accident.

---

## 7. Notes

* **Seed and splits.** Everything uses seed `2026`. The test splits are the released held-out
  benchmark files; the training rows are the `split == 0` part of the wide table, filtered by the
  rules in §3.2.
* **Reproducing a single row.** `train_summary.json` records the step budget
  (`151510 / 512 → 296 steps per epoch`, 10 epochs) so a resumed run can be checked against the
  original.
* **Memory footprint.** The item table is the large artefact (≈ 200 GiB at 4,096-d float32);
  `output/` is git-ignored and expected to live on a scratch volume.
* **Licence of the data.** The corpus and the item embeddings are redistributed upstream; follow
  the terms of the OpenOneRec release rather than this repository when reusing them.
