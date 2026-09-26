"""Build the shared three-level normalized residual Tiger codebook.

Six stages, reported through the shared progress reporter (console.log + progress.json
+ progress.jsonl) exactly like the data pipeline and DeYi fit-PCA:

    plan -> sample -> fit -> assign -> dedup -> publish

Efficiency notes (see docs/new/tiger_codebook.md):
* the fit sample is drawn as **contiguous blocks** of catalog rows (sequential IO instead
  of 600k scattered 16 KB reads; the scattered pattern measures 0.7 MB/s in this project);
* the full-catalog assignment prefetches the next block in a background thread while the
  GPU assigns the current one, hiding ~218 GB of reads behind the matmuls;
* ``codebook.space`` selects the representation (``text`` = 4096-d item text, ``pca`` =
  1024-d DeYi projection) without touching the rest of the chain;
* every level runs the full ``codebook.iterations``: faiss exposes no per-iteration hook, and
  a probe-and-rerun approximation costs extra work on the not-converged path, so the level is
  simply run to completion and the inertia curve is recorded for inspection.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from omegaconf import OmegaConf

from src.common.progress import Progress, fmt_bytes, fmt_seconds, run_directory
from src.data.dataset import open_store
from src.data.runtime import configure_runtime
from . import load_config, parser

STAGES = ("plan", "sample", "fit", "assign", "dedup", "publish")


def assign_tensors(values, centers):
    values = torch.nn.functional.normalize(values.float(), dim=1)
    distances = values.square().sum(1, keepdim=True) + centers.square().sum(1)[None] - 2 * values @ centers.T
    labels = distances.argmin(1)
    return labels, (values - centers[labels]).to(torch.float16)


def assign(values, centers, device):
    labels, residuals = assign_tensors(torch.as_tensor(values, dtype=torch.float32, device=device), centers)
    return labels.cpu().numpy(), residuals.cpu().numpy()


def dedup_digit(codes, width):
    keys = (codes.astype(np.int64) * np.asarray([width ** 2, width, 1])).sum(1)
    order = np.argsort(keys, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    ordinal = np.arange(len(order)) - np.repeat(starts, counts) + 1
    ordinal[np.repeat(counts, counts) == 1] = 0
    result = np.empty(len(order), dtype=np.int32)
    result[order] = ordinal
    return result


def sample_blocks(cfg, catalog):
    """Contiguous catalog blocks covering ``codebook.sample_items`` rows."""
    rng = np.random.default_rng(cfg.seed)
    total = min(int(cfg.codebook.sample_items), len(catalog))
    block = max(1, min(int(cfg.codebook.get("block_rows", 1024)), total))
    count = max(1, min(total // block, len(catalog) // block))
    starts = np.sort(rng.choice(max(1, len(catalog) - block), size=count, replace=False))
    return [catalog[int(start):int(start) + block] for start in starts]


def vectors(source, rows):
    """Sequential read of one contiguous row block (near-sequential file access)."""
    span = source[int(rows[0]):int(rows[-1]) + 1]
    values = np.asarray(span, dtype=np.float32)
    offsets = np.asarray(rows, dtype=np.int64) - int(rows[0])
    return values[offsets]


def item_source(cfg, dataset):
    """Return ``(source, dim, label)`` for the representation being quantized."""
    space = str(cfg.codebook.get("space", "text"))
    if space == "text":
        return dataset.items.text, int(dataset.items.text.shape[1]), "text(4096d)"
    if space == "pca":
        path = Path(str(cfg.codebook.get("pca_path", "output/model/deyi/asset/item_space/embeddings.npy")))
        if not path.is_file():
            raise FileNotFoundError(
                "codebook.space=pca requires the DeYi item space at %s; "
                "run scripts/model/deyi/prepare.sbatch first" % path
            )
        pca = np.load(path, mmap_mode="r")
        if pca.shape[0] != len(dataset.items.pids):
            raise ValueError("DeYi item space rows %d do not match the item table rows %d"
                             % (pca.shape[0], len(dataset.items.pids)))
        return pca, int(pca.shape[1]), "pca(%dd)" % pca.shape[1]
    raise ValueError("codebook.space must be 'text' or 'pca'")


def log_directory(cfg, stage="tiger/prepare"):
    """Run directory: $MODEL_LOG_DIR (set by scripts/model/common_logging.sh) or a fresh one."""
    configured = os.environ.get("MODEL_LOG_DIR")
    if configured:
        path = Path(configured)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return run_directory(Path("log/model") / stage, os.environ.get("SLURM_JOB_ID"))


def _fit_level(cfg, faiss, device, sample, dim, level):
    """One k-means level (L2 by default, spherical when configured), always running the full ``iterations``."""
    fit_values = np.ascontiguousarray(sample, dtype=np.float32)
    faiss.normalize_L2(fit_values)
    options = dict(niter=int(cfg.codebook.iterations), nredo=int(cfg.codebook.get("nredo", 1)),
                   seed=int(cfg.seed) + level, gpu=device.type == "cuda", verbose=True,
                   spherical=bool(cfg.codebook.get("spherical", False)))
    if bool(cfg.codebook.get("use_float16", False)):
        # faiss 1.15 exposes fp16 only on gpu indexes (GpuIndexFlatConfig.useFloat16), not on
        # ClusteringParameters, which rejects unknown fields at construction time.
        if not hasattr(faiss.ClusteringParameters(), "use_float16"):
            raise ValueError("codebook.use_float16=true needs a faiss build with an fp16 k-means "
                             "path; installed faiss %s is fp32-only" % faiss.__version__)
        options["use_float16"] = True
    kmeans = faiss.Kmeans(dim, int(cfg.codebook.width), **options)
    kmeans.train(fit_values)
    del fit_values
    return kmeans, [float(value) for value in getattr(kmeans, "obj", [])]


def prepare(cfg):
    import faiss

    configure_runtime(cfg.runtime.cpu_threads)
    torch.set_num_threads(cfg.runtime.cpu_threads)
    faiss.omp_set_num_threads(cfg.runtime.cpu_threads)
    # Match the original assignment arithmetic (float32 matmuls with TF32 allowed).
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(cfg.runtime.device)
    root, destination = Path(cfg.data_dir), Path(cfg.prepared_dir)
    destination.mkdir(parents=True, exist_ok=True)
    progress = Progress(log_directory(cfg), len(STAGES),
                        interval=float(cfg.codebook.get("progress_interval", 5)))
    started = monotonic()
    if (destination / "codebook.json").exists():
        progress.line("plan", "completed codebook exists: %s" % destination, force=True)
        progress.finish("codebook reused (%s)" % destination)
        return

    progress.stage_begin(1, "plan", "validate item table + read inputs")
    dataset = open_store(str(root)).item_space()
    source, dim, space_label = item_source(cfg, dataset)
    catalog = np.asarray(dataset.catalog, dtype=np.int64)
    threads = int(cfg.runtime.cpu_threads)
    prefetch_workers = max(1, int(cfg.codebook.get("prefetch_workers", 2)))
    residual_dtype = np.dtype(str(cfg.codebook.get("residual_dtype", "float16")))
    progress.banner([
        "=" * 96,
        "Tiger codebook  |  job=%s  device=%s" % (os.environ.get("SLURM_JOB_ID"), device),
        "  space       : %s   catalog=%d   text_rows=%d" % (space_label, len(catalog), len(source)),
        "  codebook    : width=%d layers=%d sample_items=%d niter=%d nredo=%d spherical=%s" % (
            int(cfg.codebook.width), int(cfg.codebook.layers), int(cfg.codebook.sample_items),
            int(cfg.codebook.iterations), int(cfg.codebook.get("nredo", 1)),
            bool(cfg.codebook.get("spherical", False))),
        "  residuals   : %s   use_float16=%s" % (residual_dtype.name,
                                                   bool(cfg.codebook.get("use_float16", False))),
        "  output      : %s" % destination,
        "  logs        : %s" % progress.log_dir,
        "  plan        : " + " -> ".join("%d.%s" % (i + 1, name) for i, name in enumerate(STAGES)),
        "=" * 96,
    ])
    progress.stage_end("space=%s catalog=%d" % (space_label, len(catalog)))

    progress.stage_begin(2, "sample", "sample_items=%d block_rows=%d" % (
        int(cfg.codebook.sample_items), int(cfg.codebook.get("block_rows", 1024))))
    blocks = sample_blocks(cfg, catalog)
    sampled = np.concatenate(blocks)
    parts = []
    for index, block in enumerate(blocks, start=1):
        parts.append(vectors(source, block))
        progress.tick(index, len(blocks), "blocks=%d/%d rows=%d/%d %s" % (
            index, len(blocks), sum(len(part) for part in parts), len(sampled),
            fmt_bytes(sum(part.nbytes for part in parts))),
            counters={"rows_sampled": int(sum(len(part) for part in parts))})
    sample = np.concatenate(parts)
    del parts
    progress.stage_end("rows=%d shape=%s" % (len(sample), sample.shape))

    progress.stage_begin(3, "fit", "layers=%d nredo=%d" % (int(cfg.codebook.layers),
                                                           int(cfg.codebook.get("nredo", 1))))
    centers, inertia = [], []
    for level in range(int(cfg.codebook.layers)):
        level_started = monotonic()
        progress.line("fit", "level=%d/%d START niter=%d nredo=%d dim=%d rows=%d" % (
            level + 1, int(cfg.codebook.layers), int(cfg.codebook.iterations),
            int(cfg.codebook.get("nredo", 1)), dim, len(sample)), force=True)
        kmeans, objective = _fit_level(cfg, faiss, device, sample, dim, level)
        current = np.asarray(kmeans.centroids, dtype=np.float32)
        centers.append(current.copy())
        inertia.append(objective[-1] if objective else None)
        del kmeans
        progress.line("fit", "level=%d/%d iterations=%d seconds=%.1f inertia=%s" % (
            level + 1, int(cfg.codebook.layers), int(cfg.codebook.iterations),
            monotonic() - level_started, inertia[-1]), force=True)
        if level < int(cfg.codebook.layers) - 1:
            residuals = np.empty(sample.shape, dtype=residual_dtype)
            gpu_centers = torch.as_tensor(current, device=device)
            for start in range(0, len(sample), int(cfg.codebook.assignment_batch_size)):
                end = min(start + int(cfg.codebook.assignment_batch_size), len(sample))
                _, residuals[start:end] = assign(sample[start:end], gpu_centers, device)
            sample = residuals
    del sample
    progress.stage_end("inertia=%s" % inertia)

    progress.stage_begin(4, "assign", "catalog=%d batch=%d prefetch=%d" % (
        len(catalog), int(cfg.codebook.assignment_batch_size), prefetch_workers))
    gpu_centers = [torch.as_tensor(c, device=device) for c in centers]
    triples = np.empty((len(catalog), 3), dtype=np.int32)
    batch = int(cfg.codebook.assignment_batch_size)
    assign_started = monotonic()
    with ThreadPoolExecutor(max_workers=prefetch_workers) as pool:
        pending = None
        for start in range(0, len(catalog), batch):
            end = min(start + batch, len(catalog))
            values_np = pending.result() if pending is not None else vectors(source, catalog[start:end])
            if end < len(catalog):
                pending = pool.submit(vectors, source, catalog[end:min(end + batch, len(catalog))])
            else:
                pending = None
            values = torch.as_tensor(values_np, device=device)
            for level, current in enumerate(gpu_centers):
                labels, values = assign_tensors(values, current)
                triples[start:end, level] = labels.cpu().numpy()
            progress.tick(end, len(catalog), "items=%d/%d %.0f/s" % (
                end, len(catalog), end / max(monotonic() - assign_started, 1e-6)),
                counters={"items_assigned": int(end)})
    progress.stage_end("items=%d seconds=%.1f" % (len(catalog), monotonic() - assign_started))

    progress.stage_begin(5, "dedup", "collision suffix")
    suffix = dedup_digit(triples, int(cfg.codebook.width))
    if int(suffix.max()) >= int(cfg.codebook.width):
        raise ValueError("Collision suffix exceeds the configured Tiger width; the codebook is not publishable")
    progress.stage_end("max_dedup=%d" % int(suffix.max()))

    progress.stage_begin(6, "publish", "codes/centers/codebook.json")
    codes = np.lib.format.open_memmap(destination / "codes.npy", mode="w+", dtype=np.int16,
                                      shape=(len(source), 4))
    codes[:] = -1
    codes[catalog, :3], codes[catalog, 3] = triples, suffix
    codes.flush()
    np.save(destination / "centers.npy", np.stack(centers))
    OmegaConf.save(cfg, destination / "config.yaml")
    summary = {"format": "tiger-normalized-rq3-dedup-v1", "width": int(cfg.codebook.width),
               "layers": 3, "seed": int(cfg.seed), "catalog_items": len(catalog),
               "text_rows": len(source), "sample_items": len(sampled), "max_dedup": int(suffix.max()),
               "space": space_label, "residual_dtype": residual_dtype.name,
               "kmeans": {"niter": int(cfg.codebook.iterations),
                          "nredo": int(cfg.codebook.get("nredo", 1)),
                          "use_float16": bool(cfg.codebook.get("use_float16", False)),
                          "spherical": bool(cfg.codebook.get("spherical", False)),
                          "inertia": inertia},
               "seconds": monotonic() - started, "row_index": "canonical physical item row",
               "fit": "contiguous catalog blocks; faiss kmeans; seed+level",
               "residual": "L2 normalize each level, subtract centroid, store as %s" % residual_dtype.name}
    (destination / "codebook.json").write_text(json.dumps(summary, indent=2) + "\n")
    progress.stage_end("codes/centers/codebook.json")
    progress.finish("codebook ready (%d items, %s)" % (len(catalog), fmt_seconds(monotonic() - started)))


def main():
    args = parser().parse_args()
    prepare(load_config(args.config, args.task, args.mode, args.run_dir))


if __name__ == "__main__":
    main()
