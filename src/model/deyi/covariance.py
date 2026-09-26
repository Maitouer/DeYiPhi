"""Stages 1-2 of the item semantic space: normalize, accumulate, eigendecompose.

Per ``00-Reconstruct/00_item_semantic_space.md`` the Qwen3 text vector is L2-normalized
**before** any centering, so PCA describes semantic direction change rather than the raw
embedding-norm variation. Everything after that lives in one Euclidean PCA space.

By default the mean and the Gram are accumulated over the **whole catalog** (``pca.samples:
0``): both come out of the same sequential pass, and an exact mean matters more than an
exact basis because a mean error shifts every coordinate while a basis error only moves
mass inside the discarded tail. Setting ``pca.samples`` to a positive number falls back to
contiguous sampled blocks (item rows follow the shard-scan order and carry no semantics, so
blocks are still a valid random sample) at 25x the read throughput of scattered rows.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from src.model.deyi.runtime import device


def row_blocks(cfg, catalog):
    """Return the list of contiguous catalog slices to accumulate over."""
    block_rows = max(1, int(cfg.pca.get("block_rows", 16384)))
    samples = int(cfg.pca.get("samples", 0))
    if samples <= 0:
        return [catalog[start:start + block_rows] for start in range(0, len(catalog), block_rows)]
    total = min(samples, len(catalog))
    block = max(1, min(block_rows, total))
    count = max(1, total // block)
    generator = torch.Generator().manual_seed(int(cfg.seed))
    starts = torch.randperm(max(1, len(catalog) - block), generator=generator)[:count].numpy()
    return [catalog[start:start + block] for start in sorted(starts.tolist())]


def accumulate(cfg, space, blocks, progress):
    """Stream the blocks through the device; return ``(mean, covariance, stats)``."""
    target = device()
    text = space.items.text
    dim = int(text.shape[1])
    total = torch.zeros(dim, device=target)
    gram = torch.zeros((dim, dim), device=target)
    processed = zero_norm = 0
    rows_total = sum(len(block) for block in blocks)

    def take(block):
        # ``np.asarray`` on a read-only mmap slice returns a view, and the normalization below
        # writes in place; copy once so the block owns a writable buffer.
        values = np.array(text[int(block[0]):int(block[-1]) + 1], dtype=np.float32)
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        # A zero vector cannot be direction-normalized; count it and leave it as zero so the
        # accumulation stays finite. build() aborts if any is inside the catalog.
        zeros = int((norm <= 0).sum())
        np.divide(values, np.where(norm > 0, norm, 1.0), out=values)
        return values, zeros

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(take, blocks[0])
        for index, block in enumerate(blocks):
            values_np, zeros = pending.result()
            if index + 1 < len(blocks):
                pending = pool.submit(take, blocks[index + 1])
            offsets = np.asarray(block, dtype=np.int64) - int(block[0])
            values = torch.from_numpy(values_np[offsets]).to(target).float()
            total += values.sum(0)
            gram.addmm_(values.T, values)
            processed += len(block)
            zero_norm += zeros
            progress.tick(processed, rows_total,
                          "blocks=%d/%d normalized=%d/%d" % (index + 1, len(blocks), processed,
                                                             rows_total))
    mean = total / processed
    covariance = gram / processed - mean.outer(mean)
    stats = {"rows": int(processed), "zero_norm_rows": int(zero_norm),
             "exact": int(cfg.pca.get("samples", 0)) <= 0,
             "catalog_rows": int(len(space.catalog))}
    return mean, covariance, stats


def fit(covariance, dim):
    """Top ``dim`` principal components of the centered covariance, descending."""
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[:, -dim:].T.flip(0).contiguous()
    eigenvalues = eigenvalues[-dim:].flip(0).contiguous()
    return eigenvalues, components
