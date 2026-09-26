"""Stage 3: project every item into the 1024-d space, in parallel.

The stage is I/O bound (231 GB read, 28.8 GiB written), so the workers run the matmul on
CPU: the debug node offers 10 cores, while sharing one 40 GiB MIG between six CUDA
contexts would only add contention. Each worker writes its own contiguous row range of the
pre-allocated ``embeddings.npy`` — no shuffle, no merge, no temporary copies.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch


def _worker(job):
    text_path, out_path, mean, components, start, stop, batch, threads = job
    torch.set_num_threads(int(threads))
    text = np.load(text_path, mmap_mode="r")
    out = np.load(out_path, mmap_mode="r+")
    mean_t = torch.from_numpy(mean).float()
    components_t = torch.from_numpy(components).float()
    written = 0
    with torch.inference_mode():
        for begin in range(start, stop, batch):
            end = min(begin + batch, stop)
            values = torch.from_numpy(np.asarray(text[begin:end])).float()
            projected = torch.nn.functional.linear(values - mean_t, components_t)
            out[begin:end] = projected.numpy().astype(np.float16)
            written += end - begin
    out.flush()
    return written


def project_all(cfg, space, mean, components, out_path, workers, progress):
    rows = len(space.items)
    state_dim = int(cfg.deyi.state_dim)
    batch = int(cfg.pca.project_batch_size)
    text_path = str(space.items.text_path)
    matrix = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                       shape=(rows, state_dim))
    matrix.flush()
    del matrix

    threads = max(1, (10 // max(1, workers)) or 1)
    bounds = np.linspace(0, rows, int(workers) + 1, dtype=np.int64)
    jobs = [(text_path, str(out_path), np.asarray(mean, dtype=np.float32),
             np.asarray(components, dtype=np.float32), int(bounds[i]), int(bounds[i + 1]),
             batch, threads) for i in range(int(workers))]
    started = time.time()
    written = 0
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            written += int(future.result())
            elapsed = max(time.time() - started, 1e-6)
            progress.tick(done, len(jobs),
                          "workers=%d/%d rows=%d/%d %.0fMiB/s" % (
                              done, len(jobs), written, rows,
                              written * 1024 * 2 / 2**20 / elapsed),
                          counters={"rows_written": written})
    return {"rows": written, "seconds": time.time() - started, "workers": int(workers)}
