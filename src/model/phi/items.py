"""The frozen item side: the unit semantic vectors e(i) and the task catalog phi samples from.

``docs/v1_algorithm/00_item_semantic_space.md`` fixes the two views of the item space:
``embeddings.npy`` stores the Euclidean PCA coordinate ``p(i)`` (used by RQ-Kmeans) and every
consumer that needs cosine-style matching applies ``e(i) = normalize(p(i))``. Phi is such a
consumer, both in the DeYi predictive distribution ``p_s(i) ∝ exp(z_s^T e(i) / tau_p)`` and in
the tail proposal, so nothing here ever reads a second item representation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

BLOCK = 1 << 20


class FrozenItems:
    """Physical item row -> unit semantic vector, read through the frozen DeYi item space."""

    def __init__(self, cfg):
        from src.model.deyi import artifacts as deyi_artifacts
        from src.model.deyi import item_space as deyi_item_space

        # ``require_item_space`` refuses a space built from another item table or another PCA
        # setting, so phi cannot silently consume a stale projection.
        self.manifest = deyi_item_space.require_item_space(cfg)
        self.path = str(deyi_artifacts.embeddings(cfg))
        self.embeddings = np.load(self.path, mmap_mode="r")
        self.rows = int(self.embeddings.shape[0])
        self.dim = int(self.embeddings.shape[1])
        if self.dim != int(self.manifest["dim"]):
            raise ValueError("item space dimension disagrees with its manifest")

    def unit(self, rows, *, device=None, dtype=torch.float32) -> torch.Tensor:
        """``e(i)`` for physical item rows, with the reads sorted into ascending order.

        Random rows of a 28 GB mmap are the dominant cost of every phi stage; sorting each block
        turns them into near-sequential IO and the inverse permutation restores the caller's
        order, so the returned vectors are exactly the ones asked for.
        """
        values = np.asarray(rows, dtype=np.int64).reshape(-1)
        output = torch.empty((values.size, self.dim), dtype=torch.float32)
        for start in range(0, values.size, BLOCK):
            block = values[start:start + BLOCK]
            order = np.argsort(block, kind="stable")
            gathered = torch.from_numpy(np.asarray(self.embeddings[block[order]], dtype=np.float32))
            gathered = torch.nn.functional.normalize(gathered, dim=-1)
            inverse = np.empty_like(order)
            inverse[order] = np.arange(order.size)
            output[start:start + block.size] = gathered[torch.from_numpy(inverse)]
        return output if device is None else output.to(device=device, dtype=dtype)


class TaskCatalog:
    """The task's frozen candidate universe plus the predictive proposal ``q(i)``.

    ``q(i) = lambda * f_fit(i) + (1 - lambda) / |C|`` (``03_phi.md`` §6.3): ``f_fit`` is the fit
    interaction frequency -- the same quantity DeYi's own negative-sampling proposal uses -- and
    the uniform term keeps every catalog item reachable, which is what makes the tail estimator
    unbiased rather than merely plausible.

    The catalog is the *task's* candidate universe, not the dataset's: the predictive
    denominator has to be taken over exactly the items the model may recommend.
    """

    def __init__(self, cfg, items: FrozenItems, *, lam: float = 0.5, split: str = "train",
                 device=None):
        from src.model.deyi.data import projected_catalog

        catalog = projected_catalog(cfg)
        rows = np.asarray(catalog.catalog_rows.numpy(), dtype=np.int64)
        # ``_frequency`` counts targets into this catalog, so the rows have to be in place first.
        self._setup(items, rows, None, lam, device)
        self._proposal_from(self._frequency(cfg, split), lam, device)

    @classmethod
    def from_rows(cls, items, rows, frequency=None, *, lam: float = 0.5, device=None):
        """A catalog over explicit rows: the same object the pipeline builds, without a dataset."""
        instance = cls.__new__(cls)
        instance._setup(items, np.asarray(rows, dtype=np.int64), frequency, lam, device)
        return instance

    def _setup(self, items, rows, frequency, lam: float, device) -> None:
        self.rows = np.asarray(rows, dtype=np.int64)
        self.size = int(self.rows.size)
        if self.size < 2:
            raise ValueError("the task catalog is too small for a predictive vocabulary")
        self.items = items
        self._proposal_from(frequency, lam, device)

    def _proposal_from(self, frequency, lam: float, device) -> None:
        """Build the tail proposal ``q(i) = lam * f_fit(i) + (1 - lam) / |C|``."""
        if frequency is None:
            frequency = torch.ones(self.size, dtype=torch.float64)
        frequency = torch.as_tensor(frequency, dtype=torch.float64).reshape(-1)
        if int(frequency.numel()) != self.size:
            raise ValueError("the frequency table does not span the catalog")
        frequency = frequency.clamp_min(0.0)
        frequency = frequency / max(float(frequency.sum()), 1e-30)
        proposal = float(lam) * frequency + (1.0 - float(lam)) / self.size
        self.proposal = torch.as_tensor(proposal, dtype=torch.float64, device=device)
        self.log_proposal = self.proposal.clamp_min(1e-300).log()
        self.cumulative = torch.cumsum(self.proposal, dim=0)
        self.cumulative[-1] = 1.0

    def _frequency(self, cfg, split: str) -> torch.Tensor:
        """Fit interaction frequency of the task catalog, counted from the target column."""
        from src.model.tasks import TaskSamples
        from src.model.deyi import artifacts as deyi_artifacts

        canonical = TaskSamples(Path(str(cfg.data.root)), str(cfg.task), split,
                                dataset=deyi_artifacts.dataset(cfg))
        targets = np.asarray(canonical.target, dtype=np.int64)
        lengths = np.asarray(canonical.target_lengths, dtype=np.int64)
        if targets.size:
            valid = np.arange(targets.shape[1])[None, :] < lengths[:, None]
            flat = targets[valid]
            flat = flat[flat > 0]
        else:
            flat = np.zeros(0, dtype=np.int64)
        if not len(flat):
            return torch.zeros(self.size, dtype=torch.float64)
        position = np.searchsorted(self.rows, flat)
        position = np.clip(position, 0, max(self.size - 1, 0))
        inside = self.rows[position] == flat
        counts = np.bincount(position[inside], minlength=self.size)
        return torch.from_numpy(counts.astype(np.float64))

    def unit(self, index, *, device=None, dtype=torch.float32) -> torch.Tensor:
        """``e(i)`` of catalog *positions* (the order of ``catalog_rows``)."""
        values = np.asarray(index, dtype=np.int64).reshape(-1)
        return self.items.unit(self.rows[values], device=device, dtype=dtype)

    def draw(self, count: int, generator: torch.Generator, device=None) -> torch.Tensor:
        """``count`` catalog positions drawn i.i.d. from ``q`` (with replacement).

        The draw runs on the CPU on purpose: a seeded CPU generator makes the tail pool identical
        whether the statistics are estimated on a GPU or on the CPU, so the two runs differ only in
        their arithmetic and not in which items they happened to sample.
        """
        uniforms = torch.rand(int(count), generator=generator, dtype=torch.float64)
        return torch.searchsorted(self.cumulative.cpu(), uniforms).clamp_max(
            self.size - 1).to(torch.long)


class CatalogUnits:
    """``e(i)`` for the whole task catalog: resident on the device when it fits, else streamed.

    The head search reads every catalog row once *per query block*, so a resident matrix turns
    ``query_blocks`` full passes into one load. The resident copy is a cache and never a
    requirement: when it does not fit, the same math runs from the mmap with more IO, which is
    why the streamed path has to stay correct.
    """

    def __init__(self, catalog: TaskCatalog, *, device, budget_bytes: int = 0,
                 dtype=torch.float16, resident: str = "auto"):
        self.catalog = catalog
        self.items = catalog.items
        self.device = device
        self.dtype = dtype
        self.resident = None
        wanted = str(resident or "auto").lower()
        need = int(catalog.size) * int(catalog.items.dim) * 2
        if wanted == "never" or device.type != "cuda":
            return
        if wanted == "always" or (budget_bytes and need <= int(budget_bytes)):
            self.resident = self._materialise()

    def _materialise(self):
        catalog, items = self.catalog, self.items
        matrix = torch.empty((catalog.size, items.dim), dtype=torch.float16)
        for start in range(0, catalog.size, BLOCK):
            stop = min(start + BLOCK, catalog.size)
            rows = catalog.rows[start:stop]
            block = torch.from_numpy(np.asarray(items.embeddings[rows], dtype=np.float32))
            matrix[start:stop] = torch.nn.functional.normalize(block, dim=-1).to(torch.float16)
        return matrix.to(self.device)

    def block(self, start: int, stop: int):
        if self.resident is not None:
            return self.resident[start:stop]
        rows = self.catalog.rows[int(start):int(stop)]
        return self.items.unit(rows, device=self.device, dtype=self.dtype)

    def gather(self, index, *, device=None, chunk: int = 1 << 18):
        """Rows of ``e(i)`` for catalog positions, keeping ``index``'s leading shape.

        The head search asks with ``(queries, head)`` and expects ``(queries, head, dim)`` back, so
        flattening is an implementation detail here rather than part of the contract.
        """
        target = self.device if device is None else device
        wanted = torch.as_tensor(index, dtype=torch.long)
        shape = tuple(wanted.shape)
        values = wanted.reshape(-1)
        out = torch.empty((values.numel(), self.items.dim), dtype=self.dtype)
        for start in range(0, values.numel(), int(chunk)):
            stop = min(start + int(chunk), values.numel())
            piece = values[start:stop]
            if self.resident is not None:
                out[start:stop] = self.resident[piece.to(self.resident.device)].cpu()
            else:
                # The caller may ask with a device tensor (the estimator samples on the GPU),
                # and ``TaskCatalog.unit`` reads the fp16 catalog from its host memmap, so the
                # index has to come home first.
                out[start:stop] = self.catalog.unit(piece.detach().cpu().numpy(), dtype=self.dtype)
        return out.reshape(*shape, self.items.dim).to(target)
