"""Read canonical task rows and the shared physical-row-indexed Tiger codes."""

from pathlib import Path

import numpy as np
import torch

from src.model.deyi.consumer import load_states
from src.model.tasks import TaskSamples


def _compact_history_rows(samples, indices, mode, recent):
    """Build chronological item-row tensors without per-sample Python loops."""
    selected = np.asarray(indices, dtype=np.int64)
    blocks, masks = [], []
    for name in samples.channels:
        values = np.asarray(samples.histories[name][selected], dtype=np.int64)
        lengths = np.asarray(samples.lengths[name][selected], dtype=np.int64)
        blocks.append(values)
        masks.append(np.arange(values.shape[1])[None] < lengths[:, None])
    values = np.concatenate(blocks, axis=1)
    valid = np.concatenate(masks, axis=1)
    lengths = valid.sum(axis=1, dtype=np.int64)
    width = int(lengths.max()) if len(lengths) else 0
    compact = np.zeros((len(selected), width), dtype=np.int64)
    rows, columns = np.nonzero(valid)
    if len(rows):
        starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
        compact[rows, np.arange(len(rows)) - starts] = values[rows, columns]
    if mode != "recent":
        return compact, lengths
    kept = np.minimum(lengths, int(recent))
    recent_width = int(kept.max()) if len(kept) else 0
    output = np.zeros((len(selected), recent_width), dtype=np.int64)
    if recent_width:
        positions = lengths[:, None] - kept[:, None] + np.arange(recent_width)[None]
        mask = np.arange(recent_width)[None] < kept[:, None]
        output[mask] = compact[np.arange(len(selected))[:, None], positions][mask]
    return output, kept


def _device_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    tensor = torch.from_numpy(np.ascontiguousarray(value))
    if device.type == "cuda":
        # Pinned host memory makes the H2D copy asynchronous instead of leaving the GPU
        # waiting on the Python-side batch assembly (same trick as the reference Tiger).
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


class TigerDataset:
    def __init__(self, cfg, split):
        self.cfg = cfg
        # The predictive vocabulary was fitted on one dataset's canonical rows, so this reader has
        # to open *that* dataset: a task name alone is ambiguous when it exists in two of them.
        self.phi_pred = str(cfg.get("phi", {}).get("variant", "")) == "phi"
        # product always reads single_channel (user decision 2026-09-25): the DeYi arm only
        # exists there, and full/recent/deyi/phi must all be comparable on one dataset.
        self.samples = TaskSamples(cfg.data_dir, cfg.task, split,
                                   dataset=cfg.get("dataset"))
        self.codes = np.load(Path(cfg.prepared_dir) / "codes.npy", mmap_mode="r")
        from src.data.dataset import open_items

        self.pids = open_items(str(cfg.data_dir)).pids
        # phi (v6): every item body carries its own region token, the row's interest slots carry
        # (region, first codebook digit), and the target chain starts with the target's region.
        # Only the predictive-vocabulary mode survives: the retired item-region ``tree`` variant
        # went with the tree (docs/v1_algorithm/03_phi.md §16).
        self.phi = str(cfg.get("phi", {}).get("variant", "")) == "tree"
        self.phi_v7 = str(cfg.get('phi', {}).get('variant', '')) == 'v7'
        self.deyi_v7 = str(cfg.get('deyi', {}).get('variant', '')) == 'v7'
        self.phi_pg = str(cfg.get('phi', {}).get('variant', '')) in ('v10_pg', 'v100')
        # v100 keeps the one-chronological-sequence window: no per-channel recent budget.
        self.phi_v100 = str(cfg.get('phi', {}).get('variant', '')) == 'v100'
        self.window = None
        self.phi_v8 = str(cfg.get('phi', {}).get('variant', '')) in ('v8', 'v9', 'v9_fast', 'v10_pg', 'v100')
        if self.phi_pg:
            self.item_g = np.load(cfg.phi.item_codes, mmap_mode='r')
        self.fine_to_coarse = None
        if self.phi_pg and cfg.phi.get('hierarchical',False):
            self.fine_to_coarse = torch.load(cfg.phi.tree,map_location='cpu',weights_only=False)['fine_to_coarse'].numpy()
        self.user_item_g = None
        if self.phi_pg and cfg.phi.get('item_assignment') == 'user_candidates':
            split_folder = 'train' if split == 'validation' else split
            self.user_item_g = np.load(Path(cfg.phi.assignment_root)/split_folder/'item_g.npy',mmap_mode='r')
            if self.user_item_g.shape != (len(self.samples),32+self.samples.target.shape[1]):
                raise ValueError('User item G shape differs from canonical windows')
        self.deyi_v8 = str(cfg.get('deyi', {}).get('variant', '')) in ('v8', 'v9', 'v100')
        if (self.phi_v7 or self.deyi_v7 or self.deyi_v8
                or (self.phi_v8 and not self.phi_v100)):
            from src.data.windows_v7 import WindowSamples
            self.window = WindowSamples(self.samples)
        if self.phi_v7 or self.phi_v8:
            folder = Path(cfg.phi.assignment_root) / ('train' if split == 'validation' else split)
            if not np.array_equal(np.load(folder / 'source_rows.npy'), self.samples.source_rows):
                raise ValueError('v7 source rows differ from canonical samples')
            self.v7_codes = np.load(folder / 'codes.npy')
            self.v7_mask = np.load(folder / 'mask.npy')
            # v100 slots are channel free, so its export records the DeYi channel tag for
            # traceability only and the window comparison does not apply.
            self.v7_channels = None if self.phi_v100 else np.load(folder / 'channels.npy')
            if (not self.phi_v100
                    and not np.array_equal(self.v7_channels, self.window.slot_channels)):
                raise ValueError('v7 assignment channel order differs from the current window')
        self.leaf_g1 = None
        self.slot_g = self.slot_mask = self.slot_digit = None
        self.pred_tokens = self.pred_routes = None
        self.phi_none_id = 0
        if self.phi_pred:
            # The predictive vocabulary: per-slot tokens and per-target routes, both keyed by
            # ``source_row_idx`` so a permuted export can never be absorbed silently.
            prepared = "train" if split == "validation" else split
            tokens = torch.load(str(cfg.phi.tokens[prepared]), map_location="cpu",
                                weights_only=False)
            expected = np.asarray(self.samples.source_rows, dtype=np.int64)
            # Routes are decoder *supervision*, so they exist for the supervised splits only: the
            # test split still needs its encoder tokens, and demanding routes there would make an
            # evaluation impossible.
            routes = None
            if Path(str(cfg.phi.routes[prepared])).is_file():
                routes = torch.load(str(cfg.phi.routes[prepared]), map_location="cpu",
                                    weights_only=False)
            for name, payload in (("tokens", tokens), ("routes", routes)):
                if payload is None:
                    continue
                got = np.asarray(payload["source_row_idx"], dtype=np.int64)
                if not np.array_equal(got, expected):
                    raise ValueError("phi %s rows do not match the canonical %s split"
                                     % (name, prepared))
            self.pred_tokens, self.pred_routes = tokens, routes
            self.phi_none_id = int(cfg.phi.none_id)
        self.num_g1 = 0
        if self.phi:
            import torch as _torch

            payload = _torch.load(str(cfg.phi.tree), map_location="cpu", weights_only=False)
            self.leaf_g1 = np.asarray(payload["label_g1"], dtype=np.int64)
            self.num_g1 = int(payload["g1_profile"].shape[0])
            prepared = "train" if split == "validation" else split
            folder = Path(str(cfg.phi.assignment_root)) / cfg.task / prepared
            self.slot_g = np.load(folder / "g.npy")
            self.slot_mask = np.load(folder / "mask.npy")
            # The slot's second token is the interest state's own first codebook digit
            # (``tiger_code.py``); without it the v6 interest code has no second half.
            digit_path = folder / "st_a.npy"
            if not digit_path.is_file():
                raise FileNotFoundError(
                    "phi v6 needs the interest-slot codebook digit: %s "
                    "(run scripts/model/phi/tiger_code.py for this teacher)" % digit_path)
            self.slot_digit = np.load(digit_path)
        self.mode, self.recent = cfg.mode, int(cfg.recent_history[cfg.task])
        self.deyi_states = self.deyi_mask = None
        deyi = cfg.get("deyi", {})
        if self.deyi_v7 or self.deyi_v8:
            folder = Path(deyi.root) / cfg.task / 'k4'
            if not (folder / 'encoding_complete.json').exists():
                raise ValueError('v7 teacher encoding is incomplete')
            bank = folder / ('train' if split == 'validation' else split)
            if not np.array_equal(np.load(bank / 'source_rows.npy'), self.samples.source_rows):
                raise ValueError('v7 continuous source rows differ from canonical rows')
            if not np.array_equal(np.load(bank / 'channels.npy'), self.window.slot_channels):
                raise ValueError('v7 continuous slot channels differ from the window')
            # Copy-on-write mapping shares the immutable bank without materialising another file.
            self.deyi_states = torch.from_numpy(np.load(bank / 'states.npy', mmap_mode='c'))
            self.deyi_mask = torch.from_numpy(np.load(bank / 'mask.npy', mmap_mode='c'))
            if tuple(self.deyi_states.shape) != (len(self.samples), 4, 1024):
                raise ValueError('Unexpected v7 continuous state shape')
            if tuple(self.deyi_mask.shape) != (len(self.samples), 4):
                raise ValueError('Unexpected v7 continuous mask shape')
        elif bool(deyi.get("enabled", False)):
            from src.model.deyi.consumer import dataset_of, recent_of

            # New arm contract: <root>/<dataset>/<task>/k<K>_r<R>; dataset and recent come
            # from the consumer config (dataset / recent_history).
            states, mask = load_states(
                deyi.root, dataset_of(cfg), cfg.task, int(deyi.num_interests),
                "train" if split == "validation" else split,
                self.samples.source_rows, recent_of(cfg),
            )
            self.deyi_states, self.deyi_mask = states, mask
        history_lengths = sum(self.samples.lengths.values())
        self.lengths = (
            np.minimum(history_lengths, self.recent)
            if self.mode == "recent" else history_lengths
        )
        # One-shot materialised tensors (scripts/model/tiger/materialize.sbatch) replace the
        # per-step memmap gather + channel compaction; falls back to the canonical path when the
        # file is missing or a DeYi stream needs the extra per-row arrays.
        self.materialized = None
        store = Path("output/model/tiger/data") / cfg.dataset / cfg.task / cfg.mode / f"{split}.pt"
        # The predictive-vocabulary mode always builds its batch from the canonical rows: the
        # sequence cache predates routes/tokens and reading it would silently drop both.
        if (store.exists() and self.deyi_states is None and not self.phi_pred
                and not (self.phi_v7 or self.phi_v8)):
            payload = torch.load(store, map_location="cpu", weights_only=False)
            if int(payload["codebook_width"]) != int(cfg.codebook.width):
                raise ValueError(f"materialised sequences {store} use another codebook width")
            recorded = payload.get("history_order")
            if recorded is not None:
                recorded = [part for part in str(recorded).split(",") if part]
                current = [str(name) for name in self.samples.channels]
                if recorded != current:
                    raise ValueError(
                        "materialised sequences %s were built with channel order %s but the "
                        "canonical split now reads %s; re-run scripts/model/tiger/"
                        "materialize.sbatch" % (store, recorded, current))
            if int(payload["rows"]) != len(self.samples):
                raise ValueError(f"materialised sequences {store} have {payload['rows']} rows, "
                                 f"canonical split has {len(self.samples)}")
            self.materialized = payload

    def __len__(self):
        return len(self.samples)

    def _anchor_ids(self, codes):
        """Item codes -> region ids; leaves excluded from the tree use the ``none`` id."""
        c0 = np.asarray(codes)[..., 0].astype(np.int64)
        g1 = self.leaf_g1[c0]
        # Excluded leaves take the region head's "none" class (its last class).
        return np.where(g1 >= 0, g1, self.num_g1)

    def batch(self, indices, device, training=True):
        if self.materialized is not None:
            return self._materialized_batch(indices, device, training)
        selected = np.asarray(indices, dtype=np.int64)
        if self.window is not None:
            recent = [self.window.recent(i) for i in selected]
            lengths = np.array([len(x) for x in recent])
            item_rows = np.zeros((len(selected), max(1, int(lengths.max()))), dtype=np.int64)
            for i, items in enumerate(recent):
                item_rows[i, :len(items)] = [row for _, row in items]
        else:
            item_rows, lengths = _compact_history_rows(self.samples, selected, self.mode, self.recent)
        history = self.codes[item_rows].astype(np.int64, copy=False)
        history_mask = np.arange(history.shape[1]) < lengths[:, None]
        # Physical row zero is padding and has no published code.
        history[~history_mask] = 0
        batch = {"history_raw": history, "history_mask": history_mask}
        if self.phi_pg:
            anchors = np.asarray(self.item_g[item_rows] if self.user_item_g is None else
                                 self.user_item_g[selected,:item_rows.shape[1]], dtype=np.int64)
            if (anchors[history_mask] < 0).any():
                raise ValueError('Missing global item G for a recent item')
            batch['history_anchor'] = np.where(history_mask, anchors, 0)
        if self.phi_v7 or self.phi_v8:
            batch['interest_codes'] = self.v7_codes[selected].astype(np.int64)
            batch['interest_mask'] = self.v7_mask[selected].astype(bool)
            batch['interest_channels'] = np.broadcast_to(self.v7_channels, (len(selected), 4)).copy()
        if self.phi:
            batch["history_anchor"] = self._anchor_ids(history)
            slots = self.slot_g[selected]
            slot_mask = self.slot_mask[selected].astype(bool)
            digits = self.slot_digit[selected]
            # Masked slots carry legal ids (the model embeds every slot, then masks it).
            batch["interest_anchor"] = np.stack(
                [np.where(slot_mask, slots[..., 0], self.num_g1),
                 np.where(slot_mask, digits, 0)], axis=-1)
            batch["interest_anchor_mask"] = slot_mask
        if self.phi_pred:
            none_id = self.phi_none_id
            # The phi artifacts are torch tensors; this reader assembles numpy batches and hands
            # them to ``_device_tensor`` at the end of ``batch``.
            codes = np.asarray(self.pred_tokens["codes"][torch.as_tensor(selected)].numpy(),
                               dtype=np.int64)
            interest_mask = np.asarray(self.pred_tokens["mask"][torch.as_tensor(selected)].numpy(),
                                       dtype=bool)
            # Masked slots carry the ``none`` class: the model embeds every slot and masks it,
            # which keeps the encoder length fixed (one position per DeYi interest).
            batch["interest_codes"] = np.where(interest_mask, codes, none_id)
            batch["interest_mask"] = interest_mask
            if training:
                if self.pred_routes is None:
                    # ``PHI_ROUTE_MODE=none`` is Analysis III's "memory only" arm: it has no route
                    # artifact *by design*, so the missing file is only an error for a routing arm.
                    # (The previous message also referenced ``cfg``, which this method never had in
                    # scope, so a legitimate run died with NameError instead of a clear error.)
                    if str(self.cfg.phi.get("route_mode", "contextual")) != "none":
                        raise FileNotFoundError(
                            "the phi route labels are missing for this split: run the phi `route` "
                            "stage (python -m src.model.phi.cli route --task %s)" % self.cfg.task)
                else:
                    route = np.asarray(self.pred_routes["route"][torch.as_tensor(selected)].numpy(),
                                       dtype=np.int64)
                    route_mask = np.asarray(
                        self.pred_routes["target_mask"][torch.as_tensor(selected)].numpy(), dtype=bool)
                    # The chain scores one *position* per positive, so the route is flattened exactly
                    # like ``target_raw``/``target_owner``; the two masks must agree, otherwise the
                    # routes would be attached to the wrong targets.
                    width = route_mask.shape[1]
                    sample_mask = (np.arange(width)[None]
                                   < np.asarray(self.samples.target_lengths, dtype=np.int64)[selected]
                                   [:, None])
                    if not np.array_equal(route_mask, sample_mask):
                        raise ValueError(
                            "phi routes and the canonical targets disagree about which targets are "
                            "supervised in this split")
                    batch["target_route"] = np.where(route_mask, route, none_id)[route_mask][:, None]
                    batch["target_route_mask"] = route_mask[route_mask]
        if self.deyi_states is not None:
            positions = torch.as_tensor(selected, dtype=torch.long)
            batch["continuous_states"] = self.deyi_states.index_select(0, positions)
            batch["continuous_mask"] = self.deyi_mask.index_select(0, positions)
        if training:
            targets = np.asarray(self.samples.target[selected], dtype=np.int64)
            target_lengths = np.asarray(self.samples.target_lengths[selected], dtype=np.int64)
            target_mask = np.arange(targets.shape[1])[None] < target_lengths[:, None]
            batch.update(
                target_raw=self.codes[targets[target_mask]].astype(np.int64, copy=False),
                target_owner=np.nonzero(target_mask)[0].astype(np.int64, copy=False),
            )
            if self.phi_pg:
                anchors = np.asarray(self.item_g[targets[target_mask]] if self.user_item_g is None else
                                     self.user_item_g[selected,32:][target_mask], dtype=np.int64)
                if (anchors < 0).any():
                    raise ValueError('Missing global item G for a target item')
                batch['target_anchor'] = (anchors[:, None] if self.fine_to_coarse is None else
                    np.stack([self.fine_to_coarse[anchors],anchors],axis=1))
            if self.phi:
                # ``[N, 1]``: the chain contract is one *position* per anchor level, exactly like the
                # three digit positions next to it (``decode_chain`` indexes ``shape[1]``).
                batch["target_anchor"] = self._anchor_ids(
                    self.codes[targets[target_mask]])[:, None]
        return {key: _device_tensor(value, device) for key, value in batch.items()}

    def _anchor_tensors(self, selected):
        """Per-item, per-slot and per-target anchors for a batch of sample indices."""
        import torch as _torch

        store = self.materialized
        digits = store["history_digits"].index_select(0, selected).numpy()
        batch = {"history_anchor": self._anchor_ids(digits)}
        slots = self.slot_g[selected.numpy()]
        slot_mask = self.slot_mask[selected.numpy()].astype(bool)
        slot_digit = self.slot_digit[selected.numpy()]
        batch["interest_anchor"] = np.stack(
            [np.where(slot_mask, slots[..., 0], self.num_g1),
             np.where(slot_mask, slot_digit, 0)], axis=-1).astype(np.int64)
        batch["interest_anchor_mask"] = slot_mask
        target_digits = store["target_digits"].index_select(0, selected).numpy()
        target_mask = store["target_mask"].index_select(0, selected).numpy().astype(bool)
        batch["target_anchor"] = self._anchor_ids(target_digits)[target_mask][:, None]
        return {key: _torch.as_tensor(value) for key, value in batch.items()}

    def _materialized_batch(self, indices, device, training):
        """Slice the pre-built tensors: no parsing, no per-row Python, pinned H2D copies."""
        store = self.materialized
        selected = torch.as_tensor(np.asarray(indices, dtype=np.int64))
        anchors = self._anchor_tensors(selected) if self.phi else {}
        lengths = store["history_lengths"].index_select(0, selected).to(torch.long)
        items = int(lengths.max()) if lengths.numel() else 0
        history = store["history_digits"].index_select(0, selected)[:, :items].to(torch.long)
        batch = {"history_raw": history,
                 "history_mask": torch.arange(items).unsqueeze(0) < lengths.unsqueeze(1)}
        if training:
            target_digits = store["target_digits"].index_select(0, selected).to(torch.long)
            target_mask = store["target_mask"].index_select(0, selected).bool()
            batch["target_raw"] = target_digits[target_mask]
            batch["target_owner"] = torch.arange(selected.numel()).unsqueeze(1).expand_as(
                target_mask)[target_mask]
        batch.update(anchors)
        moved = {}
        for key, value in batch.items():
            tensor = value if key == "history_mask" or key == "target_mask" else value
            if device.type == "cuda":
                moved[key] = tensor.pin_memory().to(device, non_blocking=True)
            else:
                moved[key] = tensor.to(device)
        return moved
