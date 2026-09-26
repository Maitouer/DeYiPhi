"""Declarative dataset and task definitions for the recommendation data pipeline.

The channel order declared here is authoritative. It is the order the paper defines
(``H_video + H_ad`` and ``H_video + H_product``), the order every split file records in its
own header, and the order the reader concatenates channels in. Nothing downstream infers
channel order from tensor key names.

``history`` is a *list*, not a mapping, so the order is explicit syntax rather than a
property of how YAML happens to preserve keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class History:
    channel: str
    field: str
    limit: int


@dataclass(frozen=True)
class Task:
    dataset: str
    name: str
    primary: str
    histories: tuple[History, ...]
    target_field: str
    target_limit: int
    test_file: str
    test_history: dict[str, str]
    test_target: str
    behaviors: tuple[str, ...]

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(history.channel for history in self.histories)

    def history(self, channel: str) -> History:
        for history in self.histories:
            if history.channel == channel:
                return history
        raise KeyError("%s/%s has no channel %r" % (self.dataset, self.name, channel))


@dataclass(frozen=True)
class Dataset:
    name: str
    tasks: tuple[Task, ...]

    def task(self, name: str) -> Task:
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError("dataset %s has no task %r" % (self.name, name))


@dataclass(frozen=True)
class PipelineConfig:
    source: Path
    embeddings: Path
    output: Path
    item_dir: str
    log_root: Path
    runtime: dict
    min_primary_history: int
    datasets: tuple[Dataset, ...]
    task_owners: dict[str, str]
    raw: dict

    def dataset(self, name: str) -> Dataset:
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        raise KeyError("no dataset named %r" % name)


def _required(mapping, key, where):
    if key not in mapping:
        raise ValueError("%s is missing the required key %r" % (where, key))
    return mapping[key]


def _build_task(dataset_name: str, name: str, spec) -> Task:
    where = "datasets.%s.%s" % (dataset_name, name)
    history_spec = _required(spec, "history", where)
    if not isinstance(history_spec, list) or not history_spec:
        raise ValueError(
            "%s.history must be a non-empty list of {channel, field, limit}; "
            "the list order is the channel order" % where
        )
    histories = tuple(
        History(
            str(_required(entry, "channel", where)),
            str(_required(entry, "field", where)),
            int(_required(entry, "limit", where)),
        )
        for entry in history_spec
    )
    channels = [history.channel for history in histories]
    if len(set(channels)) != len(channels):
        raise ValueError("%s declares a duplicate channel: %s" % (where, channels))
    primary = str(_required(spec, "primary", where))
    if primary not in channels:
        raise ValueError("%s.primary=%r is not one of %s" % (where, primary, channels))
    target = _required(spec, "target", where)
    test_history = {str(k): str(v) for k, v in _required(spec, "test_history", where).items()}
    if sorted(test_history) != sorted(channels):
        raise ValueError(
            "%s.test_history must cover exactly %s, got %s"
            % (where, channels, sorted(test_history))
        )
    return Task(
        dataset=dataset_name,
        name=name,
        primary=primary,
        histories=histories,
        target_field=str(_required(target, "field", where + ".target")),
        target_limit=int(_required(target, "limit", where + ".target")),
        test_file=str(_required(spec, "test", where)),
        test_history=test_history,
        test_target=str(_required(spec, "test_target", where)),
        behaviors=tuple(str(item) for item in spec.get("behaviors", ())),
    )


def load_config(path="config/data.yaml") -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("%s must contain a YAML mapping" % path)
    datasets = tuple(
        Dataset(str(name), tuple(
            _build_task(str(name), str(task_name), spec)
            for task_name, spec in tasks.items()
        ))
        for name, tasks in _required(raw, "datasets", "config").items()
    )
    if not datasets or any(not dataset.tasks for dataset in datasets):
        raise ValueError("every dataset must declare at least one task")
    rules = raw.get("rules") or {}
    min_primary_history = int(rules.get("min_primary_history", 0))
    if min_primary_history < 1:
        raise ValueError("rules.min_primary_history must be a positive integer")
    for dataset in datasets:
        for task in dataset.tasks:
            if task.history(task.primary).limit < min_primary_history:
                raise ValueError(
                    "%s/%s: primary channel limit %d is below min_primary_history %d"
                    % (dataset.name, task.name, task.history(task.primary).limit,
                       min_primary_history)
                )
    task_owners = {}
    ambiguous = set()
    for dataset in datasets:
        for task in dataset.tasks:
            if task_owners.setdefault(task.name, dataset.name) != dataset.name:
                ambiguous.add(task.name)
    explicit = {str(k): str(v) for k, v in (raw.get("owners") or {}).items()}
    missing = sorted(name for name in ambiguous if name not in explicit)
    if missing:
        raise ValueError(
            "these task names exist in more than one dataset, so config 'owners' must say "
            "which dataset a bare task name means: %s" % missing
        )
    for name, owner in explicit.items():
        if name not in task_owners:
            raise ValueError("owners.%s names a task that no dataset declares" % name)
        if owner not in {dataset.name for dataset in datasets}:
            raise ValueError("owners.%s names an unknown dataset %r" % (name, owner))
        task_owners[name] = owner
    return PipelineConfig(
        source=Path(str(_required(raw, "source", "config"))),
        embeddings=Path(str(_required(raw, "embeddings", "config"))),
        output=Path(str(_required(raw, "output", "config"))),
        item_dir=str(_required(raw, "item_dir", "config")),
        log_root=Path(str(_required(raw, "log_root", "config"))),
        runtime=dict(raw.get("runtime") or {}),
        min_primary_history=min_primary_history,
        datasets=datasets,
        task_owners=task_owners,
        raw=raw,
    )


def all_tasks(config: PipelineConfig) -> tuple[Task, ...]:
    return tuple(task for dataset in config.datasets for task in dataset.tasks)
