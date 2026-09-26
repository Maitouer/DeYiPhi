"""Console + machine-readable progress for the data pipeline.

One run owns one directory::

    <project>/log/data/prepare/<run-id>/
        console.log      human narrative (banner, stage progress, closing summary)
        progress.json    current state snapshot, atomically replaced
        progress.jsonl   append-only event stream
        errors.log       warnings / tracebacks

The first lines are printed within two seconds of start so that a fresh directory
never looks empty, and every stage emits a line at least every ``heartbeat``
seconds even when a single unit of work takes longer than that.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path


def fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return "%.1f%s" % (n, unit)
        n /= 1024.0


def fmt_seconds(seconds):
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    if seconds < 60:
        return "%02ds" % seconds
    if seconds < 3600:
        return "%02dm%02ds" % (seconds // 60, seconds % 60)
    return "%dh%02dm" % (seconds // 3600, (seconds % 3600) // 60)


class Progress:
    """Stage-aware progress reporter (console + json + jsonl)."""

    def __init__(self, log_dir, stage_total, job_id=None, interval=5.0, heartbeat=10.0):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.log_dir / "progress.json"
        self.events_path = self.log_dir / "progress.jsonl"
        self.stage_total = int(stage_total)
        self.stage_index = 0
        self.stage = "startup"
        self.job_id = str(job_id if job_id is not None else os.environ.get("SLURM_JOB_ID", "local"))
        self.interval = float(interval)
        self.heartbeat = float(heartbeat)
        self.started = time.time()
        self.stage_started = self.started
        self.percent = 0.0
        self.eta_seconds = None
        self.counters = {}
        self.extra = {}
        self._last_print = 0.0
        self._write_state(status="starting")

    # ---------------------------------------------------------------- console
    def line(self, stage, message, force=False):
        now = time.time()
        if not force and now - self._last_print < self.interval:
            return
        self._last_print = now
        text = "[%s] [%-10s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), stage, message)
        print(text, flush=True)
        self.event("line", stage=stage, message=message)

    def banner(self, lines):
        for text in lines:
            print(text, flush=True)
        self.event("banner", lines=list(lines))

    # ----------------------------------------------------------------- stages
    def stage_begin(self, index, name, detail=""):
        self.stage_index = int(index)
        self.stage = name
        self.stage_started = time.time()
        self.percent = 0.0
        self.eta_seconds = None
        self.counters = {}
        self.extra = {}
        self.line(name, "START %s" % detail, force=True)
        self._write_state(status="running")

    def stage_end(self, detail="", seconds=None):
        seconds = time.time() - self.stage_started if seconds is None else seconds
        self.line(self.stage, "DONE %s (%.1fs)" % (detail, seconds), force=True)
        self.event("stage_end", stage=self.stage, seconds=seconds, detail=detail)
        self._write_state(status="running", last_stage_seconds=seconds)
        return seconds

    def tick(self, done, total, extra="", counters=None, force=False):
        """Report progress inside the current stage (rate limited + heartbeat)."""
        now = time.time()
        due = (now - self._last_print >= self.interval) or (now - self._last_print >= self.heartbeat)
        if not force and not due:
            return
        self.percent = 100.0 * done / total if total else 100.0
        elapsed = max(now - self.stage_started, 1e-6)
        rate = done / elapsed
        self.eta_seconds = (total - done) / rate if rate > 0 else None
        if counters:
            self.counters.update(counters)
        if extra:
            self.extra.update({"detail": extra})
        self.line(self.stage, "%d/%d (%.1f%%) %s rate=%.2f/s eta=%s" % (
            done, total, self.percent, extra, rate, fmt_seconds(self.eta_seconds)), force=force)
        self._write_state(status="running")

    def counters_update(self, **values):
        self.counters.update(values)
        self._write_state(status="running")

    # ----------------------------------------------------------------- events
    def event(self, kind, **fields):
        record = {"t": round(time.time(), 3), "kind": kind, "stage": self.stage}
        record.update(fields)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def fail(self, exc, resume_command="sbatch scripts/data/prepare.sbatch --resume"):
        elapsed = time.time() - self.started
        text = "%s: %s" % (type(exc).__name__, exc)
        self.line(self.stage, "FAIL %s" % text, force=True)
        print("[%s] [%-10s] resume with: %s" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                                self.stage, resume_command), flush=True)
        with (self.log_dir / "errors.log").open("a", encoding="utf-8") as handle:
            handle.write("=== %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            handle.write(traceback.format_exc())
        self.event("fail", error=text, seconds=elapsed)
        self._write_state(status="failed", error=text, resume_command=resume_command)

    def finish(self, detail="", **fields):
        elapsed = time.time() - self.started
        self.line("done", "DATA READY %s total=%s" % (detail, fmt_seconds(elapsed)), force=True)
        self.event("finish", seconds=elapsed, detail=detail, **fields)
        self._write_state(status="finished", total_seconds=elapsed)
        return elapsed

    # ------------------------------------------------------------------ state
    def _write_state(self, status, **overrides):
        state = {
            "job_id": self.job_id,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started)),
            "stage": self.stage,
            "stage_index": self.stage_index,
            "stage_total": self.stage_total,
            "status": status,
            "percent": round(self.percent, 2),
            "eta_seconds": None if self.eta_seconds is None else int(self.eta_seconds),
            "elapsed_seconds": int(time.time() - self.started),
            "counters": self.counters,
            "log_dir": str(self.log_dir),
        }
        state.update(overrides)
        # Per-process tmp name: two processes that share one log directory used to publish
        # through the same ``progress.json.tmp`` and one of them died with FileNotFoundError.
        temporary = self.state_path.parent / ("%s.%d.tmp" % (self.state_path.name, os.getpid()))
        temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)


def run_directory(log_root, job_id=None, when=None):
    """Create and return a fresh run directory plus update the ``latest`` symlink."""
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(when))
    run_id = "%s-%s" % (stamp, job_id if job_id is not None else os.getpid())
    directory = log_root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    latest = log_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_id)
    return directory
