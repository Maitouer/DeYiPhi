"""CPU preprocessing runtime; importing this module does not change process policy."""

import ctypes
import sys

import pyarrow as pa


def configure_runtime(num_threads):
    # Avoid synchronous huge-page compaction for transient preprocessing arrays. This changes
    # this process only, not the host or subsequent training jobs.
    if sys.platform == "linux":
        PR_SET_THP_DISABLE = 41
        ctypes.CDLL(None).prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)
    pa.set_cpu_count(int(num_threads))
    pa.set_io_thread_count(min(int(num_threads), 4))
