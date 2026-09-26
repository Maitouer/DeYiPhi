#!/usr/bin/env bash
# Tiger phi smoke: one SFT step, then a small beam check of the route-prefixed chain
# (``<g> <c0> <c1> <c2> <c3>``: the predictive route first, then the four catalog-constrained
# digits -- docs/v1_algorithm/03_phi.md section 16.2).
#
#   sbatch scripts/model/tiger/phi_smoke.sbatch [task] [mode]
#SBATCH --partition=debug
#SBATCH --qos=debug_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --job-name=tiger:phi:smoke
set -euo pipefail
cd ${DEYIPHI_ROOT:-.}
source scripts/model/common_logging.sh tiger/phi_smoke
export LOG_STAGE="model/tiger/phi_smoke"
PY=python
export CUDA_LAUNCH_BLOCKING=1
run="output/model/tiger_smoke/phi-$SLURM_JOB_ID"
task="${1:-short_video}"
mode="${2:-recent}"
echo "[smoke] sft task=$task mode=$mode $(date '+%H:%M:%S') run=$run"
TIGER_NPROC=1 "$PY" -u -m torch.distributed.run --standalone --nproc_per_node=1 \
    --module src.model.tiger.train --config config/model/tiger.yaml \
    --task "$task" --mode "$mode" --phi --steps 1 --nproc 1 --run-dir "$run"
echo "[smoke] beam $(date '+%H:%M:%S')"
"$PY" -u - <<PY2
import numpy as np, torch, os
from omegaconf import OmegaConf
from src.model.tiger import model as tiger_model
from src.model.tiger.data import TigerDataset
from src.model.tiger import evaluate as tiger_eval
run = "$run"
cfg = OmegaConf.load(os.path.join(run, "config.yaml"))
cfg.runtime.nproc = 1
m = tiger_model.Tiger(cfg)
weights = os.path.join(run, "model", "model.pt")
if os.path.exists(weights):
    m.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
m.eval().cuda()
d = TigerDataset(cfg, "test")
index = tiger_eval.PrefixIndex(np.load(f"{cfg.prepared_dir}/codes.npy", mmap_mode="r"),
                               np.load(f"{cfg.data_dir}/items/pids.npy", mmap_mode="r"),
                               int(cfg.codebook.width))
batch = d.batch(np.arange(4), device=torch.device("cuda"), training=False)
print("[smoke] phi_pred", m.phi_pred, "classes", m.anchor_width, "none", m.empty_g_id)
print("[smoke] batch ranges:", {k: (int(v.min()), int(v.max()))
                                for k, v in batch.items()
                                if k in ("interest_codes", "interest_mask")})
codes, scores = tiger_eval.generate(m, batch, index, beams=8)
print("[smoke] chain beam codes", codes.shape, "scores", scores.shape,
      "finite:", bool(np.isfinite(scores[:, 0]).all()))
keys, pids = index.lookup(codes[:, 0])
print("[smoke] top-1 catalog keys", keys[:4].tolist(), "pids", pids[:4].tolist())
PY2
echo "[smoke] done $(date '+%H:%M:%S')"
