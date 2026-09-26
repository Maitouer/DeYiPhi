"""Unified Tiger entry point for all RecIF tasks and history views."""

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from src.model.phi import artifacts as phi_paths

DEFAULT_CONFIG = "config/model/tiger.yaml"

RUNTIME = {
    "python": "python",
    "nproc": 4,
    "device": "cuda",
    "eval_batch_size": int(os.environ.get("TIGER_EVAL_BATCH", 32)),
    "workers": 2,
    "cpu_threads": 4,
    "gradient_checkpointing": True,
}


def runtime_config(nproc=None, global_batch_size=128):
    """Build a valid topology without changing the scientific global batch."""
    requested = nproc if nproc is not None else os.environ.get("TIGER_NPROC", RUNTIME["nproc"])
    ranks = int(requested)
    global_batch = int(global_batch_size)
    if ranks < 1 or global_batch % ranks:
        raise ValueError(
            f"Tiger nproc={ranks} must be positive and divide global_batch_size={global_batch}"
        )
    # Gradient checkpointing trades memory for recompute; larger models/batches may prefer the
    # raw activations (also keeps a real, non-placeholder GPU memory footprint).
    checkpointing = os.environ.get("TIGER_GRADIENT_CHECKPOINTING", "1") != "0"
    return OmegaConf.create({**RUNTIME, "nproc": ranks, "micro_batch_size": global_batch // ranks,
                             "gradient_checkpointing": checkpointing})


def validate_parallelism(cfg, world):
    """Validate the declared Tiger topology before allocating model state."""
    expected = int(cfg.runtime.nproc)
    if expected < 1:
        raise ValueError(f"Tiger runtime requires a positive rank count, got nproc={expected}")
    if world not in (1, expected):
        raise ValueError(f"Tiger expected one CPU process or {expected} DDP ranks, got {world}")
    if str(cfg.runtime.device).startswith("cuda") and world != expected:
        raise ValueError(f"Tiger CUDA runs require exactly {expected} DDP ranks")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if str(cfg.runtime.device).startswith("cuda") and visible and visible != "-1":
        allocated = len([value for value in visible.split(",") if value])
        if allocated != expected:
            raise ValueError(
                f"Tiger DDP ranks ({expected}) must match the {allocated} GPUs allocated by Slurm"
            )


def load_config(config=DEFAULT_CONFIG, task=None, mode=None, run_dir=None, deyi_k=None,
                nproc=None, phi=False, deyi_root=None,
                phi_deyi_k=None, phi_deyi_arm=None, phi_v7=False, deyi_v7=False, phi_v8=False, deyi_v8=False, phi_v9=False, deyi_v9=False, phi_v9_fast=False, phi_v10_pg=False, phi_pg_root=None, phi_no_user_prefix=False,
                phi_v100=False, deyi_v100=False):
    if phi_v100 and any([phi_v8, deyi_v8, phi_v9, deyi_v9, phi_v9_fast, phi_v7, deyi_v7,
                         phi, phi_v10_pg, deyi_v100, deyi_k is not None]):
        raise ValueError("--phi-v100 selects its own conditioning and cannot be combined")
    if deyi_v100 and any([phi_v100, phi_v10_pg, phi_v8, deyi_v8, phi_v9, deyi_v9, phi_v9_fast,
                          phi_v7, deyi_v7, phi]):
        raise ValueError("--deyi-v100 selects its own conditioning and cannot be combined")
    if phi_no_user_prefix and not phi_v10_pg:
        raise ValueError('Removing user prefix requires PG')
    if phi_v10_pg and any([phi_v8, deyi_v8, phi_v9, deyi_v9, phi_v9_fast, phi_v7, deyi_v7, phi, deyi_k is not None]):
        raise ValueError('PG cannot be combined with another conditioning variant')
    if sum(bool(x) for x in [phi_v8, deyi_v8, phi_v9, deyi_v9, phi_v9_fast, phi_v7, deyi_v7, phi, deyi_k is not None]) > 1 and (phi_v8 or deyi_v8 or phi_v9 or deyi_v9 or phi_v9_fast):
        raise ValueError('Select one v8 conditioning variant')
    if deyi_k is not None and int(deyi_k) < 1:
        raise ValueError("deyi-k must be positive")
    cfg = OmegaConf.load(config)
    cfg.task = task or cfg.task
    cfg.mode = mode or cfg.mode
    if cfg.task not in cfg.tasks:
        raise ValueError(f"Unknown task {cfg.task!r}; choose from {list(cfg.tasks)}")
    if cfg.mode not in ("full", "recent"):
        raise ValueError(f"Unknown mode {cfg.mode!r}; choose full or recent")
    recent_override = os.environ.get("TIGER_RECENT_HISTORY")
    if recent_override is not None:
        cfg.recent_history[cfg.task] = int(recent_override)
        if cfg.recent_history[cfg.task] < 1:
            raise ValueError("TIGER_RECENT_HISTORY must be positive")
    # Block A of the explicit-memory analysis (docs/v1_algorithm/06_analysis_baseline.md §1):
    # one global learnable token, shared by every user, prepended to the interaction sequence and
    # trained with the ordinary next-item objective. Selected through the environment so the sft
    # and eval stages resolve the same arm, exactly like TIGER_RECENT_HISTORY.
    _dummy_flag = str(os.environ.get("TIGER_DUMMY_MEMORY", "")).strip().lower()
    dummy_memory = _dummy_flag in ("1", "true", "yes", "on")
    # Capacity-matched control: ``TIGER_DUMMY_SLOTS`` (default 1) sets how many shared learnable
    # tokens the dummy memory contributes, so Block A can be run at the same K as the other blocks.
    dummy_slots = int(os.environ.get("TIGER_DUMMY_SLOTS") or 1)
    if dummy_slots < 1:
        raise ValueError("TIGER_DUMMY_SLOTS must be positive")
    cfg.model.dummy_memory = bool(dummy_memory)
    cfg.model.dummy_slots = dummy_slots
    if dummy_memory and (phi or deyi_k is not None):
        raise ValueError("dummy memory cannot be combined with phi or DeYi conditioning")
    cfg.runtime = runtime_config(nproc, cfg.train.global_batch_size)
    configured_deyi = cfg.get("deyi", {})
    configured_phi = cfg.get("phi", {})
    # Only the phi variant survives (docs/new/phi_v6_final.md §8): the Phi residual and
    # joint codebooks, their ``src/model/phi`` package and their interest-code data are deleted.
    phi_enabled = bool(phi)
    # ``phi`` now means the predictive vocabulary of docs/v1_algorithm/03_phi.md: per-slot
    # ``<|g_k|>`` tokens in the encoder and a per-target route token in front of the native SID.
    # The retired item-region ``tree`` variant is gone with the tree it read.
    variant = "phi" if phi else ""
    # ``deyi_root`` selects which frozen DeYi teacher to condition on (default ``output/model/deyi``).
    # A non-default root is one alternative *arm* (e.g. ``output/model/compress/cause``): it keeps its
    # own run directory, so two teachers of the same task sit side by side instead of overwriting
    # each other.
    default_deyi_root = str(configured_deyi.get("root", "output/model/deyi"))
    deyi_variant = ""
    if deyi_root is not None and str(deyi_root).rstrip("/") != default_deyi_root.rstrip("/"):
        deyi_variant = Path(str(deyi_root)).name
    # phi assets are keyed by the frozen teacher (same rule as the phi pipeline itself).
    # The teacher is selected *independently* of the DeYi continuous-state conditioning: a plain
    # phi run (no DeYi states, exactly like full/recent) still has to name the tree it consumes.
    #   --phi-deyi-k / --phi-deyi-arm  -> teacher only (defaults: --deyi-k / a non-default --deyi-root)
    teacher_k = int(phi_deyi_k if phi_deyi_k is not None else (deyi_k or 4))
    teacher_arm = str(phi_deyi_arm if phi_deyi_arm is not None else deyi_variant)
    # The phi pipeline's own overrides win here too: an arm may be empty (production teacher) or
    # literally ``deyi`` (HPO arm), and it decides which vocabulary this run consumes.
    if os.environ.get("PHI_DEYI_K"):
        teacher_k = int(os.environ["PHI_DEYI_K"])
    if os.environ.get("PHI_DEYI_ARM") is not None:
        teacher_arm = os.environ["PHI_DEYI_ARM"]
    # The teacher's old/recent cut: the phi pipeline's own override wins, otherwise the config's
    # per-task window. One value feeds both the vocabulary path and the arm name below, so a run
    # can never read one teacher and be named after another.
    teacher_recent = int(os.environ.get("PHI_DEYI_RECENT") or cfg.recent_history[cfg.task])
    # ``PHI_OUTPUT_DIR`` lets a downstream arm condition on an alternative vocabulary of the same
    # frozen teacher without editing the shared config (Analysis II's euclidean/cosine variants).
    phi_output_dir = os.environ.get("PHI_OUTPUT_DIR") or configured_phi.get(
        "output_dir", "output/model/phi")
    phi_root = str(phi_paths.arm_dir(
        phi_output_dir,
        str(cfg.get("dataset", "")), str(cfg.task), teacher_k, teacher_recent))
    cfg.deyi = OmegaConf.create({
        "enabled": deyi_k is not None,
        "root": str(deyi_root if deyi_root is not None else default_deyi_root),
        "state_dim": int(configured_deyi.get("state_dim", 1024)),
        "num_interests": int(deyi_k or 0),
    })
    # Analysis III (05_analysis_design.md §16-17): the *generation interface* is the controlled
    # variable, so the three arms share one frozen vocabulary and differ only in how memory guides
    # the target -- ``contextual`` keeps the user-dependent route, ``static`` uses the item-only
    # route of the same vocabulary, ``none`` drops the route token entirely.
    route_mode = str(os.environ.get("PHI_ROUTE_MODE") or configured_phi.get("route_mode", "contextual"))
    if route_mode not in ("contextual", "static", "none"):
        raise ValueError("phi.route_mode must be contextual|static|none, got %r" % route_mode)
    route_dir = {"contextual": "routes", "static": "routes_static",
                 "none": "routes_absent"}[route_mode]
    cfg.phi = OmegaConf.create({
        "enabled": phi_enabled,
        "variant": variant,
        # One frozen teacher's predictive vocabulary: <output_dir>/<dataset>/<task>/k<K>_r<R>.
        # Every artifact below is written by the phi chain and read here through the path helper,
        # so producer and consumer cannot name different arms.
        "root": phi_root,
        "codebook": str(Path(phi_root) / "codebook.pt"),
        "tokens": {name: str(Path(phi_root) / "tokens" / ("%s.pt" % name))
                   for name in ("train", "test")},
        # ``routes_absent`` is a directory that is never written: the data path drops the route
        # when the file is missing, which is exactly the "memory only" arm.
        "routes": {name: str(Path(phi_root) / route_dir / ("%s.pt" % name))
                   for name in ("train", "test")},
        "route_mode": route_mode,
        # The token vocabulary: L shared predictive concepts plus the ``none`` sentinel that a
        # masked slot / an unattributed target carries. Sizes come from the frozen codebook so the
        # model can be built without re-reading tensors.
        "vocabulary": 0,
        "none_id": 0,
        "token_names": [],
        "dataset": str(cfg.get("dataset", "")),
        # The r<R> of the frozen teacher this arm consumes (== the run's own recent window: the
        # consumer does not get a second knob, deyi/config.yaml states the same in one place).
        "recent": teacher_recent,
        # Recorded for traceability: which frozen teacher this run's vocabulary came from.
        "deyi_k": teacher_k,
        "deyi_arm": teacher_arm,
    })
    if phi_enabled:
        codebook_path = Path(str(cfg.phi.codebook))
        if not codebook_path.is_file():
            raise FileNotFoundError(
                "the predictive vocabulary of this teacher is missing: %s "
                "(run scripts/model/phi/chain.sbatch %s %d %d first)"
                % (codebook_path, cfg.task, teacher_k, cfg.phi.recent))
        import torch as _torch

        payload = _torch.load(str(codebook_path), map_location="cpu", weights_only=False)
        cfg.phi.vocabulary = int(payload["center"].shape[0])
        cfg.phi.token_names = list(payload.get("tokens") or [])
        cfg.phi.none_id = int(cfg.phi.vocabulary)          # the sentinel is the last class
    if phi:
        # The route head *is* the chain's first position, followed by the four codebook digits:
        # <g*><c0><c1><c2><c3>. Its width came from the frozen codebook above, and the digit width
        # from the runner's own codebook, so nothing else has to be sized here.
        cfg.phi.digit_width = int(cfg.codebook.width)
    cfg.task_spec = cfg.tasks[cfg.task]
    if phi:
        # The vocabulary's own arm name keeps this run distinct from the native run of the same
        # (task, mode) and from a vocabulary fitted for another teacher.
        suffix = "_phi_k%d_r%d" % (teacher_k, int(cfg.phi.recent))
        if cfg.deyi.enabled:
            suffix += f"_state{int(deyi_k)}"
    else:
        suffix = f"_deyi_k{int(deyi_k)}" if deyi_k is not None else ""
    if suffix and deyi_variant and not phi:
        suffix += f"_{deyi_variant}"
    if dummy_memory:
        suffix += "_dummy" if dummy_slots == 1 else "_dummy%d" % dummy_slots
    cfg.experiment = f"{cfg.task}_{cfg.mode}{suffix}"
    cfg.prepared_dir = "output/model/item/tiger_codebook"
    recent = int(cfg.recent_history[cfg.task])
    default_root = Path(cfg.output_dir)
    if phi:
        # Same layout as the other arms: <dataset>/<task>/phi_r<R>_k<K>[/state<N>]/seed<seed>.
        # A state-conditioned variant keeps its own directory instead of overwriting the
        # state-free one.
        default_root /= cfg.task
        arm = ("phi_r%d_k%d" % (int(cfg.phi.recent), teacher_k) if str(cfg.mode) == "recent"
               else "phi_%s_r%d_k%d" % (cfg.mode, int(cfg.phi.recent), teacher_k))
        # An alternative vocabulary root (Analysis II's euclidean/cosine quantizations) keeps its
        # own arm, otherwise two partitions of the same frozen teacher would share one run directory.
        if str(phi_output_dir) != str(configured_phi.get("output_dir", "output/model/phi")):
            arm = "%s_%s" % (arm, Path(str(phi_output_dir)).name)
        # One run directory per generation interface: the three Analysis III arms are the same
        # memory with a different target format, so they must not share a checkpoint.
        if route_mode != "contextual":
            arm = "%s_%s" % (arm, route_mode)
        default_root /= (arm if not cfg.deyi.enabled
                         else "%s_state%d" % (arm, int(deyi_k)))
    elif cfg.deyi.enabled:
        # The arm leads with what conditions it (``deyi``), not with the history view, so it
        # cannot be mistaken for a plain recent arm:
        #   <dataset>/<task>/deyi_r<R>_k<K>[/<arm>]/seed<seed>
        default_root /= cfg.task
        # ``recent`` keeps the historical name; another history view (e.g. ``full``, used by the
        # K-sweep control) must not share the arm directory, or the second run would ``--resume``
        # into the first one's checkpoint.
        arm_name = ("deyi_r%d_k%d" % (recent, int(deyi_k)) if str(cfg.mode) == "recent"
                    else "deyi_%s_r%d_k%d" % (cfg.mode, recent, int(deyi_k)))
        default_root /= arm_name
        if deyi_variant:
            default_root /= deyi_variant
    else:
        default_root /= cfg.task
        # arm name carries the recent window: different ``recent`` values are different
        # models and must never share a run directory.
        arm = "%s_r%d" % (cfg.mode, recent)
        if dummy_memory:
            # The Dummy control is the native model plus shared trainable tokens, not a
            # memory-bank arm, so it keeps its own checkpoint (one directory per slot count).
            arm += "_dummy" if dummy_slots == 1 else "_dummy%d" % dummy_slots
        default_root /= arm
    cfg.run_dir = run_dir or str(
        Path(cfg.output_dir) / cfg.dataset
        / default_root.relative_to(cfg.output_dir) / f"seed{cfg.seed}")
    if deyi_v7 and (phi_v7 or phi or deyi_k is not None):
        raise ValueError('Select exactly one v7 conditioning variant')
    if phi_v7:
        if phi or deyi_k is not None:
            raise ValueError('v7 cannot be combined with v6/continuous conditioning')
        import torch
        root = Path('output/model/phi/v7') / cfg.task / 'k4'
        payload = torch.load(root / 'tree.pt', map_location='cpu', weights_only=False)
        cfg.mode = 'recent'
        cfg.phi = OmegaConf.create({'enabled': True, 'variant': 'v7', 'tree': str(root / 'tree.pt'),
            'assignment_root': str(root), 'leaves': len(payload['leaf_profile'])})
        cfg.experiment = f'{cfg.task}_phi_v7_k4'
        cfg.run_dir = run_dir or str(Path(cfg.output_dir) / 'phi' / cfg.task / 'v7_k4' / f'seed{cfg.seed}')
    if deyi_v7:
        cfg.mode = 'recent'
        cfg.deyi = OmegaConf.create({'enabled': True, 'variant': 'v7',
            'root': 'output/model/deyi/v7', 'state_dim': 1024, 'num_interests': 4})
        cfg.experiment = f'{cfg.task}_deyi_v7_k4'
        cfg.run_dir = run_dir or str(Path(cfg.output_dir) / 'deyi' / cfg.task / 'v7_k4' / f'seed{cfg.seed}')
    if phi_v8 or deyi_v8 or phi_v9 or deyi_v9 or phi_v9_fast:
        version = 'v9_fast' if phi_v9_fast else ('v9' if phi_v9 or deyi_v9 else 'v8')
        cfg.mode = 'recent'
        teacher_root = Path('output/model/deyi') / ('v9' if phi_v9_fast else version) / cfg.task / 'k4'
        if phi_v8 or phi_v9 or phi_v9_fast:
            import torch
            root = Path('output/model/phi') / version / cfg.task / 'k4'
            if not (root / 'complete.json').exists():
                raise ValueError('Complete v8 tree and assignments before SFT')
            payload = torch.load(root / 'tree.pt', map_location='cpu', weights_only=False)
            cfg.phi = OmegaConf.create({'enabled': True, 'variant': version, 'tree': str(root / 'tree.pt'),
                'assignment_root': str(root), 'teacher_root': str(teacher_root),
                'leaves': len(payload['leaf_prototypes']), 'embedding_trainable': True})
            if phi_v9_fast:
                summary = json.loads((root / 'complete.json').read_text())
                cfg.phi.approximation = {'view_target_met': summary['view_target_met'],
                    'fit_retained_by_slot': summary['fit_retained_by_slot'], 'teacher_retrained': False}
        else:
            cfg.deyi = OmegaConf.create({'enabled': True, 'variant': version,
                'root': f'output/model/deyi/{version}', 'state_dim': 1024, 'num_interests': 4})
        kind = 'phi' if phi_v8 or phi_v9 or phi_v9_fast else 'deyi'
        cfg.experiment = f'{cfg.task}_{kind}_{version}_k4'
        cfg.run_dir = run_dir or str(Path(cfg.output_dir) / kind / cfg.task / f'{version}_k4' / f'seed{cfg.seed}')
    if phi_v100:
        import torch
        root = Path('output/model/phi/v100') / cfg.task / 'k4'
        if not (root / 'complete.json').exists():
            raise ValueError('Complete the v100 Phi fit and item labels before SFT')
        summary = json.loads((root / 'complete.json').read_text())
        if summary.get('task') not in (None, cfg.task):
            raise ValueError('Phi v100 artifacts belong to task %s, not %s'
                             % (summary.get('task'), cfg.task))
        payload = torch.load(root / 'tree.pt', map_location='cpu', weights_only=False)
        if payload.get('root_count') != 1 or payload.get('hierarchical'):
            raise ValueError('Phi v100 ships one shared, non-hierarchical tree')
        cfg.mode = 'recent'
        # The downstream window is the teacher's window: read it back, never re-derive it.
        budget = int(summary['recent_budget'])
        override = os.environ.get('TIGER_RECENT_HISTORY')
        if override is not None and int(override) != budget:
            raise ValueError('v100 recent window must match the frozen teacher export')
        cfg.recent_history[cfg.task] = budget
        cfg.phi = OmegaConf.create({
            'enabled': True, 'variant': 'v100', 'tree': str(root / 'tree.pt'),
            'assignment_root': str(root),
            'teacher_root': 'output/model/deyi/v100/%s/k4' % cfg.task,
            'leaves': len(payload['leaf_prototypes']),
            'num_g1': len(payload['leaf_prototypes']), 'embedding_trainable': True,
            'item_codes': str(root / 'item_codes.npy'), 'user_prefix': True,
            'recent': budget,
            'window': 'single_sequence_recent_%d' % budget,
            'protocol': 'user_G_recent_G_nativeSID_to_target_G_nativeSID',
            'approximation': summary.get('view_quality', {})})
        cfg.experiment = '%s_phi_v100_k4' % cfg.task
        cfg.run_dir = run_dir or str(Path(cfg.output_dir) / 'phi' / cfg.task
                                     / 'v100_k4' / ('seed%d' % cfg.seed))
    if deyi_v100:
        manifest = Path('output/model/deyi/v100') / cfg.task / 'k4/train/manifest.json'
        if not manifest.exists():
            raise ValueError('Complete the v100 teacher encoding before the continuous arm')
        budget = int(json.loads(manifest.read_text())['recent_budget'])
        override = os.environ.get('TIGER_RECENT_HISTORY')
        if override is not None and int(override) != budget:
            raise ValueError('v100 recent window must match the frozen teacher export')
        cfg.mode = 'recent'
        cfg.recent_history[cfg.task] = budget
        cfg.deyi = OmegaConf.create({'enabled': True, 'variant': 'v100',
            'root': 'output/model/deyi/v100', 'state_dim': 1024, 'num_interests': 4})
        cfg.experiment = '%s_deyi_v100_k4' % cfg.task
        cfg.run_dir = run_dir or str(Path(cfg.output_dir) / 'deyi' / cfg.task
                                     / 'v100_k4' / ('seed%d' % cfg.seed))
    if phi_v10_pg:
        if cfg.task != 'ad':
            raise ValueError('The first unified PG implementation is Tiger/ad')
        import torch
        root = Path(phi_pg_root) if phi_pg_root else Path('output/model/phi/v10_pg') / cfg.task / 'k4'
        label_version = root.parent.parent.name + ('_itemonly' if phi_no_user_prefix else '')
        if not (root/'complete.json').exists():
            raise ValueError('Complete unified tree and item labels before SFT')
        payload = torch.load(root/'tree.pt', map_location='cpu', weights_only=False)
        if payload.get('root_count') != 1:
            raise ValueError('PG requires one shared tree')
        cfg.mode = 'recent'
        cfg.phi = OmegaConf.create({'enabled': True, 'variant': 'v10_pg', 'tree': str(root/'tree.pt'),
            'assignment_root': str(root), 'teacher_root': f'output/model/deyi/v9/{cfg.task}/k4',
            'leaves': len(payload['leaf_prototypes']), 'num_g1': len(payload['leaf_prototypes']),
            'embedding_trainable': True, 'item_codes': str(root/'item_codes.npy'),
            'user_prefix': not phi_no_user_prefix,
            'protocol': ('recent_G_nativeSID_to_target_G_nativeSID' if phi_no_user_prefix else
                         'user_G_recent_G_nativeSID_to_target_G_nativeSID'),
            'approximation': json.loads((root/'complete.json').read_text()).get('view_quality', {})})
        assignment = json.loads((root/'complete.json').read_text())
        if payload.get('hierarchical',False):
            cfg.phi.hierarchical = True
            cfg.phi.num_g1 = int(payload['coarse_count'])
            cfg.phi.num_g2 = int(payload['fine_count'])
            cfg.phi.protocol = 'user_G1_G2_recent_G1_G2_SID_to_target_G1_G2_SID'
        if assignment.get('item_assignment') == 'user_candidates':
            cfg.phi.user_g_only = True
            cfg.phi.empty_g_id = int(assignment['empty_g_id'])
            cfg.phi.item_assignment = 'user_candidates'
        cfg.experiment = f'{cfg.task}_phi_{label_version}_k4'
        cfg.run_dir = run_dir or str(Path(cfg.output_dir)/'phi'/cfg.task/f'{label_version}_k4'/f'seed{cfg.seed}')
    return cfg


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--config", default=DEFAULT_CONFIG)
    result.add_argument("--task", choices=("short_video", "ad", "product", "product1ch"))
    result.add_argument("--mode", choices=("full", "recent"))
    result.add_argument("--run-dir")
    result.add_argument("--deyi-k", type=int, help="Use frozen DeYi states with K interest slots")
    result.add_argument("--deyi-root",
                        help="Frozen DeYi teacher root (default output/model/deyi; an HPO arm such "
                             "as output/model/deyi/hpo/sharp keeps its own run directory)")
    result.add_argument("--phi-deyi-k", type=int, default=None,
                        help="Interest-slot count of the teacher whose tree/assignment this phi "
                             "run consumes (default: --deyi-k, else 4). Selects the artifacts only; "
                             "it does not inject DeYi states into the prompt")
    result.add_argument("--phi-deyi-arm", default=None,
                        help="HPO arm of that teacher (default: the basename of a non-default "
                             "--deyi-root, else the production teacher)")
    result.add_argument("--phi", action="store_true",
                        help="Condition on the frozen predictive vocabulary (route + SID chain)")
    result.add_argument("--deyi-v7", action="store_true")
    result.add_argument("--phi-v7", action="store_true")
    result.add_argument("--deyi-v8", action="store_true")
    result.add_argument("--phi-v8", action="store_true")
    result.add_argument("--deyi-v9", action="store_true")
    result.add_argument("--phi-v9", action="store_true")
    result.add_argument("--phi-v9-fast", action="store_true")
    result.add_argument("--phi-v10-pg", action="store_true")
    result.add_argument("--phi-no-user-prefix", action="store_true")
    result.add_argument("--phi-pg-root", help="Completed PG tree and item assignment directory")
    result.add_argument("--phi-v100", action="store_true",
                        help="Condition on the frozen Phi v100 region token")
    result.add_argument("--deyi-v100", action="store_true",
                        help="Condition on the frozen DeYi v100 continuous states")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--steps", type=int, help="Stop after a formal training prefix, without changing its budget")
    result.add_argument("--nproc", type=int,
                        help="DDP ranks; must divide global_batch_size (defaults to TIGER_NPROC or 4)")
    return result


def saved_config(args):
    cfg = load_config(args.config, args.task, args.mode, args.run_dir, args.deyi_k,
                      args.nproc, args.phi,
                      deyi_root=getattr(args, "deyi_root", None),
                      phi_deyi_k=getattr(args, "phi_deyi_k", None),
                      phi_deyi_arm=getattr(args, "phi_deyi_arm", None),
                      phi_v7=getattr(args, "phi_v7", False),
                      deyi_v7=getattr(args, "deyi_v7", False),
                      phi_v8=getattr(args, "phi_v8", False), deyi_v8=getattr(args, "deyi_v8", False),
                      phi_v9=getattr(args, "phi_v9", False), deyi_v9=getattr(args, "deyi_v9", False),
                      phi_v9_fast=getattr(args, "phi_v9_fast", False),
                      phi_v10_pg=getattr(args, "phi_v10_pg", False),
                      phi_pg_root=getattr(args, "phi_pg_root", None),
                      phi_no_user_prefix=getattr(args, "phi_no_user_prefix", False),
                      phi_v100=getattr(args, "phi_v100", False),
                      deyi_v100=getattr(args, "deyi_v100", False))
    saved = OmegaConf.load(Path(cfg.run_dir) / "config.yaml")
    saved.runtime = runtime_config(cfg.runtime.nproc, saved.train.global_batch_size)
    saved.run_dir = cfg.run_dir
    return saved
