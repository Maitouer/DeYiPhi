"""Phi: a shared predictive vocabulary over frozen DeYi memories, plus target routes.

Phi reads the frozen DeYi chain and answers two questions the DeYi paper leaves open:

* **what can be shared?** Different users' interest slots whose *predictions* agree should
  share one discrete token (predictive-function quantization, not embedding quantization);
* **why is this item relevant now?** Every supervised target is attributed to one shared
  token, so the downstream generator predicts a predictive route before the native SID.

The two design documents that define the chain are
``docs/v1_algorithm/02_phi.md`` (v200: the predictive-KL codebook, the route attribution and
the route-prefixed generation interface) and ``docs/v1_algorithm/03_phi.md`` (v300: the
user-balanced reservoir, the head-tail estimator, the medoid codebook and the distilled
router that make the same geometry affordable on a catalog the size of this project's).

Layout of the module::

    config.py      the effective config: the DeYi config is the base, phi adds its sections
    artifacts.py   where every stage writes, formats, atomic writes and the reuse guards
    runtime.py     device choice, seeding, logging and progress
    items.py       the frozen item space e(i) and the task catalog with its proposal q(i)
    states.py      the frozen DeYi arm: states, mask, readout alpha, split rows
    reservoir.py   the user-balanced codebook reservoir
    estimator.py   ANN head + importance-sampled tail -> (A_hat, mu_hat), plus calibration
    codebook.py    predictive-KL medoid codebook and the exact token normalizers
    router.py      the distilled fast router and its validation
    tokenize.py    full-state tokenization (no catalog access)
    route.py       target route labels g*(r, y)
    audit.py       the diagnostics the documents require before a result counts
    pipeline.py    stage orchestration
    cli.py         ``python -m src.model.phi.cli <stage>``
"""

