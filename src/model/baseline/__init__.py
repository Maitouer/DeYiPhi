"""Controlled long-history compression baselines (docs/v1_algorithm/04_baseline.md).

Each module here instantiates one compression *principle* under the shared TIGER + Product
setting. They all speak the same artifact language as the DeYi encoder, so the downstream
chain (TIGER sft / eval) reads them without a single change.
"""
