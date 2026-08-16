# Prometheus integration provenance

This repository is the source of truth for LV Robotics Flow Matching policy
training and inference internals. PrometheusV4 consumes it as the pinned
submodule `third_party/flow_matching`; the Prometheus rollout adapter remains
`prometheus.policy.fm_policy.FMPolicy`.

The integration history is:

- original upstream: `https://github.com/zangyujie2004/flow_matching.git`
- upstream capability baseline: `5128ff35c19ee34cadcfc8a2953d3f3d70ed2244`
  (`Expose async memory status`)
- LV fork baseline before reintegration:
  `f9c92971b81805784ec810daa88e13ff68f7d4f3`
- PrometheusV4 source snapshot used for reintegration:
  `9a4cc6a696d271699b46345882ae62b031646a32`

The reintegration preserves the LV fork's remove-hand camera/cache behavior and
the newer Prometheus Memory, DDP, strict resume, finite-value checks, batched
latent precompute, evaluation, and deployment-inference changes. Internal
imports now use the collision-free `lv_flow_matching` package namespace.

Training data, DINO weights, latent caches, TensorBoard logs, and checkpoints
are runtime artifacts and are intentionally excluded from Git.
