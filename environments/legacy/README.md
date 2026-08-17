# Historical environment locks

Files in this directory preserve exact, previously audited environments for
provenance. They are not the default dependency contract for current source.

`cobot_magic_v1_rtx5090_20260815.txt` is byte-for-byte identical to the lock
formerly stored on PrometheusV4 `hardware/cobot-daimon` at revision
`b5e15520db97e070cf3becc675ab06b36c75dc02`; its SHA-256 is
`d0a744c2cc662bad6666551ce016c28e2269f7a89aa67ebc01536cc09f43f34c`.
It records Python 3.10-era Torch 2.11.0/CUDA 12.8 and Zarr 2.18.3 state.

Current `lv-robotics-flow-matching[train]` requires Zarr 3, so this historical
lock must not be synced into a new environment or described as compatible with
the current package without a separately reviewed legacy-resume validation.
Keep it to reproduce provenance and compare an existing run, not as a solve
input for a fresh run.
