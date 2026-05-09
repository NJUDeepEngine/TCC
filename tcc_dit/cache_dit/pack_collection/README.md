## Pack Collection

This release keeps the TCC pack collector used by the DiT main-table runs:

- `compute_online_tcc.py`

The collector replays the accelerated DiT trajectory, builds per-step TCC
packs under `OUT_DIR/tcc_pack`, and writes `step_XX.pt` files consumed by
`sample_ddp.py --tcc-enable --tcc-dir OUT_DIR/tcc_pack`.

Older offline pack builders and low-rank exploratory variants are intentionally
not included in the release tree.
