# ipdae — standalone i_pDAE (interface pDAE)

A small, dependency-light (numpy-only) implementation of **i_pDAE**, the
distance-masked interface confidence score introduced in
[**BindCraft2**](https://github.com/PacesaLab/BindCraft2) (its default ranking
metric). It runs directly from a coordinate file and a PAE file, so the metric
can be reused outside the BindCraft2 pipeline.

> **Credit.** i_pDAE is the work of the BindCraft2 authors (Martin Pacesa et al.,
> University of Zurich). In BindCraft2 it is defined in `bindcraft/filters.py` as
> JAX functions coupled to BindCraft2's internal structure objects. This project
> is a faithful reimplementation in plain numpy with file IO and a CLI; the
> numerical definition is unchanged. It is a derivative work distributed under
> BindCraft2's license — see [LICENSE](LICENSE). Please cite BindCraft2.

## What it computes

Over CA–CA contacts (≤ 8 Å by default) between a **binder** chain and its
**target**, for each anchor residue *i* and its contacting partners *j*:

```
tm_ij = 1 / (1 + (PAE_ij / d0(n_i))^2)
d0(n) = max(1.24 * (max(n, 19) - 15)^(1/3) - 1.8, 1.0)
```

each anchor's `tm` is averaged over its contacts; i_pDAE is the **max over
anchors**, taken as the **max over the two interface directions** (binder-anchored
and target-anchored). A `--reduce mean` option reproduces the mean-over-anchors
variant.

Unlike the ipSAE family, the interface here is selected by **3D distance**, not by
a PAE confidence cutoff — hence "distance-masked."

## Install

```bash
pip install .           # or: pip install numpy && python ipdae.py ...
```

Only runtime dependency is `numpy`. Python ≥ 3.9.

## Usage

```bash
# AlphaFold2 / ESMFold style: PDB + scores JSON (key "pae")
ipdae model.pdb model_scores.json

# AlphaFold3 / Boltz style: mmCIF + PAE (.json or .npz)
ipdae model.cif pae.npz --binder B

# pick chains / aggregation explicitly
ipdae model.pdb pae.json --binder B --target A,C --reduce max
python -m ipdae model.cif pae.json --cutoff 8
```

By default the **shortest chain** is treated as the binder (the usual
binder-design convention) and every other chain as the target. Override with
`--binder` / `--target`.

Inputs supported:
- **Coordinates:** `.pdb`, `.cif` / `.mmcif` (optionally `.gz`). CA atoms only.
- **PAE:** `.json` (keys `pae` / `predicted_aligned_error` / `token_pair_pae`;
  AF2/AF3/Boltz/ESMFold dumps) or `.npz` (same keys).

The coordinate residue order must match the PAE matrix order (true for standard
AF2/AF3/Boltz/ESMFold outputs).

## As a library

```python
from ipdae import read_coords, read_pae, compute_ipdae

chains, coords = read_coords("model.pdb")
pae = read_pae("model_scores.json")
r = compute_ipdae(pae, chains, coords, binder="B")   # target defaults to all others
print(r["i_pdae"], r["binder_anchored"], r["target_anchored"], r["n_contacts"])
```

## Validation

Values reproduce BindCraft2's definition exactly. The test suite checks the d0
formula and the end-to-end score against hand-computed fixtures; see `tests/`.

## Citing

If you use this, cite **BindCraft2** (the source of the metric). See
[`CITATION.cff`](CITATION.cff).
