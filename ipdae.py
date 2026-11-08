#!/usr/bin/env python3
"""ipdae - a standalone implementation of i_pDAE (interface pDAE).

i_pDAE is a distance-masked interface confidence score introduced in BindCraft2
(https://github.com/PacesaLab/BindCraft2, bindcraft/filters.py) as its default
ranking metric. Over CA-CA contacts (<= 8 A) between a binder and its target, it
computes a per-anchor TM-style score of the predicted aligned error,

    tm_ij = 1 / (1 + (PAE_ij / d0(n_i))^2),
    d0(n) = max(1.24 * (max(n, 19) - 15)^(1/3) - 1.8, 1.0),

averages it over each anchor residue's contacts, and takes the maximum over
anchors and over the two interface directions.

BindCraft2's implementation is JAX coupled to its internal Protein objects. This
module is a faithful, dependency-light reimplementation (numpy only) that runs
directly from a coordinate file (PDB/mmCIF) and a PAE file (JSON/NPZ), for reuse
outside the BindCraft2 pipeline. The numerical definition is unchanged.

This is a derivative work of BindCraft2; see LICENSE and the credit in README.

Usage:
    ipdae model.pdb model_scores.json
    ipdae model.cif pae.npz --binder B
    python -m ipdae model.pdb pae.json --reduce mean --per-residue
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys

import numpy as np

__version__ = "0.1.0"

CA_CONTACT_CUTOFF = 8.0  # BindCraft2 i_pDAE default (Angstrom)


# --------------------------------------------------------------------------- IO
def _open(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_coords(path):
    """Return (chains, coords) for CA atoms in file order.

    chains: list[str] of chain ids; coords: (N, 3) float array. Order matches the
    residue order of the PAE matrix for AF2/AF3/Boltz/ESMFold outputs.
    """
    p = str(path).lower()
    if p.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz")):
        return _read_cif_ca(path)
    return _read_pdb_ca(path)


def _read_pdb_ca(path):
    chains, xyz = [], []
    with _open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                alt = line[16]
                if alt not in (" ", "A"):
                    continue
                chains.append(line[21])
                xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return chains, np.asarray(xyz, dtype=np.float64)


def _read_cif_ca(path):
    lines = _open(path).read().splitlines()
    # locate the _atom_site loop header to learn column order
    cols = {}
    header_order = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("_atom_site."):
            cols[s.split(".", 1)[1]] = len(header_order)
            header_order.append(s)
    need = ["label_atom_id", "label_comp_id", "label_asym_id",
            "Cartn_x", "Cartn_y", "Cartn_z"]
    # auth_asym_id is preferred for chain id when present
    chain_key = "auth_asym_id" if "auth_asym_id" in cols else "label_asym_id"
    alt_key = cols.get("label_alt_id")
    for k in need:
        if k not in cols:
            raise ValueError(f"{path}: mmCIF _atom_site missing column {k}")
    chains, xyz = [], []
    for ln in lines:
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        f = ln.split()
        if len(f) <= max(cols.values()):
            continue
        if f[cols["label_atom_id"]].strip('"') != "CA":
            continue
        if alt_key is not None:
            a = f[alt_key]
            if a not in (".", "?", "A"):
                continue
        chains.append(f[cols[chain_key]])
        xyz.append((float(f[cols["Cartn_x"]]), float(f[cols["Cartn_y"]]),
                    float(f[cols["Cartn_z"]])))
    return chains, np.asarray(xyz, dtype=np.float64)


_PAE_KEYS = ("pae", "predicted_aligned_error", "token_pair_pae")


def read_pae(path):
    """Load an N x N PAE matrix from a JSON (AF2/AF3/Boltz/ESMFold) or NPZ file."""
    p = str(path).lower()
    if p.endswith(".npz"):
        with np.load(path) as z:
            key = next((k for k in _PAE_KEYS if k in z), None)
            if key is None:
                raise ValueError(f"{path}: no PAE array; keys={list(z.files)}")
            pae = np.asarray(z[key], dtype=np.float64)
    else:
        obj = json.load(_open(path))
        if isinstance(obj, list):  # some AF2 dumps are a bare [ {...} ]
            obj = obj[0]
        key = next((k for k in _PAE_KEYS if k in obj), None)
        if key is None:
            raise ValueError(f"{path}: no PAE key in JSON; keys={list(obj)[:8]}")
        pae = np.asarray(obj[key], dtype=np.float64)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"{path}: PAE must be square; got {pae.shape}")
    return pae


# --------------------------------------------------------------------- metric
def d0(n):
    """TM-score distance scale used by i_pDAE (BindCraft2 tm_score_distance_scale):
    partner count floored at 19, result floored at 1.0."""
    n = np.asarray(n, dtype=np.float64)
    return np.maximum(1.24 * np.cbrt(np.maximum(n, 19.0) - 15.0) - 1.8, 1.0)


def _anchored(pae_block, contact_block):
    """Per-anchor (row) distance-masked TM score; BindCraft2 anchored_interface_tm_scores."""
    pc = contact_block.sum(axis=1).astype(np.float64)
    tm = 1.0 / (1.0 + (pae_block / d0(pc)[:, None]) ** 2)
    return np.where(contact_block, tm, 0.0).sum(axis=1) / np.maximum(pc, 1.0)


def compute_ipdae(pae, chains, coords, binder, target=None,
                  cutoff=CA_CONTACT_CUTOFF, reduce="max"):
    """Compute i_pDAE for the binder<->target interface.

    pae      : (N, N) PAE matrix, residue order matching `chains`/`coords`.
    chains   : length-N sequence of chain ids (one per CA).
    coords   : (N, 3) CA coordinates.
    binder   : chain id (or iterable of ids) treated as the binder.
    target   : chain id(s) for the target; default = every chain not in `binder`.
    cutoff   : CA-CA contact cutoff in Angstrom (default 8).
    reduce   : "max" (BindCraft2 default) or "mean" over anchor residues.

    Returns a dict: i_pdae, binder_anchored, target_anchored, n_contacts.
    """
    chains = np.asarray([str(c) for c in chains])
    coords = np.asarray(coords, dtype=np.float64)
    n = len(chains)
    if pae.shape[0] != n or coords.shape[0] != n:
        raise ValueError(f"pae {pae.shape}, {n} chains, {coords.shape[0]} coords disagree")
    bset = {binder} if isinstance(binder, str) else set(map(str, binder))
    b = np.isin(chains, list(bset))
    if target is None:
        t = ~b
    else:
        tset = {target} if isinstance(target, str) else set(map(str, target))
        t = np.isin(chains, list(tset))
    if not b.any():
        raise ValueError(f"no binder residues for chain(s) {sorted(bset)}")
    if not t.any():
        raise ValueError("no target residues")

    bi, ti = np.where(b)[0], np.where(t)[0]
    dmat = np.linalg.norm(coords[bi][:, None, :] - coords[ti][None, :, :], axis=-1)
    contact = dmat <= cutoff
    out = {"n_contacts": int(contact.sum()),
           "binder_anchored": float("nan"), "target_anchored": float("nan"),
           "i_pdae": float("nan")}
    if not contact.any():
        return out
    pae_bt = pae[np.ix_(bi, ti)]
    agg = np.max if reduce == "max" else np.mean
    out["binder_anchored"] = float(agg(_anchored(pae_bt, contact)))
    out["target_anchored"] = float(agg(_anchored(pae_bt.T, contact.T)))
    out["i_pdae"] = max(out["binder_anchored"], out["target_anchored"])
    return out


# ------------------------------------------------------------------------- CLI
def _guess_binder(chains):
    """Default binder = the shortest chain (the usual binder-design convention)."""
    order, sizes = [], {}
    for c in chains:
        if c not in sizes:
            order.append(c)
        sizes[c] = sizes.get(c, 0) + 1
    return min(order, key=lambda c: sizes[c])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ipdae", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("coords", help="coordinate file (.pdb/.cif, optionally .gz)")
    ap.add_argument("pae", help="PAE file (.json/.npz)")
    ap.add_argument("--binder", help="binder chain id (default: shortest chain)")
    ap.add_argument("--target", help="target chain id(s), comma-separated (default: all others)")
    ap.add_argument("--cutoff", type=float, default=CA_CONTACT_CUTOFF,
                    help="CA-CA contact cutoff in Angstrom (default 8)")
    ap.add_argument("--reduce", choices=["max", "mean"], default="max",
                    help="aggregation over anchor residues (BindCraft2 main uses max)")
    ap.add_argument("--version", action="version", version=f"ipdae {__version__}")
    args = ap.parse_args(argv)

    chains, coords = read_coords(args.coords)
    pae = read_pae(args.pae)
    binder = args.binder or _guess_binder(chains)
    target = args.target.split(",") if args.target else None
    r = compute_ipdae(pae, chains, coords, binder, target,
                      cutoff=args.cutoff, reduce=args.reduce)
    print(f"binder={binder} target={target or 'rest'} "
          f"contacts={r['n_contacts']} reduce={args.reduce}")
    print(f"i_pDAE                 {r['i_pdae']:.6f}")
    print(f"  binder-anchored      {r['binder_anchored']:.6f}")
    print(f"  target-anchored      {r['target_anchored']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
