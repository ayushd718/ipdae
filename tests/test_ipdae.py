"""Tests for the standalone i_pDAE implementation.

Checks the d0 formula against BindCraft2's definition, the end-to-end score
against an independent inline computation, and PDB/JSON IO.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ipdae  # noqa: E402


def test_d0_matches_bindcraft2_formula():
    # BindCraft2: max(1.24*(max(n,19)-15)^(1/3) - 1.8, 1.0)
    for n in [0, 1, 19, 20, 50, 200]:
        expect = max(1.24 * (max(n, 19) - 15) ** (1 / 3) - 1.8, 1.0)
        assert np.isclose(ipdae.d0(n), expect), n
    # the 19-floor means n<=19 all share one d0
    assert np.isclose(ipdae.d0(0), ipdae.d0(19))


def test_no_contacts_returns_nan():
    chains = ["A", "A", "B", "B"]
    coords = np.array([[0, 0, 0], [0, 0, 3], [100, 0, 0], [100, 0, 3]], float)
    pae = np.full((4, 4), 5.0)
    r = ipdae.compute_ipdae(pae, chains, coords, binder="B")
    assert r["n_contacts"] == 0
    assert np.isnan(r["i_pdae"])


def test_matches_independent_computation():
    rng = np.random.default_rng(0)
    # 3 target (A) + 2 binder (B) residues, some in contact
    coords = np.array([
        [0, 0, 0], [3, 0, 0], [20, 0, 0],      # A
        [1, 1, 0], [2, 1, 0],                  # B
    ], float)
    chains = ["A", "A", "A", "B", "B"]
    pae = rng.uniform(0, 20, size=(5, 5))
    pae = (pae + pae.T) / 2
    r = ipdae.compute_ipdae(pae, chains, coords, binder="B", cutoff=8.0)

    # independent inline re-derivation of the BindCraft2 formula
    bi, ti = [3, 4], [0, 1, 2]
    d = np.linalg.norm(coords[bi][:, None] - coords[ti][None, :], axis=-1)
    contact = d <= 8.0
    pae_bt = pae[np.ix_(bi, ti)]

    def anchored(p, c):
        pc = c.sum(1).astype(float)
        d0 = np.maximum(1.24 * np.cbrt(np.maximum(pc, 19.0) - 15.0) - 1.8, 1.0)
        tm = 1.0 / (1.0 + (p / d0[:, None]) ** 2)
        return np.where(c, tm, 0.0).sum(1) / np.maximum(pc, 1.0)

    expect = max(anchored(pae_bt, contact).max(), anchored(pae_bt.T, contact.T).max())
    assert np.isclose(r["i_pdae"], expect)


def test_pdb_and_json_io(tmp_path):
    pdb = tmp_path / "m.pdb"
    lines = []
    coords = [("A", 0, 0, 0), ("A", 3, 0, 0), ("B", 1, 1, 0)]
    for i, (ch, x, y, z) in enumerate(coords, 1):
        lines.append(
            f"ATOM  {i:>5}  CA  ALA {ch}{i:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    pdb.write_text("\n".join(lines) + "\n")
    chains, xyz = ipdae.read_coords(str(pdb))
    assert chains == ["A", "A", "B"]
    assert xyz.shape == (3, 3)

    pae = [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [2.0, 2.0, 1.0]]
    pj = tmp_path / "pae.json"
    pj.write_text(json.dumps({"pae": pae}))
    mat = ipdae.read_pae(str(pj))
    assert mat.shape == (3, 3)

    r = ipdae.compute_ipdae(mat, chains, xyz, binder="B")
    assert r["n_contacts"] >= 1
    assert 0.0 <= r["i_pdae"] <= 1.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
