"""
Assemble the in-plane bond vectors of one neighbour shell of monolayer CrSBr. Returns a real array of bond vectors.

The Cr atoms of CrSBr form a rectangular lattice, two per cell, buckled out of
the plane. Bulk lattice constants are a = 3.50, b = 4.76 and c = 7.96 Angstrom. In-plane
primitive vectors are (a, 0, 0) and (0, b, 0), the layer stacking vector is (0, 0, c), and
the two Cr sites sit at (0, 0, +d/2) and (a/2, b/2, -d/2) with a buckling d = 1.96 Angstrom.

Neighbour shells are indexed by increasing Cr-Cr bond length over the BULK crystal, so the
ranking includes bonds to Cr atoms in the layers above and below. Shells 6, 9, 10 and 12 are
interlayer; a monolayer therefore carries no shell-6 bonds at all, while shells 7 and 8 are
present. Shell indices refer to the bulk ranking throughout.

A bond running out of site a in the origin cell towards site b in cell i has the vector
r_ij + basis_b - basis_a. Return the in-plane bonds of the requested shell as full
three-dimensional vectors, ordered by the polar angle atan2(y, x) of their in-plane part,
folded into [0, 2*pi).

Returns
-------
numpy.ndarray: shape (N, 3) array of bond vectors, angle-sorted.
"""

import functools
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def neighbour_bond_vectors(shell: int, a: int, b: int) -> "np.ndarray":
    """Assemble the in-plane bond vectors of one neighbour shell of monolayer CrSBr. Returns a real array of bond vectors.

    Args:
        shell: bulk neighbour shell index, counted from 1.
        a: sublattice index of the origin site, 0 or 1.
        b: sublattice index of the neighbour site, 0 or 1.

    Returns:
        numpy.ndarray: shape (N, 3) array of bond vectors, angle-sorted.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import functools

import numpy as np


def _lattice():
    """Primitive vectors and the buckled two-site Cr basis of bulk CrSBr."""
    a, b, c, buckling = 3.50, 4.76, 7.96, 1.96
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([0.0, b, 0.0])
    a3 = np.array([0.0, 0.0, c])
    basis = [np.array([0.0, 0.0, 0.5 * buckling]),
             np.array([0.5 * a, 0.5 * b, -0.5 * buckling])]
    return a1, a2, a3, basis


@functools.lru_cache(maxsize=None)
def _bulk_bonds(nmax=3):
    """Every bulk Cr-Cr bond, ranked by length over the 3D structure (interlayer
    included). Depends on nmax alone, so the cache cannot make the ranking call-order
    dependent."""
    a1, a2, a3, basis = _lattice()
    raw = []
    for a in range(2):
        for b in range(2):
            for n1 in range(-nmax, nmax + 1):
                for n2 in range(-nmax, nmax + 1):
                    for n3 in (-1, 0, 1):
                        r_ij = n1 * a1 + n2 * a2 + n3 * a3
                        r = r_ij + basis[b] - basis[a]
                        d = float(np.linalg.norm(r))
                        if d > 1e-9:
                            raw.append((d, a, b, n3, r_ij, r))
    raw.sort(key=lambda t: t[0])
    radii = []
    for t in raw:
        if not radii or t[0] - radii[-1] > 1e-3:
            radii.append(t[0])
    out = []
    for (d, a, b, n3, r_ij, r) in raw:
        shell = next(k + 1 for k, rad in enumerate(radii) if abs(d - rad) < 1e-3)
        out.append((shell, a, b, n3, r_ij, r, d))
    return out


def _shell_bonds(shell, a, b):
    """In-plane bonds of one bulk shell from sublattice a to b, angle-sorted."""
    sel = [t for t in _bulk_bonds()
           if t[0] == shell and t[1] == a and t[2] == b and t[3] == 0]
    sel.sort(key=lambda t: np.arctan2(t[5][1], t[5][0]) % (2.0 * np.pi))
    return sel


def _oracle_neighbour_bond_vectors(shell: int, a: int, b: int) -> "np.ndarray":
    """Assemble the in-plane bond vectors of one neighbour shell of monolayer CrSBr."""
    bonds = [t[5] for t in _shell_bonds(shell, a, b)]
    return np.array(bonds, dtype=float).reshape(-1, 3)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: shell 1, the intra-sublattice bond along a
            "setup": "",
            "call": 'neighbour_bond_vectors(1, 0, 0)',
            "gold_call": '_oracle_neighbour_bond_vectors(1, 0, 0)',
        },
        {
            # normal: shell 2, the buckled inter-sublattice bond
            "setup": "",
            "call": 'neighbour_bond_vectors(2, 0, 1)',
            "gold_call": '_oracle_neighbour_bond_vectors(2, 0, 1)',
        },
        {
            # boundary: shell 7, which follows the interlayer shell 6
            "setup": "",
            "call": 'neighbour_bond_vectors(7, 0, 0)',
            "gold_call": '_oracle_neighbour_bond_vectors(7, 0, 0)',
        },
        {
            # edge: shell 8, the outermost shell that carries an exchange
            "setup": "",
            "call": 'neighbour_bond_vectors(8, 1, 0)',
            "gold_call": '_oracle_neighbour_bond_vectors(8, 1, 0)',
        },
    ]
