"""
Compute the on-site molecular field of the RPA Hamiltonian. Returns a real scalar in meV.

The exchange part of the RPA Hamiltonian carries a diagonal term

    C^RPA_ab = delta_ab * sum_c <S''z_c> v_a^T J'_ac(0) v_c,

where v_a is the local-frame vector along the ordered moment and <S''z_c> is the sublattice
magnetisation. Both CrSBr sublattices are equivalent and carry the same moment.

Since v^T v = 1 and the exchange is isotropic, this is the sum of the exchange matrix
elements at q = 0 weighted by the magnetisation.

Returns
-------
float: the molecular field in meV.
"""

import functools
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def molecular_field(sz: float) -> float:
    """Compute the on-site molecular field of the RPA Hamiltonian. Returns a real scalar in meV.

    Args:
        sz: sublattice magnetisation, shared by both sites.

    Returns:
        float: the molecular field in meV.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import functools

import numpy as np


def _exchange_constants():
    """Isotropic exchange by BULK shell index, meV; the interlayer sixth is absent."""
    return {1: 1.90, 2: 3.38, 3: 1.67, 4: 0.09, 5: 0.09, 7: -0.37, 8: 0.29}


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


def _gamma(shell, a, b, q):
    """Structure factor of one shell, sum_i exp(i q . r_i) over its bonds."""
    return sum(np.exp(1j * float(np.dot(q, t[4][:2])))
               for t in _shell_bonds(shell, a, b))


def _J_scalar(a, b, q):
    """J_ab(q): isotropic exchange summed over every shell that carries one."""
    js = _exchange_constants()
    return sum(js[shell] * _gamma(shell, a, b, q) for shell in sorted(js))


def _J_matrix(q):
    """The 2x2 matrix of J_ab(q)."""
    return np.array([[_J_scalar(a, b, q) for b in range(2)] for a in range(2)])


def _mol_field(sz):
    """C^RPA_aa = sum_c <Sz_c> v^T J'_ac(0) v, the on-site molecular field."""
    J0 = _J_matrix(np.zeros(2))
    return float(np.real(sz * J0[0].sum()))


def _oracle_molecular_field(sz: float) -> float:
    """Compute the on-site molecular field of the RPA Hamiltonian."""
    return _mol_field(sz)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: the saturated moment
            "setup": "",
            "call": 'molecular_field(1.5)',
            "gold_call": '_oracle_molecular_field(1.5)',
        },
        {
            # normal: a partly demagnetised state
            "setup": "",
            "call": 'molecular_field(1.0)',
            "gold_call": '_oracle_molecular_field(1.0)',
        },
        {
            # boundary: zero magnetisation, where the field vanishes
            "setup": "",
            "call": 'molecular_field(0.0)',
            "gold_call": '_oracle_molecular_field(0.0)',
        },
        {
            # edge: the moment reached just below the transition
            "setup": "",
            "call": 'molecular_field(0.1388530302)',
            "gold_call": '_oracle_molecular_field(0.1388530302)',
        },
    ]
