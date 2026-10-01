"""
Sum every neighbour shell that carries an exchange constant. Returns the real and imaginary parts of J_ab(q) as a two-element array.

Inelastic neutron scattering on bulk CrSBr resolves exchange out to the eighth
neighbour. Seven of those eight shells are in-plane and survive in the monolayer; the sixth
is interlayer and does not. In the sign convention H = -(1/2) sum_abij S^T_ai J_abij S_bj,
where a positive isotropic J is ferromagnetic, they are

    J1 = 1.90,  J2 = 3.38,  J3 = 1.67,  J4 = 0.09,  J5 = 0.09,
    J7 = -0.37, J8 = 0.29        (meV, by bulk shell index)

All other shells vanish. The exchange is isotropic on every shell, so the tensor is
J'_ab(q) = J_ab(q) * I_3 with

    J_ab(q) = sum_shells J_shell * gamma_shell,ab(q).

Return [Re, Im] of J_ab(q).

Returns
-------
numpy.ndarray: shape (2,) array holding the real and imaginary parts.
"""

import functools
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exchange_at_q(a: int, b: int, qx: float, qy: float) -> "np.ndarray":
    """Sum every neighbour shell that carries an exchange constant. Returns the real and imaginary parts of J_ab(q) as a two-element array.

    Args:
        a: sublattice index of the origin site, 0 or 1.
        b: sublattice index of the neighbour site, 0 or 1.
        qx: Cartesian x component of the wavevector.
        qy: Cartesian y component of the wavevector.

    Returns:
        numpy.ndarray: shape (2,) array holding the real and imaginary parts.
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


def _oracle_exchange_at_q(a: int, b: int, qx: float, qy: float) -> "np.ndarray":
    """Sum every neighbour shell that carries an exchange constant."""
    g = _J_scalar(a, b, np.array([qx, qy]))
    return np.array([g.real, g.imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: the intra-sublattice element at the zone centre
            "setup": "",
            "call": 'exchange_at_q(0, 0, 0.0, 0.0)',
            "gold_call": '_oracle_exchange_at_q(0, 0, 0.0, 0.0)',
        },
        {
            # normal: the inter-sublattice element at the zone centre
            "setup": "",
            "call": 'exchange_at_q(0, 1, 0.0, 0.0)',
            "gold_call": '_oracle_exchange_at_q(0, 1, 0.0, 0.0)',
        },
        {
            # normal: the inter-sublattice element away from high symmetry
            "setup": "",
            "call": 'exchange_at_q(0, 1, 0.9, -0.4)',
            "gold_call": '_oracle_exchange_at_q(0, 1, 0.9, -0.4)',
        },
        {
            # edge: the intra-sublattice element near the zone corner
            "setup": "",
            "call": 'exchange_at_q(1, 1, 0.8975979010256552, 0.6600444097822498)',
            "gold_call": '_oracle_exchange_at_q(1, 1, 0.8975979010256552, 0.6600444097822498)',
        },
    ]
