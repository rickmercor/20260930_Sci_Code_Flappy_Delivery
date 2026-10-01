"""
Evaluate the RPA+CD Curie temperature of monolayer CrSBr. Returns the critical temperature in K.

At the transition the sublattice magnetisation vanishes. Every magnon energy is
proportional to <S''z> there, so the Bose factor reduces to n_B -> kB T / omega, the
occupation diverges as 1/<S''z>, and Callen's expression reduces to its large-occupation
limit <S''z> -> S(S+1)/(3 Phi). Eliminating <S''z> between the two leaves the Callen
argument pinned at a pure number,

    t* = <S''z>(1 + 2 Phi)/(2 S^2) -> (S + 1)/(3 S),

and the critical temperature follows from a single Brillouin-zone average,

    kB T_C = S(S + 1) / (3 Q),
    Q = (1/N) sum_q sum_i |c_{0,i,q}|^2 M_{i,q} / (M_{i,q}^2 - beta^2).

Here M_{i,q} are the eigenvalues, and c_{0,i,q} the sublattice-0 components of the
eigenvectors, of the two-by-two Hermitian matrix carrying the exchange and the diagonal part
of the anisotropy at unit magnetisation,

    M_ab(q) = -J_ab(q) + delta_ab [ C(1) + (-A(1) + C_K(1)) Upsilon* ],

with A(1), B(1) and C_K(1) the anisotropy contractions at unit magnetisation, C(1) the
molecular field at unit magnetisation, Upsilon* the Callen factor evaluated at t*, and
beta = |B(1)| Upsilon* the anomalous element.

Sample the zone on the same Gamma-centred n by n grid used for the occupation.

Returns
-------
float: the Curie temperature in K.
"""

import functools
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def curie_temperature(n: int) -> float:
    """Evaluate the RPA+CD Curie temperature of monolayer CrSBr. Returns the critical temperature in K.

    Args:
        n: linear size of the Gamma-centred Brillouin-zone mesh.

    Returns:
        float: the Curie temperature in K.

    Raises:
        ValueError: if the neighbour shells of the monolayer do not match the
            bulk ranking, if the exchange sampled on the mesh disagrees with the
            bond-by-bond sum, if the magnon spectrum at unit magnetisation is not
            positive definite, or if the self-consistent state below the
            transition carries no ordered moment.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import functools

import numpy as np


def _spin():
    """The Cr3+ spin of CrSBr."""
    return 1.5


def _boltzmann():
    """Boltzmann's constant in meV/K."""
    return 0.08617333262


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


def _mesh_q(n):
    """Gamma-centred n x n sampling of the first Brillouin zone."""
    a1, a2, a3, basis = _lattice()
    b1 = np.array([2.0 * np.pi / a1[0], 0.0])
    b2 = np.array([0.0, 2.0 * np.pi / a2[1]])
    return np.array([i / n * b1 + j / n * b2 for i in range(n) for j in range(n)])


@functools.lru_cache(maxsize=None)
def _mesh_J(n):
    """J_ab(q) on the whole mesh; depends on the mesh size alone, so built once."""
    js = _exchange_constants()
    qs = _mesh_q(n)
    J = np.zeros((len(qs), 2, 2), dtype=complex)
    for (shell, a, b, n3, r_ij, r, d) in _bulk_bonds():
        if n3 != 0 or shell not in js:
            continue
        J[:, a, b] += js[shell] * np.exp(1j * (qs @ r_ij[:2]))
    return J


def _oracle_curie_temperature(n: int) -> float:
    """Evaluate the RPA+CD Curie temperature of monolayer CrSBr."""
    # the sixth bulk shell is interlayer, so the monolayer carries none of it
    if _oracle_neighbour_bond_vectors(6, 0, 0).shape != (0, 3):
        raise ValueError("the sixth bulk shell must contribute no in-plane bond")
    if abs(_oracle_exchange_structure_factor(2, 0, 1, 0.0, 0.0)[0] - 4.0) > 1e-9:
        raise ValueError("shell 2 must carry four in-plane bonds")

    # the vectorised mesh exchange must reproduce the bond-by-bond J_ab(q)
    qs = _mesh_q(n)
    Js = _mesh_J(n)
    for idx in (0, len(qs) // 3, len(qs) - 1):
        for a in range(2):
            for b in range(2):
                ref = _oracle_exchange_at_q(a, b, float(qs[idx][0]), float(qs[idx][1]))
                if abs(complex(ref[0], ref[1]) - Js[idx, a, b]) > 1e-9:
                    raise ValueError("the mesh exchange disagrees with J_ab(q)")

    # the Callen argument is pinned at t* = (S+1)/(3S) at the transition, which is
    # Upsilon evaluated at unit magnetisation with an occupation of 3/4
    ups = _oracle_callen_factor(1.0, 0.75)

    # the block that fixes the temperature scale, at unit magnetisation
    c0 = _oracle_molecular_field(1.0)
    a1, b1, c1 = _oracle_anisotropy_contractions(1.0)
    diag = c0 + (-a1 + c1) * ups
    beta = abs(b1) * ups

    M = -Js + np.eye(2)[None, :, :] * diag
    mvals, mvecs = np.linalg.eigh(M)
    if np.any(mvals ** 2 <= beta ** 2):
        raise ValueError("the magnon spectrum is not positive definite")
    w0 = np.abs(mvecs[:, 0, :]) ** 2
    Q = float((w0 * mvals / (mvals ** 2 - beta ** 2)).sum(axis=1).mean())
    S = _spin()
    Tc = S * (S + 1.0) / (3.0 * Q) / _boltzmann()

    # the ordered state below the transition must be reduced but not destroyed
    w = _oracle_magnon_energies(0.0, 0.0, S, 0.0)
    if not w[2] > 0.0:
        raise ValueError("the zero-temperature gap must be positive")
    sz, phi = S, 0.0
    for _ in range(40):
        phi = _oracle_magnon_occupation(0.5 * Tc, sz, phi, min(n, 48))
        sz = 0.7 * sz + 0.3 * _oracle_sublattice_magnetisation(phi)
    if not 0.0 < sz < S:
        raise ValueError("the ordered state below the transition is unphysical")
    return Tc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: a mesh dense enough for the average to have settled
            "setup": "",
            "call": 'curie_temperature(96)',
            "gold_call": '_oracle_curie_temperature(96)',
        },
        {
            # normal: denser still, which must barely move it
            "setup": "",
            "call": 'curie_temperature(192)',
            "gold_call": '_oracle_curie_temperature(192)',
        },
        {
            # boundary: a coarse mesh, where the average is not yet converged
            "setup": "",
            "call": 'curie_temperature(48)',
            "gold_call": '_oracle_curie_temperature(48)',
        },
        {
            # edge: an intermediate mesh size that is not a power of two
            "setup": "",
            "call": 'curie_temperature(144)',
            "gold_call": '_oracle_curie_temperature(144)',
        },
    ]
