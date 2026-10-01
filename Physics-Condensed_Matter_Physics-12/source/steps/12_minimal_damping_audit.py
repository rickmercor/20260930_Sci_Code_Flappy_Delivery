"""
Run the benchmark configuration end to end and report the largest retained fresh-sweep fraction that still leaves the fixed point locally stable under the safety-adjusted criterion for BOTH iteration schemes at once.

The benchmark is the single-site model at inverse temperature 1, interaction 5, chemical potential 1 away from half filling, hybridisation strength 0.5, on a box of 8 fermionic and 7 bosonic frequencies. Reach the fixed point continuously connected to the non-interacting limit by damped iteration at damping p_solve from a vanishing vertex. Then take the spectrum that controls local convergence there for BOTH sweeps: for the sweep carrying the reducible vertices at that state, and for the sweep carrying the full vertices at the same physical fixed point re-expressed as F = Phi + Gamma in the same layout. Each spectrum sets a limit on p. The damped iteration has multipliers 1 - p lambda over that spectrum's eigenvalues, and local stability requires every one of them to lie strictly inside the unit circle. Obtain from that requirement the largest p a single eigenvalue admits, multiply it by the safety factor c, take the smallest such value over the spectrum, and cap the result at one. The reported fraction is the smaller of the two spectra's bounds, so that the single retained fraction leaves BOTH iterations locally stable. If either spectrum has an eigenvalue with non-positive real part, raise instead: no retained fraction stabilises that direction. Because the iteration retains a fraction p of each fresh sweep, every sufficiently small p is stable and this quantity is an upper bound on p rather than a minimum. Provided the solve converges to the stated branch the answer does not depend on p_solve, which only decides how the fixed point is reached, and it scales linearly in the safety factor c while it stays below one.

Returns
-------
float, the largest retained fresh-sweep fraction allowed by the safety-adjusted criterion
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def minimal_damping_audit(c: float, p_solve: float) -> float:
    """Run the benchmark configuration end to end and report the largest retained fresh-sweep fraction that still leaves the fixed point locally stable under the safety-adjusted criterion for BOTH iteration schemes at once.

    Parameters
    ----------
    c : float, safety factor strictly between zero and one
    p_solve : float, damping used to reach the fixed point, in (0, 1]

    Returns
    -------
    float, the largest retained fresh-sweep fraction allowed by the safety-adjusted criterion

    Raises
    ------
    ValueError
        If c lies outside (0, 1), if p_solve lies outside (0, 1], if the reducible sweep fails to reach its fixed point at the given p_solve, which happens above roughly 0.82, or if either stability spectrum has an eigenvalue with non-positive real part.

    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def _oracle_minimal_damping_audit(c: float, p_solve: float) -> float:
    if not np.isfinite(c) or not (0.0 < c < 1.0):
        raise ValueError("the safety factor must lie strictly between zero and one")
    if not (0.0 < p_solve <= 1.0):
        raise ValueError("the damping used to reach the fixed point must lie in (0, 1]")
    beta, U, dmu, hyb, Nf, Nb = _bench()

    grids = _oracle_matsubara_grids(beta, Nf, Nb)
    if grids.size != Nf + Nb:
        raise ValueError("the grid step must return the fermionic then the bosonic frequencies")
    if abs(grids[:Nf][Nf//2] - np.pi/beta) > 1e-12:
        raise ValueError("the fermionic grid is not centred on +- pi/beta")

    G0 = _oracle_bare_propagator(beta, Nf, U, dmu, hyb)
    Lam = _oracle_bare_lambda(U)
    if abs(Lam[0] + Lam[1]) > 1e-12:
        raise ValueError("the density and magnetic bare vertices must cancel")

    zero = np.zeros((4, Nf, Nf, Nb), dtype=complex)
    if np.abs(_oracle_irreducible_vertices(zero, Lam) - Lam[:, None, None, None]).max() > 1e-12:
        raise ValueError("with no reducible vertex, Gamma must reduce to Lambda")
    bub = _oracle_pair_bubbles(G0, Nb)
    if bub.shape != (2, Nf, Nb):
        raise ValueError("the bubbles must carry the ph channel first and the pp channel second")

    state = _oracle_relax_parquet(_pack(zero, G0), Lam, beta, U, dmu, hyb, Nf, Nb, p_solve)
    if np.max(np.abs(_oracle_parquet_map(state, Lam, beta, U, dmu, hyb, Nf, Nb) - state)) > 1e-9:
        raise ValueError("the returned state is not a fixed point of the parquet map")

    # The converged propagator must satisfy the Dyson equation against the self-energy built from
    # its own converged vertices. This reaches the self-energy step directly rather than only
    # through the sweep, so a broken self-energy cannot hide behind a converged fixed point.
    Phi_fin, G_fin = _unpack(state, Nf, Nb)
    Gam_fin = _oracle_irreducible_vertices(Phi_fin, Lam)
    Sig = _oracle_self_energy(Phi_fin, Gam_fin, G_fin, beta, U)
    if np.abs((1.0/G0 - 1.0/G_fin) - Sig).max() > 1e-9:
        raise ValueError("the converged propagator does not satisfy the Dyson equation with the "
                         "self-energy of its own vertices")

    lam_r = _oracle_stability_spectrum(state, Lam, beta, U, dmu, hyb, Nf, Nb, "reducible")

    # The same physical fixed point, re-expressed in the full vertices F_r = Phi_r + Gamma_r. The
    # strong-coupling sweep inverts the Bethe-Salpeter equations instead of summing them, so it is
    # a different map with the same fixed point -- which is exactly what makes it a check on both.
    F_fin = Phi_fin + Gam_fin
    fstate = _pack(F_fin, G_fin)
    if np.max(np.abs(_oracle_strong_coupling_map(fstate, Lam, beta, U, dmu, hyb, Nf, Nb)
                     - fstate)) > 1e-9:
        raise ValueError("the full-vertex state is not a fixed point of the strong-coupling sweep, "
                         "so the two schemes do not share the physical fixed point")
    if np.abs(_oracle_gamma_from_full_vertex(F_fin, G_fin, beta, Nb) - Gam_fin).max() > 1e-9:
        raise ValueError("inverting the Bethe-Salpeter equations does not return the irreducible "
                         "vertices the fixed point was built from")
    lam_f = _oracle_stability_spectrum(fstate, Lam, beta, U, dmu, hyb, Nf, Nb, "full")

    bounds = []
    for lam in (lam_r, lam_f):
        if lam.real.min() <= 0.0:
            raise ValueError("an eigenvalue has non-positive real part; damping cannot stabilise "
                             "this fixed point and the minimal-damping formula does not apply")
        bounds.append(np.min(c*2.0*np.abs(lam.real)/np.abs(lam)**2))
    return float(min(1.0, min(bounds)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np", "call": "minimal_damping_audit(0.9, 0.3)",
         "tol": 1e-8, "gold_call": "_oracle_minimal_damping_audit(0.9, 0.3)"},
        # boundary: a much smaller safety factor, which scales the answer linearly
        {"setup": "import numpy as np", "call": "minimal_damping_audit(0.5, 0.3)",
         "tol": 1e-8, "gold_call": "_oracle_minimal_damping_audit(0.5, 0.3)"},
        # edge: a different damping used to reach the fixed point, which must not move the answer
        {"setup": "import numpy as np", "call": "minimal_damping_audit(0.9, 0.5)",
         "tol": 1e-8, "gold_call": "_oracle_minimal_damping_audit(0.9, 0.5)"},
    ]
