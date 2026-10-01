"""
Their stable steady state is found here by relaxation: from several interior initial conditions the dynamics are integrated until they settle, and the result is polished by Newton steps on the steady-state equations. Agreement of every start, positivity of every species in every patch, and a Jacobian whose eigenvalues all have negative real parts establish that the pair coexists at a unique stable state. Relaxation is slow when the species are close competitors, because the exchange of space between them decays at a small rate, and the stage reports that rate.

The steady state of these occupancy equations is also the steady state of the full rate equations of settled individuals and explorers, whatever the relative speed of explorers, because the explorer balance holds exactly at a steady state; only the approach to it and the fluctuations about it depend on the explorer speed.

Whether two competitors share a landscape is decided by the pair, not by either species alone. Each species can have a capacity above one and still be excluded by the other, and a species whose capacity is lower can still hold part of the landscape where its local rates favour it. With S species and effective kernels K_a, the deterministic occupancy dynamics

dp_ai/dt = -e_ai p_ai + (1 - sum_b p_bi) sum_j K_a,ij c_aj p_aj

couple the species only through the free space of each patch.

Returns
-------
dict, the stable coexistence state of the occupancy dynamics, keyed by occupancy, held_sites, total_held, start_spread, residual, stability_abscissa, and slowest_rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coexistence_state(
    kernels: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    sites: np.ndarray,
    n_starts: int,
) -> dict:
    """Find the unique stable steady state at which several competing species share the landscape.

    Parameters
    ----------
    kernels : np.ndarray
        Effective colonisation kernels, one per species.
    fecundity : np.ndarray
        Fecundities, one row per species.
    extinction : np.ndarray
        Extinction rates, one row per species.
    sites : np.ndarray
        Site numbers M_i.
    n_starts : int
        Number of interior initial conditions.

    Returns
    -------
    dict
        Under the keys occupancy, held_sites, total_held, start_spread, residual, stability_abscissa and slowest_rate.

    Raises
    ------
    ValueError
        When the inputs are invalid, when the relaxation fails, when any start leaves a species extinct in some patch, when the starts reach different states, or when the state reached is not stable.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp


def _check_species_arrays(kernels, fecundity, extinction, sites):
    k = np.asarray(kernels, dtype=float)
    if k.ndim != 3 or k.shape[1] != k.shape[2] or k.shape[0] < 1 or k.shape[1] < 2:
        raise ValueError("kernels must be an S by N by N array")
    if not np.all(np.isfinite(k)) or np.any(k < 0.0):
        raise ValueError("kernels must be finite and non-negative")
    s, n = k.shape[0], k.shape[1]
    c = np.asarray(fecundity, dtype=float)
    e = np.asarray(extinction, dtype=float)
    m = np.asarray(sites, dtype=float)
    for arr in (c, e):
        if arr.shape != (s, n) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("fecundity and extinction must be finite, above zero and S by N")
    if m.shape != (n,) or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("sites must be finite, above zero and one per patch")
    return k, c, e, m


def _occupancy_drift(p, pressure, e):
    free = 1.0 - p.sum(axis=0)
    return -e * p + free[None, :] * np.einsum("aij,aj->ai", pressure, p)


def _occupancy_jacobian(p, pressure, e):
    s, n = p.shape
    free = 1.0 - p.sum(axis=0)
    gain = np.einsum("aij,aj->ai", pressure, p)
    jac = np.zeros((s * n, s * n))
    for a in range(s):
        for b in range(s):
            block = -np.diag(gain[a])
            if a == b:
                block = block - np.diag(e[a]) + free[:, None] * pressure[a]
            jac[a * n:(a + 1) * n, b * n:(b + 1) * n] = block
    return jac


def _oracle_coexistence_state(
    kernels: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    sites: np.ndarray,
    n_starts: int,
) -> dict:
    """Reference implementation."""
    k, c, e, m = _check_species_arrays(kernels, fecundity, extinction, sites)
    if isinstance(n_starts, bool) or int(n_starts) != n_starts or int(n_starts) < 2:
        raise ValueError("n_starts must be an integer of at least two")
    for a in range(k.shape[0]):
        _oracle_generalised_capacity(k[a], c[a], e[a])  # noqa: F821
    s, n = c.shape
    pressure = k * c[:, None, :]

    finals = []
    for start in range(int(n_starts)):
        phase = np.arange(s * n).reshape(s, n) + 1.0
        weights = 0.2 + 0.8 * np.mod(phase * (0.6180339887 + 0.173 * start), 1.0)
        p0 = 0.9 * weights / weights.sum(axis=0)[None, :]
        sol = solve_ivp(lambda t, y: _occupancy_drift(y.reshape(s, n), pressure, e).ravel(),
                        (0.0, 6000.0), p0.ravel(), method="LSODA", rtol=1e-11, atol=1e-13)
        if not sol.success:
            raise ValueError("the relaxation of the occupancy dynamics failed")
        p = sol.y[:, -1].reshape(s, n)
        for _ in range(30):
            step = np.linalg.solve(_occupancy_jacobian(p, pressure, e), -_occupancy_drift(p, pressure, e).ravel())
            p = p + step.reshape(s, n)
            if np.max(np.abs(step)) < 1e-15:
                break
        if np.any(p <= 1e-9) or np.any(p.sum(axis=0) >= 1.0):
            raise ValueError("a start left a species extinct in some patch or the sites overfilled")
        finals.append(p)
    spread = max(float(np.max(np.abs(f - finals[0]))) for f in finals)
    if spread > 1e-8:
        raise ValueError("the starts reached different states")
    p = finals[0]
    eig = np.linalg.eigvals(_occupancy_jacobian(p, pressure, e)).real
    if np.max(eig) >= 0.0:
        raise ValueError("the state reached is not stable")
    held = p @ m
    return {
        "occupancy": p,
        "held_sites": held,
        "total_held": float(held.sum()),
        "start_spread": spread,
        "residual": float(np.max(np.abs(_occupancy_drift(p, pressure, e)))),
        "stability_abscissa": float(np.max(eig)),
        "slowest_rate": float(-np.max(eig)),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
    def flat(x):
        # flatten to a tuple of plain numeric terminals
        if isinstance(x, (tuple, list)):
            out = []
            for v in x:
                out.extend(flat(v))
            return tuple(out)
        if hasattr(x, "tolist"):
            return flat(x.tolist())
        if isinstance(x, bool):
            return (int(x),)
        return (x,)
    """

    SETUP = """
    import numpy as np
    rng = np.random.default_rng(1709)
    n = 6
    K = np.zeros((2, n, n))
    for a in range(2):
        K[a] = rng.uniform(0.0, 0.5, (n, n)) * (rng.random((n, n)) < 0.7)
        for i in range(n):
            K[a, (i + 1) % n, i] += 0.1
    C = np.array([rng.uniform(0.7, 1.3, n), rng.uniform(0.7, 1.3, n)])
    E = np.array([np.linspace(0.08, 0.30, n), np.linspace(0.30, 0.08, n)])
    M = rng.integers(40, 300, n).astype(float)
    def digest(out):
        return (np.round(out["occupancy"], 6), np.round(out["held_sites"], 3), round(out["total_held"], 3),
                int(out["start_spread"] < 1e-8), int(out["residual"] < 1e-10), round(out["stability_abscissa"], 6),
                round(out["slowest_rate"], 6))
    """
    import textwrap as _textwrap
    FLAT = _textwrap.dedent(FLAT)
    SETUP = _textwrap.dedent(SETUP)
    return [
        {
            # two species with opposed extinction gradients along the patches, which share the landscape
            "setup": SETUP + FLAT,
            "call": "flat(digest(coexistence_state(K, C, E, M, 3)))",
            "gold_call": "flat(digest(_oracle_coexistence_state(K, C, E, M, 3)))",
        },
        {
            # boundary: a single species reduces to the single-species steady state p = s / (e + s), here on a
            # constant-row-sum kernel with the uniform solution 1 - e / (c k0) = 0.5
            "setup": """
import numpy as np
R = np.array([[0.0, 0.3, 0.1, 0.3], [0.3, 0.0, 0.3, 0.1], [0.1, 0.3, 0.0, 0.3], [0.3, 0.1, 0.3, 0.0]])
def single(fn):
    out = fn(R[None, :, :], np.ones((1, 4)), np.full((1, 4), 0.35), np.full(4, 50.0), 2)
    return (np.round(out["occupancy"] - 0.5, 6), round(out["total_held"], 3), round(out["slowest_rate"], 6))
""" + FLAT,
            "call": "flat(single(coexistence_state))",
            "gold_call": "flat(single(_oracle_coexistence_state))",
        },
        {
            # the steady-state equations hold species by species: e p = (1 - sum p) K c p to the residual, and the
            # held sites are the site-weighted occupancies
            "setup": SETUP + """
def identity(fn):
    out = fn(K, C, E, M, 2)
    p = out["occupancy"]
    lhs = E * p
    rhs = (1.0 - p.sum(axis=0))[None, :] * np.einsum("aij,aj->ai", K * C[:, None, :], p)
    return (int(float(np.max(np.abs(lhs - rhs))) < 1e-10), int(float(np.max(np.abs(out["held_sites"] - p @ M))) < 1e-9))
""" + FLAT,
            "call": "flat(identity(coexistence_state))",
            "gold_call": "flat(identity(_oracle_coexistence_state))",
        },
        {
            # a species far below its own threshold cannot coexist, and malformed inputs are refused
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(kernels=K, fecundity=C, extinction=E, sites=M, n_starts=2)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
WEAK = E.copy(); WEAK[1] *= 40.0
""",
            "call": "(verdict(coexistence_state, extinction=WEAK), verdict(coexistence_state, kernels=K[0]), verdict(coexistence_state, fecundity=C[:, :4]), verdict(coexistence_state, sites=-M), verdict(coexistence_state, n_starts=1), verdict(coexistence_state, kernels=-K), verdict(coexistence_state))",
            "gold_call": "(verdict(_oracle_coexistence_state, extinction=WEAK), verdict(_oracle_coexistence_state, kernels=K[0]), verdict(_oracle_coexistence_state, fecundity=C[:, :4]), verdict(_oracle_coexistence_state, sites=-M), verdict(_oracle_coexistence_state, n_starts=1), verdict(_oracle_coexistence_state, kernels=-K), verdict(_oracle_coexistence_state))",
        },
    ]
