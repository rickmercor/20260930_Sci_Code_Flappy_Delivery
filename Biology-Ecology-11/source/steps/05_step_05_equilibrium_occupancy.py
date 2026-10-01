"""
The stage runs the fixed-point iteration from full occupancy, polishes the result with Newton steps on the residual -e_i p_i + (1 - p_i) s_i, and reports the stability of the result through the largest real part of the eigenvalues of the Jacobian J = -diag(e + s) + diag(1 - p) K C of the occupancy dynamics.

Above the persistence threshold the extinct state is unstable, and the occupancy dynamics dp_i/dt = -e_i p_i + (1 - p_i) sum_j K_ij c_j p_j settle on a non-trivial steady state. Writing s_i = sum_j K_ij c_j p_j for the colonisation pressure on patch i, a steady state satisfies

p_i = s_i / (e_i + s_i),

and for an irreducible kernel every component of a non-trivial steady state is strictly positive: a patch with p_i = 0 would need zero pressure, hence zero occupancy in every patch that colonises it, and by irreducibility in every patch. The map p -> s / (e + s) is monotone and concave on the unit cube, so starting from full occupancy its iterates decrease monotonically to the largest fixed point, which for a capacity above one is the unique positive steady state and is locally stable; for a capacity at or below one they decrease to extinction.

Unlike the capacity, the steady state depends on the site numbers. The density kernel carries the ratios of site numbers, so a large patch pressing on a small one raises its occupancy more than the reverse. The number of occupied sites, sum_i M_i p_i, is the natural landscape total.

Returns
-------
dict, the stable steady-state occupancy and its landscape total, keyed by occupancy (the p_i), occupied_sites (sum_i M_i p_i), mean_occupancy (the unweighted mean of p_i), residual (the largest modulus of the steady-state equations), stability_abscissa (the largest real part of the eigenvalues of the occupancy Jacobian), and persistent (the integer 1 if the steady state is non-trivial, else 0).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def equilibrium_occupancy(
    kernel: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    sites: np.ndarray,
) -> dict:
    """Solve for the stable steady state of the effective occupancy dynamics and its landscape total.

    Parameters
    ----------
    kernel : np.ndarray
        Effective colonisation kernel K_ij.
    fecundity : np.ndarray
        Patch fecundities c_i.
    extinction : np.ndarray
        Patch extinction rates e_i.
    sites : np.ndarray
        Site numbers M_i.

    Returns
    -------
    dict
        Under the keys occupancy, occupied_sites, mean_occupancy, residual, stability_abscissa and persistent; persistent is the integer 1 if the steady state is non-trivial and 0 otherwise.

    Raises
    ------
    ValueError
        When the kernel, the local rates or the site numbers are invalid, when the kernel is not irreducible, or when the iteration fails to converge.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_equilibrium_occupancy(
    kernel: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    sites: np.ndarray,
) -> dict:
    """Reference implementation."""
    k = np.asarray(kernel, dtype=float)
    m = np.asarray(sites, dtype=float)
    if m.ndim != 1 or k.shape != (m.size, m.size) or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("sites must be finite, above zero and one per patch of the kernel")
    capacity = _oracle_generalised_capacity(k, fecundity, extinction)["capacity"]  # noqa: F821
    c = np.asarray(fecundity, dtype=float)
    e = np.asarray(extinction, dtype=float)
    pressure_matrix = k * c[None, :]
    n = m.size

    p = np.ones(n)
    if capacity > 1.0:
        for _ in range(1000000):
            s = pressure_matrix @ p
            nxt = s / (e + s)
            if np.max(np.abs(nxt - p)) < 1e-15:
                p = nxt
                break
            p = nxt
        else:
            raise ValueError("the fixed-point iteration failed to converge")
        for _ in range(20):
            s = pressure_matrix @ p
            g = -e * p + (1.0 - p) * s
            jac = -np.diag(e + s) + (1.0 - p)[:, None] * pressure_matrix
            step = np.linalg.solve(jac, -g)
            p = p + step
            if np.max(np.abs(step)) < 1e-16:
                break
        if np.any(p <= 0.0) or np.any(p >= 1.0):
            raise ValueError("the steady state left the unit cube")
    else:
        p = np.zeros(n)

    s = pressure_matrix @ p
    residual = float(np.max(np.abs(-e * p + (1.0 - p) * s)))
    jac = -np.diag(e + s) + (1.0 - p)[:, None] * pressure_matrix
    return {
        "occupancy": p,
        "occupied_sites": float(m @ p),
        "mean_occupancy": float(p.mean()),
        "residual": residual,
        "stability_abscissa": float(np.max(np.linalg.eigvals(jac).real)),
        "persistent": int(capacity > 1.0),
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
    rng = np.random.default_rng(916)
    K = rng.uniform(0.0, 0.5, (8, 8)) * (rng.random((8, 8)) < 0.55)
    for i in range(8):
        K[(i + 1) % 8, i] += 0.1
    C = rng.uniform(0.7, 1.3, 8)
    E = rng.uniform(0.05, 0.25, 8)
    M = rng.integers(20, 300, 8).astype(float)
    def digest(out):
        return (np.round(out["occupancy"], 6), round(out["occupied_sites"], 3), round(out["mean_occupancy"], 6),
                int(out["residual"] < 1e-8), int(out["stability_abscissa"] < 0.0), round(out["stability_abscissa"], 6),
                out["persistent"])
    """
    import textwrap as _textwrap
    FLAT = _textwrap.dedent(FLAT)
    SETUP = _textwrap.dedent(SETUP)
    return [
        {
            # a seeded irreducible kernel well above threshold
            "setup": SETUP + FLAT,
            "call": "flat(digest(equilibrium_occupancy(K, C, E, M)))",
            "gold_call": "flat(digest(_oracle_equilibrium_occupancy(K, C, E, M)))",
        },
        {
            # homogeneous boundary: a kernel with constant row sums k0 gives the uniform steady state
            # p = 1 - e / (c k0), and below threshold the landscape empties
            "setup": """
import numpy as np
R = np.array([[0.0, 0.3, 0.1, 0.3], [0.3, 0.0, 0.3, 0.1], [0.1, 0.3, 0.0, 0.3], [0.3, 0.1, 0.3, 0.0]])
def homogeneous(fn):
    a = fn(R, np.full(4, 1.0), np.full(4, 0.35), np.full(4, 50.0))
    b = fn(R, np.full(4, 1.0), np.full(4, 0.9), np.full(4, 50.0))
    return (np.round(a["occupancy"] - 0.5, 7), round(a["occupied_sites"], 4), b["persistent"],
            round(b["occupied_sites"], 12), int(b["stability_abscissa"] < 0.0))
""" + FLAT,
            "call": "flat(homogeneous(equilibrium_occupancy))",
            "gold_call": "flat(homogeneous(_oracle_equilibrium_occupancy))",
        },
        {
            # close to threshold the fixed point is slow but still exact; the site numbers weight the total
            # without entering the occupancies passed in
            "setup": SETUP + """
def near(fn):
    lead = max(abs(np.linalg.eigvals(K * (C / E)[None, :])))
    a = fn(K, C, E * lead / 1.02, M)
    b = fn(K, C, E * lead / 1.02, 2.0 * M)
    return (np.round(a["occupancy"], 5), round(b["occupied_sites"] / a["occupied_sites"], 8), int(a["residual"] < 1e-8))
""" + FLAT,
            "call": "flat(near(equilibrium_occupancy))",
            "gold_call": "flat(near(_oracle_equilibrium_occupancy))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(kernel=K, fecundity=C, extinction=E, sites=M)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
REDUCIBLE = np.triu(K)
""",
            "call": "(verdict(equilibrium_occupancy, sites=M[:3]), verdict(equilibrium_occupancy, sites=0.0 * M), verdict(equilibrium_occupancy, kernel=REDUCIBLE), verdict(equilibrium_occupancy, kernel=-K), verdict(equilibrium_occupancy, extinction=-E), verdict(equilibrium_occupancy))",
            "gold_call": "(verdict(_oracle_equilibrium_occupancy, sites=M[:3]), verdict(_oracle_equilibrium_occupancy, sites=0.0 * M), verdict(_oracle_equilibrium_occupancy, kernel=REDUCIBLE), verdict(_oracle_equilibrium_occupancy, kernel=-K), verdict(_oracle_equilibrium_occupancy, extinction=-E), verdict(_oracle_equilibrium_occupancy))",
        },
    ]
