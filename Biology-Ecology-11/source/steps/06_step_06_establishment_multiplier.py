"""
Introduce the invader in small numbers into the landscape held by the resident at its own deterministic steady state p_1*. Linearised in the invader, its occupancies grow as dp_2/dt = [diag(1 - p_1*) K_2 C_2 - E_2] p_2, while the resident is left unchanged at first order. The invader increases exactly when the Perron eigenvalue of diag(1 - p_1*) K_2 C_2 E_2^(-1) exceeds one: the generalised capacity of the kernel diag(1 - p_1*) K_2, the invader's kernel thinned by the space the resident holds. Multiplying every fecundity c_2i by a common factor scales that eigenvalue by the same factor, so the multiplier at which the invader changes from declining to increasing is its reciprocal. A multiplier below one means the invader can already establish at its stated fecundities.

The stage computes the resident steady state, the resident-conditioned invasion capacity, the invader's capacity alone for comparison, and the establishment multiplier.

When a second species arrives in a landscape already held by a resident, persistence alone is no longer the question. Both species compete for the same sites: an explorer of either species that attempts a site held by any settled individual is consumed, so the effective occupancy dynamics of species a read

dp_ai/dt = -e_ai p_ai + (1 - sum_b p_bi) sum_j K_a,ij c_aj p_aj,

and competition enters only through the shared free-space factor. The capacity of the invader alone, the Perron eigenvalue of K_2 C_2 E_2^(-1), is therefore necessary for it to establish but not sufficient.

Returns
-------
dict, the resident state and the invader's establishment threshold, keyed by resident_occupancy, resident_sites, resident_capacity, invader_capacity, invasion_capacity, and multiplier.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def establishment_multiplier(
    resident_kernel: np.ndarray,
    invader_kernel: np.ndarray,
    resident_fecundity: np.ndarray,
    resident_extinction: np.ndarray,
    invader_fecundity: np.ndarray,
    invader_extinction: np.ndarray,
    sites: np.ndarray,
) -> dict:
    """Find the common fecundity multiplier at which an invader starts to increase in a landscape held by a resident.

    Parameters
    ----------
    resident_kernel : np.ndarray
        Effective colonisation kernel of the resident.
    invader_kernel : np.ndarray
        Effective colonisation kernel of the invader.
    resident_fecundity : np.ndarray
        Patch fecundities of the resident.
    resident_extinction : np.ndarray
        Patch extinction rates of the resident.
    invader_fecundity : np.ndarray
        Patch fecundities of the invader.
    invader_extinction : np.ndarray
        Patch extinction rates of the invader.
    sites : np.ndarray
        Site numbers M_i.

    Returns
    -------
    dict
        Under the keys resident_occupancy, resident_sites, resident_capacity, invader_capacity, invasion_capacity and multiplier.

    Raises
    ------
    ValueError
        When a kernel, a set of local rates or the site numbers are invalid, when the kernels differ in size, or when the resident cannot persist on its own.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_establishment_multiplier(
    resident_kernel: np.ndarray,
    invader_kernel: np.ndarray,
    resident_fecundity: np.ndarray,
    resident_extinction: np.ndarray,
    invader_fecundity: np.ndarray,
    invader_extinction: np.ndarray,
    sites: np.ndarray,
) -> dict:
    """Reference implementation."""
    k1 = np.asarray(resident_kernel, dtype=float)
    k2 = np.asarray(invader_kernel, dtype=float)
    if k1.shape != k2.shape:
        raise ValueError("the resident and invader kernels must have the same size")
    resident = _oracle_equilibrium_occupancy(k1, resident_fecundity, resident_extinction, sites)  # noqa: F821
    if resident["persistent"] != 1:
        raise ValueError("the resident cannot persist on its own")
    p1 = resident["occupancy"]
    alone = _oracle_generalised_capacity(k2, invader_fecundity, invader_extinction)  # noqa: F821
    thinned = (1.0 - p1)[:, None] * k2
    invasion = _oracle_generalised_capacity(thinned, invader_fecundity, invader_extinction)  # noqa: F821
    resident_capacity = _oracle_generalised_capacity(k1, resident_fecundity, resident_extinction)  # noqa: F821
    return {
        "resident_occupancy": p1,
        "resident_sites": resident["occupied_sites"],
        "resident_capacity": resident_capacity["capacity"],
        "invader_capacity": alone["capacity"],
        "invasion_capacity": invasion["capacity"],
        "multiplier": 1.0 / invasion["capacity"],
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
    rng = np.random.default_rng(2026)
    K1 = rng.uniform(0.0, 0.4, (7, 7)) * (rng.random((7, 7)) < 0.6)
    K2 = rng.uniform(0.0, 0.4, (7, 7)) * (rng.random((7, 7)) < 0.6)
    for i in range(7):
        K1[(i + 1) % 7, i] += 0.1
        K2[i, (i + 1) % 7] += 0.1
    C1 = rng.uniform(0.7, 1.3, 7); E1 = rng.uniform(0.08, 0.25, 7)
    C2 = rng.uniform(0.7, 1.3, 7); E2 = rng.uniform(0.08, 0.25, 7)
    M = rng.integers(30, 250, 7).astype(float)
    def digest(out):
        return (np.round(out["resident_occupancy"], 6), round(out["resident_sites"], 3), round(out["resident_capacity"], 7),
                round(out["invader_capacity"], 7), round(out["invasion_capacity"], 7), round(out["multiplier"], 7))
    """
    import textwrap as _textwrap
    FLAT = _textwrap.dedent(FLAT)
    SETUP = _textwrap.dedent(SETUP)
    return [
        {
            # seeded resident and invader kernels of different topology
            "setup": SETUP + FLAT,
            "call": "flat(digest(establishment_multiplier(K1, K2, C1, E1, C2, E2, M)))",
            "gold_call": "flat(digest(_oracle_establishment_multiplier(K1, K2, C1, E1, C2, E2, M)))",
        },
        {
            # identical species on a homogeneous kernel: the invader meets free space 1 - p1* = e / (c k0) in every
            # patch, so its thinned capacity is exactly one and the multiplier is exactly one; and at a multiplier
            # the invasion capacity of the scaled invader is one
            "setup": """
import numpy as np
R = np.array([[0.0, 0.3, 0.1, 0.3], [0.3, 0.0, 0.3, 0.1], [0.1, 0.3, 0.0, 0.3], [0.3, 0.1, 0.3, 0.0]])
def neutral(fn):
    c = np.full(4, 1.0); e = np.full(4, 0.35); m = np.full(4, 60.0)
    a = fn(R, R, c, e, c, e, m)
    return (round(a["multiplier"], 10), round(a["invasion_capacity"], 10), round(a["resident_sites"], 7))
""" + FLAT,
            "call": "flat(neutral(establishment_multiplier))",
            "gold_call": "flat(neutral(_oracle_establishment_multiplier))",
        },
        {
            # the multiplier never falls below the reciprocal of the invader's capacity alone, and scaling the
            # invader's fecundities by the multiplier puts its invasion capacity at one
            "setup": SETUP + """
def bounds(fn):
    a = fn(K1, K2, C1, E1, C2, E2, M)
    b = fn(K1, K2, C1, E1, a["multiplier"] * C2, E2, M)
    return (int(a["multiplier"] >= 1.0 / a["invader_capacity"] - 1e-12), round(b["invasion_capacity"], 9))
""" + FLAT,
            "call": "flat(bounds(establishment_multiplier))",
            "gold_call": "flat(bounds(_oracle_establishment_multiplier))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(resident_kernel=K1, invader_kernel=K2, resident_fecundity=C1, resident_extinction=E1,
                invader_fecundity=C2, invader_extinction=E2, sites=M)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "(verdict(establishment_multiplier, invader_kernel=K2[:5, :5]), verdict(establishment_multiplier, resident_extinction=50.0 * E1), verdict(establishment_multiplier, invader_fecundity=-C2), verdict(establishment_multiplier, sites=M[:3]), verdict(establishment_multiplier, resident_kernel=np.triu(K1)), verdict(establishment_multiplier))",
            "gold_call": "(verdict(_oracle_establishment_multiplier, invader_kernel=K2[:5, :5]), verdict(_oracle_establishment_multiplier, resident_extinction=50.0 * E1), verdict(_oracle_establishment_multiplier, invader_fecundity=-C2), verdict(_oracle_establishment_multiplier, sites=M[:3]), verdict(_oracle_establishment_multiplier, resident_kernel=np.triu(K1)), verdict(_oracle_establishment_multiplier))",
        },
    ]
