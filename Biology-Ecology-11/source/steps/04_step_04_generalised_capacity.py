"""
This stage computes lambda_max, the normalised positive right and left Perron vectors, the leading eigenvalue of the kernel alone for comparison with the classical capacity, and whether the kernel is irreducible. Irreducibility is decided on the pattern of positive entries, since a directed walk must join every ordered pair of patches.

The occupancy dynamics dp_i/dt = -e_i p_i + (1 - p_i) sum_j K_ij c_j p_j always admit the extinct state p = 0. Linearised about it, dp/dt = (K C - E) p with C = diag(c_i) and E = diag(e_i). When every patch has the same extinction rate e and fecundity c, the classical result applies: the species persists exactly when the leading eigenvalue of the kernel, the metapopulation capacity of the landscape, exceeds e / c. When the local rates differ from patch to patch no single ratio e / c exists, and the threshold is a global property that mixes the local rates of every patch with the kernel.

The generalisation uses the landscape matrix K C E^(-1). For an irreducible non-negative kernel it is irreducible and has a Perron-Frobenius eigenvalue lambda_max > 0 with a strictly positive left eigenvector u. Projecting the linearised dynamics on u gives d(u^T p)/dt = (lambda_max - 1) u^T E p, and bounding u^T E p between e_min u^T p and e_max u^T p shows that the extinct state is unstable when lambda_max > 1 and globally attracting when lambda_max < 1, the second because the full dynamics are bounded above by their linearisation. The generalised metapopulation capacity is lambda_max. It is dimensionless, it equals one at the persistence threshold, and multiplying every extinction rate by a common factor theta divides it by theta. The matrices K C E^(-1), E^(-1) K C and C E^(-1) K are similar, so they share lambda_max; and a similarity transformation by the site numbers, which converts the density kernel into the count kernel, leaves it unchanged too, which is why the capacity does not depend on how many sites the patches hold.

Returns
-------
dict, the generalised metapopulation capacity and its Perron eigenvectors, keyed by capacity (lambda_max), kernel_radius (the spectral radius of K), perron_right and perron_left (the positive Perron eigenvectors of K C E^(-1), each normalised to unit sum), perron_residual (the largest modulus of the two eigen-equation residuals), and irreducible (the integer 1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generalised_capacity(
    kernel: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
) -> dict:
    """Compute the generalised metapopulation capacity of a colonisation kernel with patch-dependent local rates.

    Parameters
    ----------
    kernel : np.ndarray
        Effective colonisation kernel K_ij, non-negative.
    fecundity : np.ndarray
        Patch fecundities c_i, above zero.
    extinction : np.ndarray
        Patch extinction rates e_i, above zero.

    Returns
    -------
    dict
        Under the keys capacity, kernel_radius, perron_right, perron_left, perron_residual and irreducible; each Perron vector is positive and normalised to unit sum.

    Raises
    ------
    ValueError
        When the kernel fails to be a finite, non-negative square array of at least two patches, when the fecundities or extinction rates fail to be finite, above zero and of matching length, or when the kernel is not irreducible, since the Perron eigenvalue then fails to decide persistence.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _perron(matrix):
    values, vectors = np.linalg.eig(matrix)
    lead = int(np.argmax(values.real))
    vector = vectors[:, lead]
    vector = vector * np.exp(-1j * np.angle(vector[np.argmax(np.abs(vector))]))
    vector = vector.real
    return float(values[lead].real), vector / vector.sum()


def _is_irreducible(matrix):
    n = matrix.shape[0]
    reach = (matrix > 0.0).astype(float) + np.eye(n)
    power = np.eye(n)
    for _ in range(n - 1):
        power = np.minimum(power @ reach, 1.0)
    return bool(np.all(power > 0.0))


def _oracle_generalised_capacity(
    kernel: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
) -> dict:
    """Reference implementation."""
    k = np.asarray(kernel, dtype=float)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] < 2:
        raise ValueError("kernel must be a square array of at least two patches")
    if not np.all(np.isfinite(k)) or np.any(k < 0.0):
        raise ValueError("kernel must be finite and non-negative")
    n = k.shape[0]
    c = np.asarray(fecundity, dtype=float)
    e = np.asarray(extinction, dtype=float)
    for arr in (c, e):
        if arr.shape != (n,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("fecundity and extinction must be finite, above zero and one per patch")
    if not _is_irreducible(k):
        raise ValueError("the kernel is not irreducible")

    landscape = k * (c / e)[None, :]
    capacity, right = _perron(landscape)
    _, left = _perron(landscape.T)
    residual = max(float(np.max(np.abs(landscape @ right - capacity * right))),
                   float(np.max(np.abs(left @ landscape - capacity * left))))
    return {
        "capacity": capacity,
        "kernel_radius": float(np.max(np.abs(np.linalg.eigvals(k)))),
        "perron_right": right,
        "perron_left": left,
        "perron_residual": residual,
        "irreducible": 1,
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
    rng = np.random.default_rng(20260916)
    K = rng.uniform(0.0, 0.3, (9, 9)) * (rng.random((9, 9)) < 0.6)
    for i in range(9):
        K[(i + 1) % 9, i] += 0.05
    C = rng.uniform(0.6, 1.4, 9)
    E = rng.uniform(0.1, 0.4, 9)
    def digest(out):
        return (round(out["capacity"], 10), round(out["kernel_radius"], 10), np.round(out["perron_right"], 9),
                np.round(out["perron_left"], 9), int(out["perron_residual"] < 1e-10), out["irreducible"])
    """
    import textwrap as _textwrap
    FLAT = _textwrap.dedent(FLAT)
    SETUP = _textwrap.dedent(SETUP)
    return [
        {
            # a seeded sparse irreducible kernel with heterogeneous local rates
            "setup": SETUP + FLAT,
            "call": "flat(digest(generalised_capacity(K, C, E)))",
            "gold_call": "flat(digest(_oracle_generalised_capacity(K, C, E)))",
        },
        {
            # homogeneous boundary: a circulant kernel with constant row sums has capacity row sum times
            # c / e and uniform Perron vectors; multiplying every extinction rate by 2.5 divides the capacity
            # by 2.5 in a heterogeneous landscape
            "setup": SETUP + """
R = np.array([[0.0, 0.2, 0.05, 0.05, 0.2], [0.2, 0.0, 0.2, 0.05, 0.05], [0.05, 0.2, 0.0, 0.2, 0.05],
              [0.05, 0.05, 0.2, 0.0, 0.2], [0.2, 0.05, 0.05, 0.2, 0.0]])
def boundary(fn):
    a = fn(R, np.full(5, 1.5), np.full(5, 0.3))
    b = fn(K, C, E)
    s = fn(K, C, 2.5 * E)
    return (round(a["capacity"], 12), np.round(a["perron_right"], 12), round(b["capacity"] / s["capacity"], 12))
""" + FLAT,
            "call": "flat(boundary(generalised_capacity))",
            "gold_call": "flat(boundary(_oracle_generalised_capacity))",
        },
        {
            # similarity: converting the kernel with a set of site numbers, as between the density and the
            # count forms, leaves the capacity unchanged
            "setup": SETUP + """
S = np.array([3.0, 1.0, 7.0, 2.0, 5.0, 4.0, 1.5, 6.0, 2.5])
def similar(fn):
    a = fn(K, C, E)
    b = fn(K * (S[:, None] / S[None, :]), C, E)
    return (round(a["capacity"] - b["capacity"], 12), round(b["capacity"], 10))
""" + FLAT,
            "call": "flat(similar(generalised_capacity))",
            "gold_call": "flat(similar(_oracle_generalised_capacity))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(kernel=K, fecundity=C, extinction=E)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
ISOLATED = K.copy(); ISOLATED[3, :] = 0.0
NEG = K.copy(); NEG[0, 1] = -1e-3
""",
            "call": "(verdict(generalised_capacity, kernel=ISOLATED), verdict(generalised_capacity, kernel=NEG), verdict(generalised_capacity, kernel=K[:, :4]), verdict(generalised_capacity, fecundity=C[:3]), verdict(generalised_capacity, extinction=0.0 * E), verdict(generalised_capacity, extinction=E * float('nan')), verdict(generalised_capacity))",
            "gold_call": "(verdict(_oracle_generalised_capacity, kernel=ISOLATED), verdict(_oracle_generalised_capacity, kernel=NEG), verdict(_oracle_generalised_capacity, kernel=K[:, :4]), verdict(_oracle_generalised_capacity, fecundity=C[:3]), verdict(_oracle_generalised_capacity, extinction=0.0 * E), verdict(_oracle_generalised_capacity, extinction=E * float('nan')), verdict(_oracle_generalised_capacity))",
        },
    ]
