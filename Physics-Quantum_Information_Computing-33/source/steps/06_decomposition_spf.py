"""
Decomposition-averaged stabilizer partition function of a single-qubit mixed state.

A mixed state rho with Bloch vector r admits many convex decompositions rho = sum_i p_i |phi_i><phi_i| into pure states; in Bloch-vector language, r = sum_i p_i n_i with unit vectors n_i and probabilities p_i. Each decomposition yields the average sum_i p_i Z_beta(phi_i) of the pure-state partition functions of Definition 3. Different decompositions of the same rho generally yield different averages; this step evaluates the average for one explicitly supplied decomposition and checks that the decomposition really reproduces r.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def decomposition_spf(
    r: np.ndarray, weights: np.ndarray, bloch_vectors: np.ndarray, beta: float
) -> float:
    """Average pure-state partition function over one convex decomposition.

    For the decomposition rho = sum_i weights[i] |phi_i><phi_i| of the
    single-qubit state with Bloch vector ``r``, where |phi_i> is the
    pure state with unit Bloch vector ``bloch_vectors[i]``, return
    sum_i weights[i] Z_beta(phi_i) with Z_beta the pure-state
    partition function of ``stabilizer_partition_function`` evaluated
    on the signed Pauli spectrum of each |phi_i><phi_i| (build the
    density matrices with ``bloch_density_matrix`` and the spectra with
    ``pauli_spectrum``).

    Supported domain: ``r`` finite real of shape (3,) with norm at most
    1; ``weights`` finite real of shape (m,), m >= 1, nonnegative and
    summing to 1 within 1e-9; ``bloch_vectors`` finite real of shape
    (m, 3) with every row of unit norm within 1e-9; the decomposition
    must reproduce ``r``, i.e. max_j |sum_i weights[i] bloch_vectors[i, j]
    - r[j]| <= 1e-8; ``beta`` finite and >= 0. Raise ValueError for
    inputs outside this domain, including a decomposition that does not
    reproduce ``r``.

    Parameters
    ----------
    r : np.ndarray
        Bloch vector of the decomposed state, shape (3,).
    weights : np.ndarray
        Decomposition probabilities, shape (m,).
    bloch_vectors : np.ndarray
        Unit Bloch vectors of the pure components, shape (m, 3).
    beta : float
        Inverse-temperature parameter, finite and >= 0.

    Returns
    -------
    z_avg : float
        sum_i weights[i] Z_beta(phi_i).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_decomposition_spf(
    r: np.ndarray, weights: np.ndarray, bloch_vectors: np.ndarray, beta: float
) -> float:
    """Reference oracle: validated decomposition average of pure-state Z_beta."""
    import numpy as np

    rv = np.asarray(r, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    vecs = np.asarray(bloch_vectors, dtype=float)
    b = float(beta)
    if rv.size != 3 or not np.all(np.isfinite(rv)) or float(np.linalg.norm(rv)) > 1.0 + 1e-9:
        raise ValueError("r must be a finite Bloch vector of shape (3,) with norm <= 1.")
    if w.size < 1 or not np.all(np.isfinite(w)) or np.any(w < -1e-12):
        raise ValueError("weights must be a nonempty finite nonnegative array.")
    if abs(float(np.sum(w)) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1.")
    if vecs.ndim != 2 or vecs.shape != (w.size, 3) or not np.all(np.isfinite(vecs)):
        raise ValueError("bloch_vectors must be a finite array of shape (m, 3).")
    if float(np.max(np.abs(np.linalg.norm(vecs, axis=1) - 1.0))) > 1e-9:
        raise ValueError("every Bloch vector of a pure component must have unit norm.")
    if float(np.max(np.abs(w @ vecs - rv))) > 1e-8:
        raise ValueError("the decomposition does not reproduce the Bloch vector r.")
    if not np.isfinite(b) or b < 0.0:
        raise ValueError("beta must be finite and >= 0.")

    # Steps 03, 04 and 05 are available in the concatenated Studio namespace; the inline
    # formula below is a fallback only for isolated execution of this step.
    rho_fn = globals().get("_oracle_bloch_density_matrix")
    spec_fn = globals().get("_oracle_pauli_spectrum")
    z_fn = globals().get("_oracle_stabilizer_partition_function")

    total = 0.0
    for p, n in zip(w, vecs):
        n = n / float(np.linalg.norm(n))
        if callable(rho_fn) and callable(spec_fn) and callable(z_fn):
            spec = np.asarray(spec_fn(rho_fn(n), 1), dtype=float)
            z = float(z_fn(spec, b))
        else:
            spec = np.concatenate(([1.0], n))
            a = np.abs(spec)
            z = float(0.5 * np.sum(np.exp(-b * (1.0 - a)) + np.exp(-b * (1.0 + a))))
        total += float(p) * z
    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    run = (
        "def run(fn):\n"
        "    try:\n"
        "        return [0.0, float(np.round(fn(r, w, vecs, beta), 10))]\n"
        "    except ValueError:\n"
        "        return [1.0, 0.0]\n"
    )
    return [
        {
            # Eigendecomposition of the task state: two components along +-r/|r|.
            "setup": (
                "import numpy as np\n"
                "r = np.array([0.6, 0.5, 0.3])\n"
                "nr = np.linalg.norm(r)\n"
                "w = np.array([(1.0 + nr) / 2.0, (1.0 - nr) / 2.0])\n"
                "vecs = np.array([r / nr, -r / nr])\n"
                "beta = 2.0\n" + run
            ),
            "call": "run(decomposition_spf)",
            "gold_call": "run(_oracle_decomposition_spf)",
        },
        {
            # Decomposition of a free (octahedron-interior) state into stabilizer states.
            "setup": (
                "import numpy as np\n"
                "r = np.array([0.3, 0.3, 0.3])\n"
                "w = np.array([0.3, 0.3, 0.35, 0.05])\n"
                "vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])\n"
                "beta = 2.0\n" + run
            ),
            "call": "run(decomposition_spf)",
            "gold_call": "run(_oracle_decomposition_spf)",
        },
        {
            # Trivial decomposition of the pure T state.
            "setup": (
                "import numpy as np\n"
                "r = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)\n"
                "w = np.array([1.0])\n"
                "vecs = r.reshape(1, 3)\n"
                "beta = 2.0\n" + run
            ),
            "call": "run(decomposition_spf)",
            "gold_call": "run(_oracle_decomposition_spf)",
        },
        {
            # Non-eigen two-point decomposition of the task state through |0>, at beta = 1.
            "setup": (
                "import numpy as np\n"
                "r = np.array([0.6, 0.5, 0.3])\n"
                "n1 = np.array([0.0, 0.0, 1.0])\n"
                "p = (1.0 - r @ r) / (2.0 * (1.0 - r @ n1))\n"
                "n2 = (r - p * n1) / (1.0 - p)\n"
                "w = np.array([p, 1.0 - p])\n"
                "vecs = np.array([n1, n2])\n"
                "beta = 1.0\n" + run
            ),
            "call": "run(decomposition_spf)",
            "gold_call": "run(_oracle_decomposition_spf)",
        },
        {
            # Decomposition that does not reproduce r must be rejected.
            "setup": (
                "import numpy as np\n"
                "r = np.array([0.6, 0.5, 0.3])\n"
                "w = np.array([0.5, 0.5])\n"
                "vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])\n"
                "beta = 2.0\n" + run
            ),
            "call": "run(decomposition_spf)",
            "gold_call": "run(_oracle_decomposition_spf)",
        },
    ]
