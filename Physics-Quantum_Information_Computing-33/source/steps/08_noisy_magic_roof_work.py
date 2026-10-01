"""
Stabilizer work of a noisy single-qubit magic state under the mixed-state extension.

Definition 6 of the source paper defines the stabilizer work through the core partition function of Definition 4, Z^c_beta = Z_beta - 4^n e^{-beta}, compared with the unique n-qubit stabilizer reference Z^c_beta(STAB_n) = 2^n e^{-beta}(cosh beta - 1) with a base-2 logarithm, W_beta = log2[Z^c_beta(STAB_n)/Z^c_beta]. For a mixed state the partition function is the supremum of Definition 8 (the roof of `$roof_spf$`); the subtracted constant is decomposition independent, so the core of the roof is the roof of the core. Because the paper shows the roof attains the stabilizer value exactly on mixtures of stabilizer states, the mixed-state work vanishes on every state inside the stabilizer octahedron and is positive outside it, and for a pure state it reduces to the pure-state work.

The default arguments are the task configuration: Bloch vector r = (0.6, 0.5, 0.3) at beta = 2. The eigendecomposition average and the direct evaluation of Definition 3 on the mixed state's own Pauli spectrum are both admissible-looking but wrong; the former is one decomposition rather than the supremum, the latter is not a decomposition at all.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def noisy_magic_roof_work(r: np.ndarray = (0.6, 0.5, 0.3), beta: float = 2.0) -> float:
    """Mixed-state stabilizer work of a single noisy magic qubit.

    Build the mixed-state partition function Z_beta(rho) of the state
    with Bloch vector ``r`` with ``roof_spf`` (Definition 8 of the
    source paper), form its core Z_beta(rho) - 4 e^{-beta} (Definition
    4), and return the stabilizer work of Definition 6,
    W_beta(rho) = log2[Z^c_beta(STAB_1) / Z^c_beta(rho)] with the
    single-qubit stabilizer reference Z^c_beta(STAB_1) =
    2 e^{-beta}(cosh beta - 1) and the paper's base-2 logarithm.

    The result must be exactly 0 (return 0.0) whenever ``r`` lies in
    the stabilizer octahedron |r_x| + |r_y| + |r_z| <= 1, must reduce
    to the pure-state work when ``r`` is a unit vector, and must never
    exceed the work computed from the eigendecomposition average.

    Supported domain: ``r`` finite real of shape (3,) with Euclidean
    norm at most 1; ``beta`` finite with 0.01 <= beta <= 10. Raise
    ValueError otherwise.

    Parameters
    ----------
    r : np.ndarray, optional
        Bloch vector of the state, default (0.6, 0.5, 0.3).
    beta : float, optional
        Inverse-temperature parameter, default 2.0.

    Returns
    -------
    work : float
        Mixed-state stabilizer work W_beta(rho).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_noisy_magic_roof_work(r: np.ndarray = (0.6, 0.5, 0.3), beta: float = 2.0) -> float:
    """Reference oracle: W = log2(Z^c_STAB_1 / (Z_roof - 4 e^{-beta})) via step 07."""
    import numpy as np

    rv = np.asarray(r, dtype=float).reshape(-1)
    b = float(beta)
    if rv.size != 3 or not np.all(np.isfinite(rv)) or float(np.linalg.norm(rv)) > 1.0 + 1e-9:
        raise ValueError("r must be a finite Bloch vector of shape (3,) with norm <= 1.")
    if not np.isfinite(b) or b < 0.01 or b > 10.0:
        raise ValueError("beta must be finite with 0.01 <= beta <= 10.")

    # Step 07 is available in the concatenated Studio namespace.
    roof_fn = globals().get("_oracle_roof_spf")
    if not callable(roof_fn):
        raise RuntimeError("the roof partition function of step 07 is required.")

    z_roof = float(roof_fn(rv, b))
    zc = z_roof - 4.0 * float(np.exp(-b))
    # Reference: 2 e^{-b}(cosh b - 1) = expm1(-b)**2, evaluated without cancellation.
    zc_ref = float(np.expm1(-b) ** 2)
    if not np.isfinite(zc) or zc <= 0.0 or zc_ref <= 0.0:
        raise RuntimeError("core partition functions must be positive before log2.")
    work = float(np.log2(zc_ref / zc))
    # Free states: the roof attains the stabilizer value, so the work is exactly 0.
    if float(np.sum(np.abs(rv))) <= 1.0 + 1e-12 or abs(work) < 1e-9:
        return 0.0
    if work < 0.0:
        raise RuntimeError("negative work indicates a roof above the stabilizer value.")
    # Consistency: the roof is at least the eigendecomposition average.
    unit = rv / float(np.linalg.norm(rv))
    a = np.abs(unit)
    z_eig = float(
        0.5 * (1.0 + np.exp(-2.0 * b))
        + 0.5 * np.sum(np.exp(-b * (1.0 - a)) + np.exp(-b * (1.0 + a)))
    )
    work_eig = float(np.log2(zc_ref / (z_eig - 4.0 * float(np.exp(-b)))))
    if work > work_eig + 1e-9:
        raise RuntimeError("roof work exceeds the eigendecomposition work; numerical failure.")
    return work

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pair = "float(np.round({fn}(r, beta), 8))"
    return [
        {
            # Task configuration: 0.04052153 (eigendecomposition 0.12911313, naive spectrum 0.36101020).
            "setup": "import numpy as np\nr = np.array([0.6, 0.5, 0.3])\nbeta = 2.0\n",
            "call": pair.format(fn="noisy_magic_roof_work"),
            "gold_call": pair.format(fn="_oracle_noisy_magic_roof_work"),
        },
        {
            # Free state inside the octahedron: exactly 0.
            "setup": "import numpy as np\nr = np.array([0.3, 0.3, 0.3])\nbeta = 2.0\n",
            "call": pair.format(fn="noisy_magic_roof_work"),
            "gold_call": pair.format(fn="_oracle_noisy_magic_roof_work"),
        },
        {
            # Pure T state recovers the pure-state work 0.11007675.
            "setup": "import numpy as np\nr = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)\nbeta = 2.0\n",
            "call": pair.format(fn="noisy_magic_roof_work"),
            "gold_call": pair.format(fn="_oracle_noisy_magic_roof_work"),
        },
        {
            # Task state at beta = 1: 0.01102986.
            "setup": "import numpy as np\nr = np.array([0.6, 0.5, 0.3])\nbeta = 1.0\n",
            "call": pair.format(fn="noisy_magic_roof_work"),
            "gold_call": pair.format(fn="_oracle_noisy_magic_roof_work"),
        },
        {
            # Depolarised T state (Bloch length 0.8) at beta = 2: 0.00834984.
            "setup": "import numpy as np\nr = 0.8 * np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)\nbeta = 2.0\n",
            "call": pair.format(fn="noisy_magic_roof_work"),
            "gold_call": pair.format(fn="_oracle_noisy_magic_roof_work"),
        },
        {
            "setup": "import numpy as np\nr = np.array([0.0, 0.0, 0.999999])\nbeta = 2.0\n",
            "call": "float(np.round(noisy_magic_roof_work(r, beta), 8))",
            "gold_call": "float(np.round(_oracle_noisy_magic_roof_work(r, beta), 8))",
        },
    ]
