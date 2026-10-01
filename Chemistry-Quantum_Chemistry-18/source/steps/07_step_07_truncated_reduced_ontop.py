"""
Step 07: Thresholded truncated reduced on-top density. Reduced on-top density at distance r from the centre estimated from the natural orbitals whose occupation numbers reach a threshold.

A calculation in a finite one-electron basis has access only to part of the natural-orbital expansion, and the usual
model of that situation keeps the orbitals whose occupation numbers are largest and renormalizes what is retained. The
estimate defined in the problem statement follows that model: the retained amplitudes build both the on-top density and
the one-electron density, and the retained norm multiplies the result.

Degenerate orbitals are kept or dropped together, so the retained set grows in whole shells and its size jumps as the
threshold is lowered. Because the discarded orbitals carry the part of the correlation that keeps the electrons apart,
the estimate approaches the exact value from one side only.

Returns
-------
numpy.ndarray [phi_N(r), N, K]: thresholded truncated reduced on-top density and retained orbital counts
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def truncated_reduced_ontop(omega: float, c: float, r: float, occ_min: float) -> "np.ndarray":
    '''Truncated estimate phi_N(r) from all natural orbitals with occupation lambda_n^2 >= occ_min.

    Parameters
    ----------
    omega : float
        Gaussian exponent parameter, 0.05 <= omega <= 5 (atomic units).
    c : float
        Coefficient of r12^2 in p(s) = 1 + s/2 + c s^2, 0 <= c <= 0.5.
    r : float
        Distance from the trap centre in bohr, 0 <= r <= 5 omega^(-1/2).
    occ_min : float
        Occupation threshold, 1e-9 <= occ_min <= 0.1; no occupation of the pair function lies within 10 percent of it.

    Returns
    -------
    result : np.ndarray
        Float array [phi_N(r), N, K] where N is the number of retained natural orbitals counting degeneracy and K the number
        of retained s-type orbitals. phi_N(r) is accurate to 1e-6 relative.

    Raises
    ------
    ValueError
        If occ_min lies outside its range.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_truncated_reduced_ontop(omega: float, c: float, r: float, occ_min: float) -> "np.ndarray":
    """Reference implementation."""
    occ_min = float(occ_min)
    if not 1e-9 <= occ_min <= 0.1:
        raise ValueError("occ_min out of range")
    first = second = norm = 0.0
    count = 0
    s_count = 0
    for l in range(13):
        table = _oracle_shell_orbital_densities(omega, c, l, 80, r)
        keep = table[:, 0] ** 2 >= occ_min
        if not keep.any():
            break
        first += np.sum(table[keep, 0] * table[keep, 1])
        second += np.sum(table[keep, 0] ** 2 * table[keep, 1])
        norm += (2 * l + 1) * np.sum(table[keep, 0] ** 2)
        count += (2 * l + 1) * int(keep.sum())
        if l == 0:
            s_count = int(keep.sum())
    return np.array([(first / second) ** 2 * norm, float(count), float(s_count)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: exact omega = 1/10 ground state near the density maximum ---
        {
            "setup": "import numpy as np\ndef parts(values):\n    v = np.asarray(values, dtype=float).reshape(3)\n    return np.array([np.log(v[0]), v[1], v[2]])\n",
            "call": "parts(truncated_reduced_ontop(0.1, 0.05, 3.96, 1e-7))",
            "gold_call": "parts(_oracle_truncated_reduced_ontop(0.1, 0.05, 3.96, 1e-7))",
            "tol": 1e-6,
        },
        # --- Normal: exact omega = 1/2 ground state at the centre ---
        {
            "setup": "import numpy as np\ndef parts(values):\n    v = np.asarray(values, dtype=float).reshape(3)\n    return np.array([np.log(v[0]), v[1], v[2]])\n",
            "call": "parts(truncated_reduced_ontop(0.5, 0.0, 0.0, 1e-6))",
            "gold_call": "parts(_oracle_truncated_reduced_ontop(0.5, 0.0, 0.0, 1e-6))",
            "tol": 1e-6,
        },
        # --- Boundary: a coarse threshold that keeps a handful of shells, so the estimate is far from one ---
        {
            "setup": "import numpy as np\ndef parts(values):\n    v = np.asarray(values, dtype=float).reshape(3)\n    return np.array([np.log(v[0]), v[1], v[2]])\n",
            "call": "parts(truncated_reduced_ontop(0.1, 0.05, 2.0, 0.001))",
            "gold_call": "parts(_oracle_truncated_reduced_ontop(0.1, 0.05, 2.0, 0.001))",
            "tol": 1e-6,
        },
        # --- Edge: tight trap, deep truncation, point in the density tail ---
        {
            "setup": "import numpy as np\ndef parts(values):\n    v = np.asarray(values, dtype=float).reshape(3)\n    return np.array([np.log(v[0]), v[1], v[2]])\n",
            "call": "parts(truncated_reduced_ontop(3.0, 0.1, 1.2, 5e-8))",
            "gold_call": "parts(_oracle_truncated_reduced_ontop(3.0, 0.1, 1.2, 5e-8))",
            "tol": 1e-6,
        },
        # --- Error: an occupation threshold above 0.1 must raise ValueError ---
        {
            "setup": "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(0.1, 0.05, 2.0, 0.5)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    return 0\n",
            "call": "_probe(truncated_reduced_ontop)",
            "gold_call": "_probe(_oracle_truncated_reduced_ontop)",
        },
    ]
