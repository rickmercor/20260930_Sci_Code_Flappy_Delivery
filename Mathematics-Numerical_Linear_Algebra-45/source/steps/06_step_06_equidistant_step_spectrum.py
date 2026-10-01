"""
Build the eigenvalue list of a normalised test problem with a prescribed step location and a prescribed gap around it.

A test spectrum for the matrix step function is fixed by the step location mu and the gap width, which place the two interior eigenvalues at mu - gap/2 and mu + gap/2. The number of eigenvalues above the step is the nearest integer to n*(1 - mu), and those eigenvalues are spread equidistantly from the upper interior eigenvalue to 1 while the remainder are spread equidistantly from 0 to the lower interior eigenvalue.

Returns
-------
np.ndarray, float, shape (n,): the ascending eigenvalues of the test problem.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def equidistant_step_spectrum(size: int, mu: float, gap: float) -> np.ndarray:
    """Build the eigenvalue list of the normalised test problem.

    Parameters
    ----------
    size : int
        Number of eigenvalues n, an integer with n >= 2.
    mu : float
        Step location, with 0 < mu < 1.
    gap : float
        Width of the gap around the step, with 0 < gap and the resulting
        interior eigenvalues strictly inside (0, 1).

    Returns
    -------
    eigenvalues : np.ndarray
        Shape (n,) float array of eigenvalues in ascending order, holding the
        block below the step followed by the block above it.

    Raises
    ------
    ValueError
        If ``size`` is not an integer (a bool counts as not an integer) or
        is less than 2; if ``mu`` or ``gap`` is not a real finite number; if
        ``mu`` does not satisfy 0 < mu < 1; if ``gap`` is not positive; if
        the resulting interior eigenvalues do not lie strictly inside
        (0, 1); or if either block would hold fewer than two eigenvalues.
        The function must raise rather than return a placeholder or clip the
        spectrum back into the unit interval.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(int(size), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_equidistant_step_spectrum(size: int, mu: float, gap: float) -> np.ndarray:
    import numpy as np

    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise ValueError("size must be an integer")
    size = int(size)
    if size < 2:
        raise ValueError("size must be at least 2")
    for name, val in (("mu", mu), ("gap", gap)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)):
            raise ValueError(f"{name} must be finite")
    mu = float(mu)
    gap = float(gap)
    if not 0.0 < mu < 1.0:
        raise ValueError("mu must satisfy 0 < mu < 1")
    if gap <= 0.0:
        raise ValueError("gap must be positive")

    lam_lumo = mu - 0.5 * gap
    lam_homo = mu + 0.5 * gap
    if not 0.0 < lam_lumo < lam_homo < 1.0:
        raise ValueError("the gap must lie strictly inside the unit interval")

    n_occ = int(round(size * (1.0 - mu)))
    if n_occ < 2 or size - n_occ < 2:
        raise ValueError("both blocks must hold at least two eigenvalues")

    lower = np.linspace(0.0, lam_lumo, size - n_occ)
    upper = np.linspace(lam_homo, 1.0, n_occ)
    return np.concatenate([lower, upper])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values obtained independently of the oracle. With n = 10
        # and mu = 0.4 the upper block holds 6 eigenvalues and the lower 4, and
        # both blocks are arithmetic progressions whose sums are available in
        # closed form: 4*lam_lumo/2 for the lower block and 6*(lam_homo + 1)/2
        # for the upper one. Probing the sum and the alternating sum separates a
        # wrong block size from a wrong endpoint.
        {
            "setup": """import numpy as np
n, mu, gap = 10, 0.4, 0.05
lo, hi = mu - gap / 2, mu + gap / 2
exact = np.concatenate([np.linspace(0.0, lo, 4), np.linspace(hi, 1.0, 6)])
w = np.array([(-1.0) ** j * (j + 1) for j in range(10)])
EXPECTED = float(np.sum(exact) + 1000.0 * np.dot(w, exact))
""",
            "call": ("float(np.sum(equidistant_step_spectrum(n, mu, gap))"
                     " + 1000.0 * np.dot(w, equidistant_step_spectrum(n, mu, gap)))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: a step below the middle of the interval, so the upper block
        # is the larger of the two (normal scenario) ---
        {
            "setup": """import numpy as np
n, mu, gap = 96, 0.27, 8e-4
w = np.cos(np.arange(96) * 0.31)
""",
            "call": "float(np.dot(w, equidistant_step_spectrum(n, mu, gap)))",
            "gold_call": "float(np.dot(w, _oracle_equidistant_step_spectrum(n, mu, gap)))",
        },
        # --- Valid: a step above the middle of the interval ---
        {
            "setup": """import numpy as np
n, mu, gap = 84, 0.73, 2e-3
w = np.sin(np.arange(84) * 0.19 + 0.4)
""",
            "call": "float(np.dot(w, equidistant_step_spectrum(n, mu, gap)))",
            "gold_call": "float(np.dot(w, _oracle_equidistant_step_spectrum(n, mu, gap)))",
        },
        # --- Boundary: a very narrow gap, where the two interior eigenvalues are
        # nearly coincident and the block sizes must still come out right ---
        {
            "setup": """import numpy as np
n, mu, gap = 60, 0.5, 1e-9
""",
            "call": ("float(equidistant_step_spectrum(n, mu, gap)[29]"
                     " + equidistant_step_spectrum(n, mu, gap)[30]"
                     " + 1000.0 * equidistant_step_spectrum(n, mu, gap)[0])"),
            "gold_call": ("float(_oracle_equidistant_step_spectrum(n, mu, gap)[29]"
                          " + _oracle_equidistant_step_spectrum(n, mu, gap)[30]"
                          " + 1000.0 * _oracle_equidistant_step_spectrum(n, mu, gap)[0])"),
        },
        # --- Edge: the block sizes must follow the rounding of n*(1 - mu), which
        # here is not an integer ---
        {
            "setup": """import numpy as np
n, mu, gap = 37, 0.41, 5e-3
""",
            "call": ("float(np.sum(equidistant_step_spectrum(n, mu, gap) < 0.41)"
                     " + 1000.0 * np.sum(equidistant_step_spectrum(n, mu, gap)))"),
            "gold_call": ("float(np.sum(_oracle_equidistant_step_spectrum(n, mu, gap) < 0.41)"
                          " + 1000.0 * np.sum(_oracle_equidistant_step_spectrum(n, mu, gap)))"),
        },
        # --- Invalid: the gap runs past the end of the unit interval ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        equidistant_step_spectrum(40, 0.02, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_equidistant_step_spectrum(40, 0.02, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive gap ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        equidistant_step_spectrum(40, 0.5, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_equidistant_step_spectrum(40, 0.5, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
