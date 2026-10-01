"""
Assembles the linear state matrix that governs the joint evolution of the lifted variance factors from the mean-reversion strength, the lift weights, and the mean-reversion speeds.

State matrix of the lifted factor dynamics.

Each lifted factor decays toward its own long-run level at a speed set by the refinement ratio, and all factors are pulled by the same instantaneous variance, which is the weighted sum of the factors. Written in vector form, the drift of the factor vector is therefore a diagonal decay term plus a rank-one contribution proportional to the outer product of the all-ones vector with the lift weights. The resulting state matrix is not symmetric, so its spectral decomposition must be computed with a general eigen-decomposition rather than a symmetric one. This matrix is the object every later conditional-moment and simulation step exponentiates or inverts when it propagates the factor state across a time step, and its fixed structure is what makes the large-step conditional moments tractable in closed form.

Returns
-------
A : (N, N) float array -- state matrix -lam * 1_N omega - diag(x) of the lifted system (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def state_matrix(lam: float, omega: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Assemble the state matrix of the lifted variance dynamics.

    Parameters
    ----------
    lam : float
        Mean-reversion strength of the variance in the lifted model (must be
        a real finite number >= 0).
    omega : (N,) float array
        Lift weights of the N factors.
    x : (N,) float array
        Mean-reversion speeds of the N factors.

    Returns
    -------
    A : (N, N) float array
        State matrix -lam * 1_N omega - diag(x) of the lifted system.

    Raises
    ------
    ValueError
        If lam is not a real finite number >= 0, or if omega and x are not
        one-dimensional arrays of the same length >= 1.
    """
    return A

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math


def _oracle_state_matrix(lam: float, omega: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Reference implementation of state_matrix."""
    if not isinstance(lam, (int, float, np.integer, np.floating)) or isinstance(lam, bool):
        raise ValueError("lam must be a real number >= 0")
    lam = float(lam)
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be a real finite number >= 0")
    om = np.asarray(omega, dtype=float)
    xs = np.asarray(x, dtype=float)
    if om.ndim != 1 or xs.ndim != 1:
        raise ValueError("omega and x must be one-dimensional arrays")
    if om.shape[0] != xs.shape[0] or om.shape[0] < 1:
        raise ValueError("omega and x must be one-dimensional arrays of the same length >= 1")
    n = xs.shape[0]
    return -lam * np.outer(np.ones(n), om) - np.diag(xs)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 4 cases: the pinned five-factor configuration, a single-factor
    boundary, and two invalid-input edges (dimension mismatch, negative
    mean-reversion strength).
    """
    pinned = (
        "import numpy as np\n"
        "import math\n"
        "H = 0.3\n"
        "N = 5\n"
        "rN = 1.0 + 10.0 * N ** (-0.9)\n"
        "n = np.arange(1, N + 1)\n"
        "omega = ((rN ** (0.5 - H) - 1.0) * rN ** ((H - 0.5) * (1.0 + 0.5 * N)) / (math.gamma(H + 0.5) * math.gamma(1.5 - H)) * rN ** ((0.5 - H) * n))\n"
        "x = ((0.5 - H) / (1.5 - H) * (rN ** (1.5 - H) - 1.0) / (rN ** (0.5 - H) - 1.0) * rN ** (n - 1.0 - N / 2.0))\n"
        "lam = 0.25"
    )
    guard = (
        "\n"
        "def _guard(thunk):\n"
        "    try:\n"
        "        thunk()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 2\n"
        "    except Exception:\n"
        "        return 1"
    )
    return [
        {
            # pinned benchmark configuration: trace of the rank-one state matrix
            "setup": pinned,
            "call": "float(np.trace(state_matrix(lam, omega, x)))",
            "gold_call": "float(np.trace(_oracle_state_matrix(lam, omega, x)))",
        },
        {
            # pinned configuration: the rank-one diagonal entries
            "setup": pinned,
            "call": "float(np.max(np.diag(state_matrix(lam, omega, x)))) + float(np.min(np.diag(state_matrix(lam, omega, x))))",
            "gold_call": "float(np.max(np.diag(_oracle_state_matrix(lam, omega, x)))) + float(np.min(np.diag(_oracle_state_matrix(lam, omega, x))))",
        },
        {
            # boundary: a single lifted factor
            "setup": "import numpy as np\nomega = np.array([0.5])\nx = np.array([2.0])\nlam = 0.25",
            "call": "float(state_matrix(lam, omega, x)[0, 0])",
            "gold_call": "float(_oracle_state_matrix(lam, omega, x)[0, 0])",
        },
        {
            # edge: omega and x of different lengths
            "setup": "import numpy as np\nomega = np.array([0.1, 0.2, 0.3])\nx = np.array([1.0, 2.0])\nlam = 0.25" + guard,
            "call": "_guard(lambda: state_matrix(lam, omega, x))",
            "gold_call": "_guard(lambda: _oracle_state_matrix(lam, omega, x))",
        },
        {
            # edge: a negative mean-reversion strength is invalid
            "setup": "import numpy as np\nomega = np.array([0.5])\nx = np.array([2.0])\nlam = -0.25" + guard,
            "call": "_guard(lambda: state_matrix(lam, omega, x))",
            "gold_call": "_guard(lambda: _oracle_state_matrix(lam, omega, x))",
        },
    ]
