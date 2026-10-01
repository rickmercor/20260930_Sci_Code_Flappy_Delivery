"""
Construct the controlled instrument of a finite quantum renewal clock in the specified tail-state gauge.

The age-$j$ memory is $|s_j\rangle=(N-j)^{-1/2}\sum_{k=j}^{N-1}|k\rangle$. For stimulus $x=0$, no tick has amplitude $\sqrt{(N-j-1)/(N-j)}$ and advances to $|s_{j+1}\rangle$, whereas a tick has amplitude $(N-j)^{-1/2}$ and resets to $|s_0\rangle$; both use environment label $\eta=0$. The no-tick term vanishes at $j=N-1$, with no periodic wrap. Stimulus $x=1$ sends $|s_j\rangle$ to $|s_0\rangle\otimes|y=0\rangle\otimes|s_j\rangle_E$. Recover the linear Kraus maps from these actions on a nonorthogonal spanning family, retaining the fixed computational environment basis.

Returns
-------
np.ndarray, complex Kraus array of shape (2, 2, n, n, n).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral

import numpy as np

def build_clock_instrument(n: int) -> np.ndarray:
    """Construct the controlled renewal-clock Kraus operators.

    Parameters
    ----------
    n : int
        Number of ages and memory dimension, at least 2. Boolean values are
        not integers for this contract. All indices start at zero.

    Returns
    -------
    kraus : np.ndarray
        Complex array of shape (2, 2, n, n, n), indexed by stimulus, action,
        environment, outgoing memory, incoming memory. Action 1 is a tick.
        The memory and environment gauges are the tail states described above.

    Raises
    ------
    ValueError
        If n is not an integer at least 2.
    """
    return np.zeros((2, 2, n, n, n), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_clock_instrument(n: int) -> "np.ndarray":
    import numpy as np
    from numbers import Integral

    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 2:
        raise ValueError("n must be an integer at least 2")
    n = int(n)
    kraus = np.zeros((2, 2, n, n, n), dtype=complex)
    reset = np.ones(n) / np.sqrt(n)
    kraus[0, 0, 0] = np.diag(np.ones(n - 1), -1)
    kraus[0, 1, 0, :, -1] = reset
    for eta in range(n):
        kraus[1, 0, eta, :, eta] = reset
    return kraus

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic case specifications."""
    return [
        {
            "setup": """""",
            "call": 'build_clock_instrument(6)',
            "gold_call": '_oracle_build_clock_instrument(6)',
        },
        {
            "setup": """""",
            "call": 'build_clock_instrument(2)',
            "gold_call": '_oracle_build_clock_instrument(2)',
        },
        {
            "setup": """""",
            "call": 'build_clock_instrument(9)',
            "gold_call": '_oracle_build_clock_instrument(9)',
        },
        {
            "setup": """

def _run_model():
    try:
        build_clock_instrument(1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_build_clock_instrument(1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_run_model()',
            "gold_call": '_run_gold()',
        },
        {
            "setup": """

def _run_model():
    try:
        build_clock_instrument(3.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_build_clock_instrument(3.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_run_model()',
            "gold_call": '_run_gold()',
        },
        {
            "setup": """

def _run_model():
    try:
        build_clock_instrument(True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_build_clock_instrument(True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_run_model()',
            "gold_call": '_run_gold()',
        },
    ]
