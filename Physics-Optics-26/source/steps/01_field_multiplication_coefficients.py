"""
Compute the multiplication coefficients of the lossy thermal cavity's damping eigenoperator of radial index $n$ and coherence order $k$: the coefficients of the eigenoperators produced when it is multiplied from the left or from the right by $a$ or $a^\dagger$.

The field damping basis diagonalizes the thermal cavity Liouvillian, labelling its right eigenoperators by a radial index $n$ and a coherence order $k$. Multiplying an eigenoperator by $a$ or $a^\dagger$ changes $k$ by one and $n$ by at most one, with coefficients that depend on the sign of $k$, on the thermal occupation of the reservoir, and on whether the multiplication acts from the left or from the right. These coefficients are fixed by the paper's own normalization of the eigenoperators; consult the paper's appendix on field multiplication coefficients rather than choosing a normalization of your own.

Returns
-------
np.ndarray of shape $(2, 2, 3)$: $f[\mathrm{side}, \mu, d]$, the left/right, $a$/$a^\dagger$ multiplication coefficients onto radial levels $n-1$, $n$, $n+1$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def field_multiplication_coefficients(n: int, k: int, nu: float) -> "np.ndarray":
    r"""Briegel-Englert multiplication coefficients of the field damping eigenoperator $\rho_n^{(k)}$.

    Args:
        n (int): radial damping index, $n\ge0$.
        k (int): coherence order $k$ (any integer: positive, zero or negative).
        nu (float): thermal occupation $\nu\ge0$ of the cavity reservoir.

    Raises:
        ValueError: if $n$ is negative or not an integer, if $k$ is not an integer, or if $\nu<0$.

    Expected return:
        np.ndarray of shape $(2,2,3)$, $f[\mathrm{side},\mu,d]$: $\mathrm{side}=0$ is left
        multiplication and $\mathrm{side}=1$ right multiplication; $\mu=0$ is the annihilation
        operator $a$ (target coherence order $k-1$) and $\mu=1$ the creation operator $a^\dagger$
        (target coherence order $k+1$); $d=0,1,2$ for target radial index $n-1$, $n$, $n+1$.
        Entries that do not occur are zero, every $d=0$ entry vanishes at $n=0$, and all entries
        are non-negative.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: field_multiplication_coefficients
def _oracle_field_multiplication_coefficients(n: int, k: int, nu: float) -> "np.ndarray":
    # ORACLE (hidden). Appendix A, Eqs. (A1)-(A12) of arXiv:2609.17785.
    # f[side, mu, d]: side 0 = left multiplication (a_mu rho), 1 = right (rho a_mu);
    # mu 0 = a (lowers k), 1 = a^dagger (raises k); d = delta + 1 for target radial index n + delta.
    if int(n) != n or n < 0:
        raise ValueError("radial index n must be a non-negative integer")
    if int(k) != k:
        raise ValueError("coherence order k must be an integer")
    if nu < 0:
        raise ValueError("thermal occupation nu must be non-negative")
    n = int(n); k = int(k)
    f = np.zeros((2, 2, 3), dtype=float)
    p = 1.0 + nu
    c = (n + 1) * nu / (1.0 + nu)
    ak = abs(k)
    if k > 0:
        f[0, 0, 1] = n + k; f[0, 0, 2] = c          # (A1)
        f[0, 1, 0] = p;     f[0, 1, 1] = p          # (A2)
        f[1, 0, 1] = n + k; f[1, 0, 2] = n + 1      # (A3)
        f[1, 1, 0] = p;     f[1, 1, 1] = nu         # (A4)
    elif k < 0:
        f[0, 0, 0] = p;      f[0, 0, 1] = nu        # (A5)
        f[0, 1, 1] = n + ak; f[0, 1, 2] = n + 1     # (A6)
        f[1, 0, 0] = p;      f[1, 0, 1] = p         # (A7)
        f[1, 1, 1] = n + ak; f[1, 1, 2] = c         # (A8)
    else:
        f[0, 0, 0] = p; f[0, 0, 1] = nu             # (A9)
        f[0, 1, 0] = p; f[0, 1, 1] = p              # (A10)
        f[1, 0, 0] = p; f[1, 0, 1] = p              # (A11)
        f[1, 1, 0] = p; f[1, 1, 1] = nu             # (A12)
    if n == 0:
        f[:, :, 0] = 0.0                            # rho_{-1}^{(k)} = 0
    return f

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    return [
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(field_multiplication_coefficients(3, 2, 0.3))",
            "gold_call": "as_real(_oracle_field_multiplication_coefficients(3, 2, 0.3))",
            "tol": 1e-12,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(field_multiplication_coefficients(2, -3, 0.25))",
            "gold_call": "as_real(_oracle_field_multiplication_coefficients(2, -3, 0.25))",
            "tol": 1e-12,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(field_multiplication_coefficients(0, 0, 0.1))",
            "gold_call": "as_real(_oracle_field_multiplication_coefficients(0, 0, 0.1))",
            "tol": 1e-12,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(field_multiplication_coefficients(0, 1, 0.0))",
            "gold_call": "as_real(_oracle_field_multiplication_coefficients(0, 1, 0.0))",
            "tol": 1e-12,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(field_multiplication_coefficients(7, -1, 0.05))",
            "gold_call": "as_real(_oracle_field_multiplication_coefficients(7, -1, 0.05))",
            "tol": 1e-12,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(field_multiplication_coefficients(4, 0, 0.3))",
            "gold_call": "as_real(_oracle_field_multiplication_coefficients(4, 0, 0.3))",
            "tol": 1e-12,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        field_multiplication_coefficients(-1, 0, 0.1)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_field_multiplication_coefficients(-1, 0, 0.1)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
