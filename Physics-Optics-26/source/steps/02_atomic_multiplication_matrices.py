"""
Compute the matrices of left and right multiplication by the collective ladders $S_+$ and $S_-$ on the paper's permutation-invariant atomic damping basis $R_m$ for $N$ emitters at pump parameter $s$.

For $N$ identical, incoherently pumped emitters the paper builds a permutation-invariant operator basis $R_m$ labelled by compositions $m=(m_0, m_z, m_+, m_-)$ of $N$, each $R_m$ being a normalized symmetrization of tensor products of single-atom damping eigenoperators. The collective ladders map this basis into itself with pump-dependent coefficients. Use the paper's own single-atom operators, its normalization of $R_m$ and its multiplication rules; consult the paper for their exact form. The basis ordering stated in the return description is a bookkeeping convention of this task, not physics.

Returns
-------
np.ndarray of shape $(4, D, D)$: the $S_+$-left, $S_-$-left, $S_+$-right and $S_-$-right multiplication matrices in the stated composition ordering.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def atomic_multiplication_matrices(N: int, s: float) -> "np.ndarray":
    r"""Collective-ladder multiplication matrices on the permutation-invariant atomic damping basis.

    Args:
        N (int): number of emitters, $N\ge1$.
        s (float): pump parameter, $0<s<1$.

    Raises:
        ValueError: if $N$ is not a positive integer or $s$ is not strictly inside $(0,1)$.

    Expected return:
        np.ndarray of shape $(4,D,D)$, $D$ the basis dimension. Slice $t$ is the matrix of
        $t=0$: $S_+$ from the left, $t=1$: $S_-$ from the left, $t=2$: $S_+$ from the right,
        $t=3$: $S_-$ from the right; element $[t,\alpha,\beta]$ is the coefficient of $R_\alpha$
        in the product formed from $R_\beta$ (columns = source, rows = target). Basis ordering:
        compositions $(m_0,m_z,m_+,m_-)$ of $N$ with $m_0$ descending from $N$ to $0$, then $m_z$
        descending, then $m_+$ descending, so index $0$ is $(N,0,0,0)$. Each column has at most
        four non-zero entries and every non-zero entry changes the coherence charge $m_+-m_-$ by
        exactly one unit.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: atomic_multiplication_matrices
def _oracle_atomic_multiplication_matrices(N: int, s: float) -> "np.ndarray":
    # ORACLE (hidden). Eqs. (4), (6)-(9) of arXiv:2609.17785.
    # Output A[t] with t = 0: S_+ R (left), 1: S_- R (left), 2: R S_+ (right), 3: R S_- (right);
    # (A[t])_{alpha beta} = coefficient of R_alpha in the product acting on R_beta.
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if not (0.0 < s < 1.0):
        raise ValueError("pump parameter s must lie strictly between 0 and 1")
    N = int(N)
    comp = []
    for m0 in range(N, -1, -1):
        for mz in range(N - m0, -1, -1):
            for mp in range(N - m0 - mz, -1, -1):
                comp.append((m0, mz, mp, N - m0 - mz - mp))
    idx = {c: i for i, c in enumerate(comp)}
    D = len(comp)
    A = np.zeros((4, D, D), dtype=float)

    def add(t, b, m, coef):
        if min(m) < 0 or coef == 0:
            return
        A[t, idx[m], b] += coef

    for b, (m0, mz, mp, mm) in enumerate(comp):
        # (6)  S_+ R_m
        add(0, b, (m0 - 1, mz, mp + 1, mm), (1 - s) * m0)
        add(0, b, (m0, mz - 1, mp + 1, mm), -mz)
        add(0, b, (m0 + 1, mz, mp, mm - 1), mm)
        add(0, b, (m0, mz + 1, mp, mm - 1), (1 - s) * mm)
        # (7)  S_- R_m
        add(1, b, (m0 - 1, mz, mp, mm + 1), s * m0)
        add(1, b, (m0, mz - 1, mp, mm + 1), mz)
        add(1, b, (m0 + 1, mz, mp - 1, mm), mp)
        add(1, b, (m0, mz + 1, mp - 1, mm), -s * mp)
        # (8)  R_m S_+
        add(2, b, (m0 - 1, mz, mp + 1, mm), s * m0)
        add(2, b, (m0, mz - 1, mp + 1, mm), mz)
        add(2, b, (m0 + 1, mz, mp, mm - 1), mm)
        add(2, b, (m0, mz + 1, mp, mm - 1), -s * mm)
        # (9)  R_m S_-
        add(3, b, (m0 - 1, mz, mp, mm + 1), (1 - s) * m0)
        add(3, b, (m0, mz - 1, mp, mm + 1), -mz)
        add(3, b, (m0 + 1, mz, mp - 1, mm), mp)
        add(3, b, (m0, mz + 1, mp - 1, mm), (1 - s) * mp)
    return A

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    return [
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(atomic_multiplication_matrices(1, 0.3))",
            "gold_call": "as_real(_oracle_atomic_multiplication_matrices(1, 0.3))",
            "tol": 1e-12,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(atomic_multiplication_matrices(2, 0.77))",
            "gold_call": "as_real(_oracle_atomic_multiplication_matrices(2, 0.77))",
            "tol": 1e-12,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(atomic_multiplication_matrices(3, 0.6))",
            "gold_call": "as_real(_oracle_atomic_multiplication_matrices(3, 0.6))",
            "tol": 1e-12,
        },
        {
            "setup": 'import numpy as np\ndef as_real(x):\n    return np.asarray(x, dtype=float)',
            "call": "as_real(atomic_multiplication_matrices(4, 0.88))",
            "gold_call": "as_real(_oracle_atomic_multiplication_matrices(4, 0.88))",
            "tol": 1e-12,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        atomic_multiplication_matrices(3, 1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_atomic_multiplication_matrices(3, 1.0)\n"
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
