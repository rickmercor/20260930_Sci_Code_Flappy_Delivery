"""
Evaluate the exact random-orthonormal residual-inflation benchmark.

Random orthonormal embeddings have a sharp finite-dimensional expected error law for sketch-and-solve least squares. This supplies a data-oblivious reference whose value depends only on the ambient dimension, rank, embedding dimension, and field.

Returns
-------
float, exact finite expected residual-inflation factor.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def random_orthonormal_benchmark(n: int, r: int, ell: int, field: str = "real") -> float:
    r"""Evaluate the exact random-orthonormal residual-inflation benchmark.

    Parameters
    ----------
    n : int
        Positive ambient dimension.
    r : int
        Positive matrix rank satisfying $r<n$.
    ell : int
        Positive embedding dimension satisfying $\ell\le n$. When
        $\ell=n$ the embedding is square, the factor is exactly $1$, and no
        further condition is required; otherwise the finite-expectation
        condition $r<\ell-\alpha$ must hold, with $\alpha=1$ for "real"
        and $\alpha=0$ for "complex".
    field : str, optional
        Scalar field, either "real" or "complex". The default is "real".

    Returns
    -------
    rho : float
        Exact finite expected residual-inflation factor
        $1+\frac{n-\ell}{n-r}\,\frac{r}{\ell-r-\alpha}$ (equal to $1$ when $\ell=n$).

    Raises
    ------
    ValueError
        If n, r, or ell is not an admissible integer, if field is not
        "real" or "complex", or if $\ell<n$ and the expectation is not
        finite for the supplied dimensions.
    """
    return rho

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_random_orthonormal_benchmark(n, r, ell, field="real"):
    if not all(isinstance(x, (int, np.integer)) for x in (n, r, ell)):
        raise ValueError("dimensions must be integers")
    n, r, ell = int(n), int(r), int(ell)
    if n < 1 or r < 1 or ell < 1 or r >= n or ell > n:
        raise ValueError("invalid dimensions")
    if field not in ("real", "complex"):
        raise ValueError("invalid field")
    if ell == n:
        return 1.0
    alpha = 1 if field == "real" else 0
    if r >= ell - alpha:
        raise ValueError("expectation is not finite")
    return float(1.0 + ((n - ell) / (n - r)) * (r / (ell - r - alpha)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"import numpy as np\nn=14; r=5; ell=7; field='real'",'call':'random_orthonormal_benchmark(n,r,ell,field)','gold_call':'_oracle_random_orthonormal_benchmark(n,r,ell,field)'},
        {'setup':"import numpy as np\nn=14; r=5; ell=7; field='complex'",'call':'random_orthonormal_benchmark(n,r,ell,field)','gold_call':'_oracle_random_orthonormal_benchmark(n,r,ell,field)'},
        {'setup':"import numpy as np\nn=8; r=2; ell=8; field='real'",'call':'random_orthonormal_benchmark(n,r,ell,field)','gold_call':'_oracle_random_orthonormal_benchmark(n,r,ell,field)'},
        {'setup':"import numpy as np\nn=6; r=5; ell=6; field='real'",'call':'random_orthonormal_benchmark(n,r,ell,field)','gold_call':'_oracle_random_orthonormal_benchmark(n,r,ell,field)'},
        {'setup':"""import numpy as np
n=10; r=3; ell=4; field='real'
def run_model():
    try: random_orthonormal_benchmark(n,r,ell,field); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_random_orthonormal_benchmark(n,r,ell,field); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
n=10; r=3; ell=5; field='other'
def run_model():
    try: random_orthonormal_benchmark(n,r,ell,field); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_random_orthonormal_benchmark(n,r,ell,field); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
n=5; r=5; ell=5; field='real'
def run_model():
    try: random_orthonormal_benchmark(n,r,ell,field); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_random_orthonormal_benchmark(n,r,ell,field); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
