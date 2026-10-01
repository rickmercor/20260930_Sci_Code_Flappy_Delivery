"""
Evaluate determinant derivatives without assuming an invertible base matrix.

The determinant $D(x)=\det M(x)$ is multilinear in the rows of $M(x)$.

Its ordinary derivatives $D^{(r)}(x)$ include contributions in which derivative

orders are distributed among different rows. These contributions remain

well-defined when $\det M(x)=0$.

Returns
-------
A complex NumPy array of shape $(S,)$ for value samples, or $(J,S)$ for $J$ ordinary derivative channels.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_determinant_samples(matrix_samples: np.ndarray) -> np.ndarray:
    r"""Return sampled determinants and optional ordinary derivative channels.
    
    Parameters
    ----------
    matrix_samples : np.ndarray
        Finite real or complex values of shape $(S,N,N)$, or ordinary
        matrix derivatives of shape $(J,S,N,N)$, with $S\ge1$ and
        $1\le J\le4$. In the latter form, entry $(r,j)$ is $M^{(r)}(x_j)$.
        The matrices must be square; singular matrices and $N=0$ are valid.
    
    Returns
    -------
    determinant_samples : np.ndarray
        Complex vector of shape $(S,)$ for value-only input, or array of
        shape $(J,S)$ with entry $(r,j)$ equal to $D^{(r)}(x_j)$ for
        derivative input. Here $D(x)=\det M(x)$. For $N=0$, the value
        channel is one and all higher derivative channels are zero.
    
    Raises
    ------
    ValueError
        If the input rank is neither three nor four, the sample axis is
        empty, matrices are nonsquare, the derivative-channel count is
        outside $[1,4]$, or any entry is nonfinite.
    
    Notes
    -----
    Determinant differentiation must include mixed row-derivative terms.
    The definition applies at singular matrices of any rank and requires no
    inverse of the base matrix. Unprovided higher matrix derivatives do not
    affect the requested orders. Do not mutate the input array.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _pivoted_determinant(matrix):
    work = np.array(matrix, dtype=complex, copy=True)
    determinant = 1.0 + 0.0j
    for column in range(work.shape[0]):
        pivot = column + int(np.argmax(np.abs(work[column:, column])))
        if abs(work[pivot, column]) <= np.finfo(float).tiny:
            return 0.0 + 0.0j
        if pivot != column:
            work[[column, pivot]] = work[[pivot, column]]
            determinant = -determinant
        value = work[column, column]
        determinant *= value
        if column+1 < work.shape[0]:
            factors = work[column+1:, column]/value
            work[column+1:, column+1:] -= np.outer(factors, work[column, column+1:])
    return determinant


def _oracle_compute_determinant_samples(matrix_samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(matrix_samples)
    if samples.ndim not in (3,4):
        raise ValueError("input rank must be three or four")
    scalar = samples.ndim == 3
    jets = samples[None] if scalar else samples
    if not 1 <= jets.shape[0] <= 4 or jets.shape[1] < 1 or jets.shape[2] != jets.shape[3]:
        raise ValueError("invalid jet or matrix shape")
    if not np.all(np.isfinite(jets)):
        raise ValueError("matrix data must be finite")
    if scalar:
        return np.array([_pivoted_determinant(m) for m in samples])
    channels, count, size, _ = jets.shape
    factorials = np.array([1,1,2,6][:channels], dtype=float)
    coefficients = jets/factorials[:,None,None,None]
    out = np.zeros((channels,count), dtype=complex)
    # Subset expansion over columns; products are truncated Taylor series.
    # The sign counts inversions contributed by the newly chosen column.
    for sample in range(count):
        dp = np.zeros((1 << size,channels), dtype=complex)
        dp[0,0] = 1
        for mask in range((1 << size)-1):
            row = mask.bit_count()
            for column in range(size):
                if mask & (1 << column):
                    continue
                sign = -1 if (mask >> (column+1)).bit_count() % 2 else 1
                product = np.convolve(dp[mask], coefficients[:,sample,row,column])[:channels]
                dp[mask | (1 << column)] += sign*product
        out[:,sample] = dp[-1]*factorials
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Scientific cases with independent mutable inputs on both sides."""
    return [{'setup': 'import numpy as np\nmatrix_samples=np.array([[[1,2],[3,4]],[[1+1j,0],[2,3-1j]]])',
      'call': 'compute_determinant_samples(matrix_samples.copy())',
      'gold_call': '_oracle_compute_determinant_samples(matrix_samples.copy())'},
     {'setup': 'import numpy as np\nmatrix_samples=np.array([[[1.,2.],[2.,4.]]])',
      'call': 'compute_determinant_samples(matrix_samples.copy())',
      'gold_call': '_oracle_compute_determinant_samples(matrix_samples.copy())'},
     {'setup': 'import numpy as np\n'
               'matrix_samples = np.ones((2, 3, 2))\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        compute_determinant_samples(matrix_samples.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_compute_determinant_samples(matrix_samples.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'J=np.zeros((4,3,3,3));J[0,0]=np.diag([0.,2.,3.]);J[0,1]=np.diag([0.,0.,3.]);J[1]=np.eye(3)[None];J[2,0]=np.diag([2.,-1.,.4]);J[3,0]=np.ones((3,3))',
      'call': 'compute_determinant_samples(J.copy())',
      'gold_call': '_oracle_compute_determinant_samples(J.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(42);J=rng.normal(size=(4,4,5,5))+1j*rng.normal(size=(4,4,5,5));J[0,1,:,0]=0;J[0,2,:,:2]=0;J[0,3,:,:3]=0',
      'call': 'compute_determinant_samples(J.copy())',
      'gold_call': '_oracle_compute_determinant_samples(J.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(9);L=np.eye(7);L[1:,0]=rng.normal(size=6);R=np.eye(7);R[0,1:]=rng.normal(size=6);J=np.zeros((4,1,7,7));J[0,0]=L@np.diag([0,0,0,1,2,3,4])@R;J[1,0]=L@np.diag([2,3,5,0,0,0,0])@R',
      'call': 'compute_determinant_samples(J.copy())',
      'gold_call': '_oracle_compute_determinant_samples(J.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nJ=np.zeros((4,2,0,0))',
      'call': 'compute_determinant_samples(J.copy())',
      'gold_call': '_oracle_compute_determinant_samples(J.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nJ=np.array([[[[1.,2.],[3.,4.]]]])',
      'call': 'compute_determinant_samples(J.copy())',
      'gold_call': '_oracle_compute_determinant_samples(J.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'J = np.zeros((5, 2, 3, 3))\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        compute_determinant_samples(J.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_compute_determinant_samples(J.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()',
      'tol': 0}]
