"""
Evaluate a polynomial matrix and its ordinary derivatives on a Fourier grid.

For $M(x)=\sum_{\ell=0}^{K-1}A_\ell x^\ell$, confluent sampling evaluates

$M^{(r)}(x_j)$ at $x_j=\exp(-2\pi i j/S)$. Ordinary differentiation shifts

polynomial powers. When $K>S$, coefficients with equal degree modulo $S$

contribute to the same Fourier component, so every degree must be included.

Returns
-------
A complex NumPy array of shape $(S,N,N)$ when $R=0$, or $(R+1,S,N,N)$ when $R>0$, where $R$ is `derivative_order`; channels contain ordinary derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_resultant_fft(
    coefficient_tensor: np.ndarray, sample_points: np.ndarray,
    derivative_order: int = 0,
) -> np.ndarray:
    r"""Evaluate $M$ and its ordinary derivatives with respect to $x$.
    
    Parameters
    ----------
    coefficient_tensor : np.ndarray
        Finite real or complex array of shape $(K,N,N)$ with $K\ge1$.
        Slice $n$ is the coefficient of $x^n$ in the square matrix $M(x)$.
    sample_points : np.ndarray
        Finite complex vector of shape $(S,)$ with $S\ge2$, ordered as
        $x_j=\exp(-2\pi i j/S)$ for $j=0,\ldots,S-1$.
        Each entry must match this grid within absolute tolerance $10^{-12}$.
    derivative_order : int, optional
        Highest ordinary derivative order $R$, default $0$, with $0\le R\le3$.
    
    Returns
    -------
    matrix_samples : np.ndarray
        Complex array of shape $(S,N,N)$ when $R=0$, or $(R+1,S,N,N)$
        when $R>0$. Channel $(r,j)$ is the ordinary derivative $M^{(r)}(x_j)$.
        Derivatives above the polynomial degree are zero. Every coefficient
        contributes even when $K>S$.
    
    Raises
    ------
    ValueError
        If the tensor is not rank three, its degree axis is empty, its
        matrices are nonsquare, points are not a vector of at least two
        entries, either input contains nonfinite values, $R$ is not an
        integer in $[0,3]$, or points fail the grid tolerance.
    
    Notes
    -----
    These channels contain ordinary derivatives, not Taylor coefficients.
    Use the polynomial-degree axis for evaluation, including aliased degrees
    when the coefficient axis is longer than the sampling grid.
    Do not mutate either input array.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_evaluate_resultant_fft(
    coefficient_tensor: np.ndarray, sample_points: np.ndarray,
    derivative_order: int = 0,
) -> np.ndarray:
    tensor = np.asarray(coefficient_tensor)
    points = np.asarray(sample_points)
    if (tensor.ndim != 3 or tensor.shape[0] < 1 or tensor.shape[1] != tensor.shape[2]
            or points.ndim != 1 or points.size < 2):
        raise ValueError("invalid tensor or sample shape")
    if not isinstance(derivative_order, (int, np.integer)) or not 0 <= derivative_order <= 3:
        raise ValueError("derivative_order must be an integer in 0..3")
    if not np.all(np.isfinite(tensor)) or not np.all(np.isfinite(points)):
        raise ValueError("inputs must be finite")
    count = points.size
    expected = np.exp(-2j*np.pi*np.arange(count)/count)
    if not np.allclose(points, expected, rtol=0, atol=1e-12):
        raise ValueError("incorrect sampling grid")
    output = []
    for order in range(derivative_order+1):
        folded = np.zeros((count,)+tensor.shape[1:], dtype=complex)
        for degree in range(order, tensor.shape[0]):
            factor = 1
            for j in range(order):
                factor *= degree-j
            folded[(degree-order) % count] += factor*tensor[degree]
        output.append(np.fft.fft(folded, axis=0))
    return output[0] if derivative_order == 0 else np.asarray(output)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Scientific cases with independent mutable inputs on both sides."""
    return [{'setup': 'import numpy as np\n'
               'coefficient_tensor=np.arange(44,dtype=float).reshape(11,2,2)/10\n'
               'sample_points=np.exp(-2j*np.pi*np.arange(11)/11)',
      'call': 'evaluate_resultant_fft(coefficient_tensor.copy(), sample_points.copy())',
      'gold_call': '_oracle_evaluate_resultant_fft(coefficient_tensor.copy(), sample_points.copy())'},
     {'setup': 'import numpy as np\n'
               'coefficient_tensor=np.array([[[2.]],[[3.]]])\n'
               'sample_points=np.array([1+0j,-1+0j])',
      'call': 'evaluate_resultant_fft(coefficient_tensor.copy(), sample_points.copy())',
      'gold_call': '_oracle_evaluate_resultant_fft(coefficient_tensor.copy(), sample_points.copy())'},
     {'setup': 'import numpy as np\n'
               'coefficient_tensor = np.ones((4, 2, 2))\n'
               'sample_points = np.exp(2j * np.pi * np.arange(4) / 4)\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        evaluate_resultant_fft(coefficient_tensor.copy(), sample_points.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_evaluate_resultant_fft(coefficient_tensor.copy(), '
               'sample_points.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'T=np.arange(13*4).reshape(13,2,2)/17.;z=np.exp(-2j*np.pi*np.arange(4)/4)',
      'call': 'evaluate_resultant_fft(T.copy(), z.copy(), 3)',
      'gold_call': '_oracle_evaluate_resultant_fft(T.copy(), z.copy(), 3)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(17);T=rng.normal(size=(10,3,3))+1j*rng.normal(size=(10,3,3));z=np.exp(-2j*np.pi*np.arange(3)/3)',
      'call': 'evaluate_resultant_fft(T.copy(), z.copy(), 2)',
      'gold_call': '_oracle_evaluate_resultant_fft(T.copy(), z.copy(), 2)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'T=np.array([[[2.+1j,3.],[4.,-2.]]]);z=np.exp(-2j*np.pi*np.arange(5)/5)',
      'call': 'evaluate_resultant_fft(T.copy(), z.copy(), 3)',
      'gold_call': '_oracle_evaluate_resultant_fft(T.copy(), z.copy(), 3)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nT=np.arange(36).reshape(9,2,2)/9.;z=np.array([1.,-1.])',
      'call': 'evaluate_resultant_fft(T.copy(), z.copy())',
      'gold_call': '_oracle_evaluate_resultant_fft(T.copy(), z.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'T = np.zeros((5, 2, 2))\n'
               'z = np.array([1.0, -1.0])\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        evaluate_resultant_fft(T.copy(), z.copy(), 4)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_evaluate_resultant_fft(T.copy(), z.copy(), 4)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()',
      'tol': 0}]
