"""
Extract admissible real hidden-variable candidates.

The ascending coefficients define $D(x)=\sum_{n=0}^{K-1}c_nx^n$.

A complex root $z$ is nearly real when $|\operatorname{Im}z|\le\tau$.

An optional closed interval $[a,b]$ additionally restricts

$a\le\operatorname{Re}z\le b$.

Returns
-------
A sorted real NumPy array containing all near-real roots that satisfy the optional closed interval, with multiplicities preserved.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def extract_hidden_candidates(
    coefficients: np.ndarray, imaginary_tolerance: float = 1e-8,
    interval: tuple | None = None
) -> np.ndarray:
    r"""Solve an ascending-order polynomial and retain nearly real roots.
    
    Parameters
    ----------
    coefficients : np.ndarray
        Finite one-dimensional real or complex coefficient array of length
        $K\ge2$, representing $D(x)=\sum_{n=0}^{K-1}c_nx^n$.
        The final entry is the leading coefficient and must be nonnegligible.
    imaginary_tolerance : float, optional
        Finite positive tolerance $\tau$, default $10^{-8}$. A root $z$
        passes the reality gate when $|\operatorname{Im}z|\le\tau$.
    interval : tuple or None, optional
        Default `None` applies no interval gate. Otherwise provide two
        finite real bounds $(a,b)$ with $a\le b$; retain roots satisfying
        $a\le\operatorname{Re}z\le b$, including the endpoints.
    
    Returns
    -------
    candidates : np.ndarray
        Nonempty real vector of retained real parts, sorted in ascending
        order. Preserve multiplicities; do not merge repeated roots.
    
    Raises
    ------
    ValueError
        If coefficients are not a finite vector of length at least two,
        $\tau$ is nonfinite or nonpositive, interval bounds are invalid,
        or no root survives both gates. Also raise when
        $|c_{K-1}|\le100\epsilon\max\{1,\max_n|c_n|\}$, where
        $\epsilon$ is double-precision machine epsilon.
    
    Notes
    -----
    Apply the reality test to each complex root before retaining its real
    part and applying the optional interval gate. No root polishing or
    multiplicity reduction is requested. Do not mutate the inputs.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_extract_hidden_candidates(
    coefficients: np.ndarray, imaginary_tolerance: float = 1e-8,
    interval: tuple | None = None
) -> np.ndarray:
    """Reference companion-root extraction with ascending-order handling."""
    coefficients = np.asarray(coefficients)
    if (
        coefficients.ndim != 1
        or coefficients.size < 2
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError("coefficients must be a finite one-dimensional polynomial")
    if not np.isfinite(imaginary_tolerance) or imaginary_tolerance <= 0:
        raise ValueError("imaginary_tolerance must be positive and finite")
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    if abs(coefficients[-1]) <= 100 * np.finfo(float).eps * scale:
        raise ValueError("the leading coefficient must be nonzero")
    if interval is not None:
        bounds = np.asarray(interval)
        if bounds.shape != (2,) or not np.isrealobj(bounds) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
            raise ValueError("invalid root interval")
    roots = np.roots(coefficients[::-1])
    candidates = roots[np.abs(roots.imag) <= imaginary_tolerance].real
    if interval is not None:
        candidates = candidates[(candidates >= bounds[0]) & (candidates <= bounds[1])]
    if candidates.size == 0:
        raise ValueError("no root passes the imaginary-part gate")
    return np.sort(candidates)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Scientific cases with independent mutable inputs on both sides."""
    return [{'setup': 'import numpy as np\ncoefficients=np.array([-2.,1.,1.])',
      'call': 'extract_hidden_candidates(coefficients.copy())',
      'gold_call': '_oracle_extract_hidden_candidates(coefficients.copy())'},
     {'setup': 'import numpy as np\ncoefficients=np.array([2.,-3.,1.])',
      'call': 'extract_hidden_candidates(coefficients.copy())',
      'gold_call': '_oracle_extract_hidden_candidates(coefficients.copy())'},
     {'setup': 'import numpy as np\n'
               'coefficients = np.array([1.0, 2.0, 0.0])\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        extract_hidden_candidates(coefficients.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_extract_hidden_candidates(coefficients.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\na=np.polynomial.polynomial.polyfromroots([-2.,.5,1.5,2.5,4.])',
      'call': 'extract_hidden_candidates(a.copy(), interval=(0.0, 3.0))',
      'gold_call': '_oracle_extract_hidden_candidates(a.copy(), interval=(0.0, 3.0))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\na=np.array([0.,-2.,1.])',
      'call': 'extract_hidden_candidates(a.copy(), interval=(0.0, 0.0))',
      'gold_call': '_oracle_extract_hidden_candidates(a.copy(), interval=(0.0, 0.0))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'a=np.polynomial.polynomial.polymul(np.array([-1.,1.]),np.array([1.,0.,1.]))',
      'call': 'extract_hidden_candidates(a.copy())',
      'gold_call': '_oracle_extract_hidden_candidates(a.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'a = np.array([-1.0, 1.0])\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        extract_hidden_candidates(a.copy(), interval=(3.0, 0.0))\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_extract_hidden_candidates(a.copy(), interval=(3.0, 0.0))\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()',
      'tol': 0},
     {'setup': 'import numpy as np\n'
               'a = np.array([-1.0, 1.0])\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        extract_hidden_candidates(a.copy(), interval=(2.0, 3.0))\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_extract_hidden_candidates(a.copy(), interval=(2.0, 3.0))\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()',
      'tol': 0}]
