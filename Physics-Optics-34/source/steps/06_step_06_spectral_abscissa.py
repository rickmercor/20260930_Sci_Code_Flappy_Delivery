"""
Return the maximum real part of all eigenvalues of a finite complex square matrix. The input dimension is 1 through 100 and every entry magnitude is at most 1e6. No filtering by imaginary part, eigenvector norm or propagation direction is performed. Return a native float without decimal rounding. Invalid shape, nonfinite input, eigensolver failure or nonfinite eigenvalues raises ValueError.

For the dimensionless tangential Maxwell operator this scalar is the greatest real part of k3/k0 among the retained propagation modes. The target is a finite-matrix observable, so a mode that a continuum interpretation might discard still belongs to its spectrum.

Returns
-------
native float, spectral abscissa of the input matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_abscissa(matrix: "np.ndarray | list | tuple") -> float:
    """Return the greatest real part of the full eigenvalue spectrum.

    Parameters
    ----------
    matrix : array_like
        Complex square matrix of dimension 1 through 100, entries at most 1e6.

    Returns
    -------
    float
        Maximum real eigenvalue component, without rounding.

    Raises
    ------
    ValueError
        For invalid shape, nonfinite data, exceeded bounds or eigensolver failure.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_spectral_abscissa(matrix: "np.ndarray | list | tuple") -> float:
    try:a=np.asarray(matrix,dtype=complex)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('matrix') from exc
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not 1<=len(a)<=100 or not np.all(np.isfinite(a)) or np.max(np.abs(a))>1e6:raise ValueError('matrix')
    try:v=np.linalg.eigvals(a)
    except np.linalg.LinAlgError as exc:raise ValueError('eigensolver') from exc
    if not np.all(np.isfinite(v)):raise ValueError('spectrum')
    return float(np.max(np.real(v)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases=[
        {'setup':'import numpy as np','call':'spectral_abscissa([[1,4],[2,3j]])','gold_call':'_oracle_spectral_abscissa([[1,4],[2,3j]])'},
        {'setup':'import numpy as np','call':'spectral_abscissa([[-2.5+.1j]])','gold_call':'_oracle_spectral_abscissa([[-2.5+.1j]])'},
        {'setup':'import numpy as np','call':'spectral_abscissa([[1,1e6],[0,1]])','gold_call':'_oracle_spectral_abscissa([[1,1e6],[0,1]])'},
        {'setup':'import numpy as np','call':'spectral_abscissa(np.diag([3-4j,2+5j,-1+2j]))','gold_call':'_oracle_spectral_abscissa(np.diag([3-4j,2+5j,-1+2j]))'}]
    for a in ('[]','[[float("inf")]]','np.zeros((2,3))','[[1e7]]'):
        setup='import numpy as np\n'
        for name,fn in [('run_model','spectral_abscissa'),('run_gold','_oracle_spectral_abscissa')]:
            setup+=f'def {name}():\n    try:\n        {fn}({a})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    return cases
