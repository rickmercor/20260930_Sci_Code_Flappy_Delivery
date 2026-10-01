"""
Construct the binary detector matrix from circulant column supports. For modulus m and an integer array offsets of shape (b,k), column a*m+q contains ones at rows (q+offsets[a,t]) modulo m. Require 3<=m<=16, 1<=b<=4, 2<=k<=m, b*m<=64 and distinct offsets in each row, with each offset in [0,m-1]. Return an integer array of shape (m,b*m). These are synthetic construction bounds, not a fitted physical regime. Invalid inputs raise ValueError.

Each column describes the detectors flipped by one binary fault. Translation of a fixed support creates regular detector incidence without importing an experimental error model. Matrix multiplication modulo two subsequently maps a fault pattern to its syndrome.

Returns
-------
numpy.ndarray, binary detector incidence
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def detector_matrix(modulus, offsets):
    """Construct circulant fault supports.

    Parameters
    ----------
    modulus : int
        Nonboolean integer in [3,16].
    offsets : array_like
        Integer-valued support table with the bounds stated above.

    Returns
    -------
    numpy.ndarray
        Binary detector matrix.

    Raises
    ------
    ValueError
        If construction bounds or distinct-support conditions fail.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_detector_matrix(modulus, offsets):
    if isinstance(modulus,(bool,np.bool_)) or not isinstance(modulus,(int,np.integer)) or not 3<=modulus<=16:
        raise ValueError('invalid modulus')
    a=np.asarray(offsets)
    if a.ndim!=2 or a.dtype.kind not in 'iuf' or not np.all(np.isfinite(a)):
        raise ValueError('invalid offsets')
    b,k=a.shape
    if not 1<=b<=4 or not 2<=k<=modulus or b*modulus>64 or np.any(a!=np.floor(a)) or np.any(a<0) or np.any(a>=modulus):
        raise ValueError('offset bounds')
    if any(len(set(row))!=k for row in a.tolist()):raise ValueError('duplicate offset')
    H=np.zeros((modulus,b*modulus),dtype=int)
    for f in range(b):
        for q in range(modulus):H[(q+a[f].astype(int))%modulus,f*modulus+q]=1
    return H

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\na=np.array([[0,1,2],[0,1,3],[0,2,4]])',
            'call': 'detector_matrix(7,a.copy())',
            'gold_call': '_oracle_detector_matrix(7,a.copy())',
        },
        {
            'setup': 'import numpy as np\na=np.array([[0,1]])',
            'call': 'detector_matrix(3,a.copy())',
            'gold_call': '_oracle_detector_matrix(3,a.copy())',
        },
        {
            'setup': 'import numpy as np\na=np.array([[0,2,4,5],[1,2,3,5]])',
            'call': 'detector_matrix(6,a.copy())',
            'gold_call': '_oracle_detector_matrix(6,a.copy())',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try: detector_matrix(3,[[0,0]])\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_gold():\n    try: _oracle_detector_matrix(3,[[0,0]])\n    except ValueError: return 1\n    except Exception: return 2\n    return 0',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
