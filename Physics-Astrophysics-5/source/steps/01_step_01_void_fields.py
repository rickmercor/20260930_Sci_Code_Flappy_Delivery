"""
Construct dimensionless density ratio and peculiar-velocity divergence on an integer grid. For coordinates p=(i,j,k), use s(p)=min_a sum_b ((p_b-centers[a,b])/scales[a,b])**2, density ratio 1-0.92 exp(-s) and divergence 1.2-s+0.000001*(i+n*j+n*n*k). The full additive term participates in all subsequent comparisons. Accept an integer grid side from 5 through 65, nonempty real center and scale arrays of equal shape (m,3), centers inside [0,n-1] and scales inside [0.25,n]. Return two float arrays of shape (n,n,n), with axes ordered x,y,z. Invalid inputs raise ValueError.

The synthetic wells isolate the geometrical part of a cosmic-void finder from particle interpolation and survey selection. The density contrast is the density ratio minus one, while positive divergence describes expanding material. The index spacing is one; the conversion to physical cell volume belongs to the final measurement rather than these fields.

Returns
-------
tuple of two float arrays, density ratio and divergence on the index grid
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def void_fields(n, centers, scales):
    """Return density ratio and divergence fields.

    Parameters
    ----------
    n : int
        Grid side from 5 through 65, excluding booleans.
    centers : array_like
        Nonempty real array of shape (m,3) inside [0,n-1].
    scales : array_like
        Real array of the same shape inside [0.25,n].

    Returns
    -------
    tuple of ndarray
        Density ratio and divergence, each with shape (n,n,n).

    Raises
    ------
    ValueError
        If the input domain, shape or finiteness contract is violated.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_void_fields(n, centers, scales):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not 5 <= n <= 65:
        raise ValueError("n must be an integer from 5 through 65")
    arrays = []
    for value in (centers, scales):
        try:
            raw = np.asarray(value)
            if raw.dtype.kind not in 'iuf' or raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0:
                raise ValueError("nonempty real (m,3) array required")
            arr = raw.astype(float)
        except (TypeError, OverflowError) as exc:
            raise ValueError("invalid array") from exc
        if not np.isfinite(arr).all():
            raise ValueError("finite arrays required")
        arrays.append(arr)
    c, a = arrays
    if c.shape != a.shape or np.any(c < 0) or np.any(c > n-1) or np.any(a < .25) or np.any(a > n):
        raise ValueError("centers or scales outside domain")
    xyz = np.indices((n,n,n), dtype=float)
    s = np.full((n,n,n), np.inf)
    for center, scale in zip(c,a):
        value = np.sum(((xyz-center[:,None,None,None])/scale[:,None,None,None])**2, axis=0)
        s = np.minimum(s, value)
    return 1-.92*np.exp(-s), 1.2-s+1e-6*(xyz[0]+n*xyz[1]+n*n*xyz[2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases = [
        {'setup': '', 'call': 'np.stack(void_fields(9, [[4,4,4]], [[3,3,3]]), axis=0)', 'gold_call': 'np.stack(_oracle_void_fields(9, [[4,4,4]], [[3,3,3]]), axis=0)'},
        {'setup': '', 'call': 'np.stack(void_fields(5, [[0,0,0]], [[.25,.25,.25]]), axis=0)', 'gold_call': 'np.stack(_oracle_void_fields(5, [[0,0,0]], [[.25,.25,.25]]), axis=0)'},
        {'setup': '', 'call': 'np.stack(void_fields(7, [[2,3,4],[4,3,2]], [[2,3,2],[3,2,3]]), axis=0)', 'gold_call': 'np.stack(_oracle_void_fields(7, [[2,3,4],[4,3,2]], [[2,3,2],[3,2,3]]), axis=0)'},
    ]
    for args in ('4, [[0,0,0]], [[1,1,1]]', '5, [[0,0,0]], [[0,1,1]]', '5, [[float("nan"),0,0]], [[1,1,1]]'):
        setup = ''
        for name, function in (('run_model','void_fields'), ('run_gold','_oracle_void_fields')):
            setup += f'def {name}():\n    try:\n        {function}({args})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    return cases
