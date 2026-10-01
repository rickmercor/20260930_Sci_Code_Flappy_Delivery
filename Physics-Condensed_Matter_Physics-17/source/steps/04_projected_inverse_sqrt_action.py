"""
Evaluate the compressed overlap inverse-square-root action on arbitrary right-hand sides in a supplied orthonormal subspace.

Compression and taking a matrix inverse square root do not commute. The finite Galerkin action lives only in the retained subspace, but the input probes need not lie there. A weak positive overlap direction is physical here, not a license to replace the operation by a thresholded pseudoinverse; uniform overlap rescaling must preserve the inverse-square-root scaling law.

Returns
-------
np.ndarray with the same shape as probes, the finite Galerkin inverse-square-root action.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real
import numpy as np

def projected_inverse_sqrt_action(
    overlap: np.ndarray,
    probes: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Apply the Galerkin inverse square root in a supplied subspace.

    Parameters
    ----------
    overlap : np.ndarray
        Finite real symmetric strictly positive-definite matrix (n, n), n>=1.
        Positive definiteness applies also outside the supplied subspace.
    probes : np.ndarray
        Finite real right-hand-side block (n, m), m>=1. Zero or dependent
        columns and components outside the span of basis are valid inputs.
    basis : np.ndarray
        Finite real orthonormal frame (n, r), 1<=r<=n. It may be rectangular
        and need not contain probes or be an invariant subspace of overlap.
        Its Gram matrix equals identity within rtol=1e-10, atol=1e-12.
        Overlap symmetry uses rtol=0, atol=1e-12.

    Returns
    -------
    action : np.ndarray
        Real (n, m) action: compress overlap to basis, apply its spectral
        inverse square root to the projected probes, then lift back. The
        output lies in the supplied span; no action on its orthogonal
        complement is appended. Preserve original probe amplitudes.
        Retain every strictly positive projected eigenvalue without a
        cutoff or regularization. Scalar overlap rescaling by a>0 must
        scale the action by a**(-1/2); no absolute eigenvalue floor is used.
        Return finite values whenever the specified action is representable.

    Raises
    ------
    ValueError
        If shapes, realness, finiteness, symmetry, full-overlap or projected
        positive definiteness, or orthonormality fail, or the final action
        is not finite in binary64.

    Notes
    -----
    The supplied frame is the entire finite-projection model: do not
    enlarge it from the probes. Uniformly small/large overlaps and weak
    positive spectral directions are supported. Use stable scalar
    equilibration where needed, without changing the mathematical map.
    Do not mutate inputs. Equivalent matrix-function evaluations are valid.
    """
    return np.empty_like(probes, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_projected_inverse_sqrt_action(overlap: np.ndarray, probes: np.ndarray, basis: np.ndarray) -> np.ndarray:
    import numpy as np

    if any(np.iscomplexobj(a) for a in (overlap, probes, basis)):
        raise ValueError("inputs must be real")
    s = np.asarray(overlap, dtype=float)
    o = np.asarray(probes, dtype=float)
    q = np.asarray(basis, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] < 1:
        raise ValueError("overlap must be a nonempty square matrix")
    n = s.shape[0]
    if o.ndim != 2 or o.shape[0] != n or o.shape[1] < 1:
        raise ValueError("probes must have shape (n,m)")
    if q.ndim != 2 or q.shape[0] != n or not 1 <= q.shape[1] <= n:
        raise ValueError("basis must have shape (n,r), 1<=r<=n")
    if not all(np.all(np.isfinite(a)) for a in (s, o, q)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(s, s.T, rtol=0.0, atol=1e-12):
        raise ValueError("overlap must be symmetric")
    if not np.allclose(q.T @ q, np.eye(q.shape[1]), rtol=1e-10, atol=1e-12):
        raise ValueError("basis columns must be orthonormal")
    scale = float(np.max(np.abs(s)))
    if scale == 0.0:
        raise ValueError("overlap must be positive definite")
    scaled = s / scale
    if float(np.linalg.eigvalsh(scaled)[0]) <= 0.0:
        raise ValueError("overlap must be positive definite on the full space")
    projected = q.T @ scaled @ q
    projected = 0.5 * projected + 0.5 * projected.T
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("projected overlap must be positive definite")
    coefficients = eigenvectors.T @ (q.T @ o)
    coefficients = coefficients / np.sqrt(eigenvalues[:, None])
    result = (q @ (eigenvectors @ coefficients)) / np.sqrt(scale)
    if not np.all(np.isfinite(result)):
        raise ValueError("the action is not representable in binary64")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Original regressions and independent scientific/stability regimes."""
    return [{'setup': 'import numpy as np\n'
           'S=np.array([[2.,0.3],[0.3,1.2]]); O=np.array([[1.],[-1.]]); Q=np.eye(2)',
  'call': 'projected_inverse_sqrt_action(S,O,Q)',
  'gold_call': '_oracle_projected_inverse_sqrt_action(S,O,Q)'},
 {'setup': 'import numpy as np\n'
           'S=np.diag([1.,4.,9.]); O=np.array([[1.],[2.],[3.]]); '
           'Q=np.array([[1.,0.],[0.,1.],[0.,0.]])',
  'call': 'projected_inverse_sqrt_action(S,O,Q)',
  'gold_call': '_oracle_projected_inverse_sqrt_action(S,O,Q)'},
 {'setup': 'import numpy as np\nS=np.array([[1.0,1e-9],[1e-9,1.0+2e-9]]); O=np.eye(2); Q=np.eye(2)',
  'call': 'projected_inverse_sqrt_action(S,O,Q)',
  'gold_call': '_oracle_projected_inverse_sqrt_action(S,O,Q)'},
 {'setup': 'import numpy as np\nS=np.diag([1e-12,4.0]); O=np.eye(2); Q=np.eye(2)',
  'call': 'projected_inverse_sqrt_action(S,O,Q)',
  'gold_call': '_oracle_projected_inverse_sqrt_action(S,O,Q)'},
 {'setup': 'import numpy as np\n'
           'S=np.diag([1.,-1.]); O=np.ones((2,1)); Q=np.eye(2)\n'
           'def run_model():\n'
           '    try:\n'
           '        projected_inverse_sqrt_action(S,O,Q)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           'def run_gold():\n'
           '    try:\n'
           '        _oracle_projected_inverse_sqrt_action(S,O,Q)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2',
  'call': 'run_model()',
  'gold_call': 'run_gold()'},
 {'setup': 'import numpy as np\n'
           'S=np.eye(2); O=np.ones((2,1)); Q=np.array([[1.,1.],[0.,0.]])\n'
           'def run_model():\n'
           '    try:\n'
           '        projected_inverse_sqrt_action(S,O,Q)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           'def run_gold():\n'
           '    try:\n'
           '        _oracle_projected_inverse_sqrt_action(S,O,Q)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2',
  'call': 'run_model()',
  'gold_call': 'run_gold()'},
 {'setup': 'import numpy as np\n'
           'S=np.array([[2.,.4,.2,0.],[.4,1.5,.1,.3],[.2,.1,3.,.2],[0.,.3,.2,1.]])\n'
           'Q=np.array([[.6,0.],[0.,.8],[.8,0.],[0.,-.6]])\n'
           'O=np.array([[1.,0.,2.,0.],[2.,-1.,4.,0.],[-.5,2.,-1.,0.],[3.,.2,6.,0.]])',
  'call': 'projected_inverse_sqrt_action(S,O,Q)',
  'gold_call': '_oracle_projected_inverse_sqrt_action(S,O,Q)'},
 {'setup': 'import numpy as np\n'
           'S=np.diag([1e-18,4.,9.]); Q=np.eye(3)\n'
           'O=np.array([[2e-9,-3e-9],[2.,0.],[0.,-6.]])',
  'call': 'projected_inverse_sqrt_action(S,O,Q)',
  'gold_call': '_oracle_projected_inverse_sqrt_action(S,O,Q)'},
 {'setup': 'import numpy as np\n'
           'V=np.array([[.6,-.8],[.8,.6]])\n'
           'S=V@np.diag([1.,1.+1e-7])@V.T; Q=np.array([[.8,.6],[-.6,.8]])\n'
           'O=np.array([[1.,-2.,0.],[3.,1.,0.]])',
  'call': 'projected_inverse_sqrt_action(S,O,Q)',
  'gold_call': '_oracle_projected_inverse_sqrt_action(S,O,Q)'},
 {'setup': 'import numpy as np\n'
           'S=np.array([[2.,.25],[.25,1.]]); O=np.array([[1.,-.5],[2.,3.]]); Q=np.eye(2)\n'
           'def _scale_pair(fun):\n'
           '    return np.stack([fun(S*1e-200,O*1e-100,Q),fun(S*1e200,O*1e100,Q)])',
  'call': '_scale_pair(projected_inverse_sqrt_action)',
  'gold_call': '_scale_pair(_oracle_projected_inverse_sqrt_action)'},
 {'setup': 'import numpy as np\n'
           'S=np.diag([1.,4.,-1.]);O=np.ones((3,1));Q=np.eye(3)[:,:2]\n'
           'def run_model():\n'
           '    try:\n'
           '        projected_inverse_sqrt_action(S,O,Q)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '\n'
           'def run_gold():\n'
           '    try:\n'
           '        _oracle_projected_inverse_sqrt_action(S,O,Q)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n',
  'call': 'run_model()',
  'gold_call': 'run_gold()'},
 {'setup': 'import numpy as np\n'
           'S=np.eye(2);O=np.array([[1.+1j],[2.+0j]]);Q=np.eye(2)\n'
           'def run_model():\n'
           '    try:\n'
           '        projected_inverse_sqrt_action(S,O,Q)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '\n'
           'def run_gold():\n'
           '    try:\n'
           '        _oracle_projected_inverse_sqrt_action(S,O,Q)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n',
  'call': 'run_model()',
  'gold_call': 'run_gold()'}]
