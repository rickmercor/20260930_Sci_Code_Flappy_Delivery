"""
Preconditioned nonsymmetric Krylov inverse action with a prescribed rank cap.

A finite-rank approximation to the inverse residual Jacobian supplies the auxiliary restoring action. Left preconditioning does not make the composed operator symmetric.

Returns
-------
array (D,), integer rank and array (rank,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def krylov_action(J, K0, residual, tol, max_rank):
    """Approximate an inverse-residual-Jacobian action with a preconditioned Krylov space.

Parameters
----------
J, K0 : finite real arrays (D,D), D>=1
    J is the residual Jacobian; K0 is a fixed LEFT preconditioner.
    Both matrices have smallest/largest singular-value ratio >1e-14.
residual : finite real vector (D,)
tol : finite scalar, 0<=tol<1
max_rank : integer (not bool), 1<=max_rank<=D

Returns
-------
(z, rank, errors) : numerical tuple
    Set A=K0@J, b=K0@residual and beta=||b||_2. If beta<=1e-14,
    return a zero vector, rank 0, and an empty float array.
    Otherwise start v_1=b/beta. At rank m, V has the m orthonormal
    Arnoldi columns and W=A@V. Find the minimum-norm least-squares
    minimizer y of ||W*y-b||_2, with relative SVD cutoff 1e-14.
    Set z=V*y and append ||W*y-b||_2/beta to errors. Stop at the first
    error<=tol or at max_rank; otherwise generate the next v from
    A@v_m by TWO passes of modified Gram-Schmidt against ALL existing
    columns in creation order. Stop on breakdown if its remaining norm
    is <=1e-13*max(1,||A@v_m||_2); else normalize it and continue.
    z has shape (D,), rank is the number of used columns, and errors
    has shape (rank,). On a rank cap return the current approximation;
    a full inverse action is a different result and must not replace it.
    Do not mutate inputs. Do not assume J, K0 or A are symmetric.

Raises
------
ValueError
    If inputs are not finite/real, their shapes disagree, a singular-value
    ratio is <=1e-14, tol is outside [0,1), or max_rank is not an integer
    in [1,D]."""
    return (np.zeros(0), 0, np.zeros(0))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import math
import numpy as np
from scipy.special import gammainc, gammaln
import math
import numpy as np

def _real(value, name, shape=None):
    try:
        if np.iscomplexobj(value):
            raise ValueError(name + ' must be real')
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be finite and real') from exc
    if not np.all(np.isfinite(out)) or (shape is not None and out.shape != shape):
        raise ValueError(name + ' has an invalid shape or nonfinite entry')
    return out

def _scalar(value, name, positive=False, nonnegative=False):
    out = _real(value, name, ())
    x = float(out)
    if positive and x <= 0 or (nonnegative and x < 0):
        raise ValueError(name + ' is outside its domain')
    return x

def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if value < minimum:
        raise ValueError(name + ' is too small')
    return int(value)

def _oracle_krylov_action(J, K0, residual, tol, max_rank):
    """Approximate an inverse-residual-Jacobian action with a preconditioned Krylov space.

Parameters
----------
J, K0 : finite real arrays (D,D), D>=1
    J is the residual Jacobian; K0 is a fixed LEFT preconditioner.
    Both matrices have smallest/largest singular-value ratio >1e-14.
residual : finite real vector (D,)
tol : finite scalar, 0<=tol<1
max_rank : integer (not bool), 1<=max_rank<=D

Returns
-------
(z, rank, errors) : numerical tuple
    Set A=K0@J, b=K0@residual and beta=||b||_2. If beta<=1e-14,
    return a zero vector, rank 0, and an empty float array.
    Otherwise start v_1=b/beta. At rank m, V has the m orthonormal
    Arnoldi columns and W=A@V. Find the minimum-norm least-squares
    minimizer y of ||W*y-b||_2, with relative SVD cutoff 1e-14.
    Set z=V*y and append ||W*y-b||_2/beta to errors. Stop at the first
    error<=tol or at max_rank; otherwise generate the next v from
    A@v_m by TWO passes of modified Gram-Schmidt against ALL existing
    columns in creation order. Stop on breakdown if its remaining norm
    is <=1e-13*max(1,||A@v_m||_2); else normalize it and continue.
    z has shape (D,), rank is the number of used columns, and errors
    has shape (rank,). On a rank cap return the current approximation;
    a full inverse action is a different result and must not replace it.
    Do not mutate inputs. Do not assume J, K0 or A are symmetric.

Raises
------
ValueError
    If inputs are not finite/real, their shapes disagree, a singular-value
    ratio is <=1e-14, tol is outside [0,1), or max_rank is not an integer
    in [1,D]."""
    f = _real(residual, 'residual')
    if f.ndim != 1 or len(f) < 1:
        raise ValueError('residual must be a nonempty vector')
    n = len(f)
    J, K0 = (_real(J, 'J', (n, n)), _real(K0, 'K0', (n, n)))
    tol = _scalar(tol, 'tol', nonnegative=True)
    max_rank = _integer(max_rank, 'max_rank', 1)
    if tol >= 1 or max_rank > n:
        raise ValueError('Require 0<=tol<1 and max_rank<=dimension')
    for A in (J, K0):
        singular = np.linalg.svd(A, compute_uv=False)
        if singular[-1] <= 1e-14 * singular[0]:
            raise ValueError('J and K0 must have singular-value ratio above 1e-14')
    b = K0 @ f
    beta = float(np.linalg.norm(b))
    if beta <= 1e-14:
        return (np.zeros(n), 0, np.zeros(0))
    A = K0 @ J
    V, W, errors = ([], [], [])
    v = b / beta
    for _ in range(max_rank):
        V.append(v.copy())
        w = A @ v
        W.append(w.copy())
        Vm, Wm = (np.column_stack(V), np.column_stack(W))
        y = np.linalg.lstsq(Wm, b, rcond=1e-14)[0]
        z = Vm @ y
        error = float(np.linalg.norm(Wm @ y - b) / beta)
        errors.append(error)
        if error <= tol:
            break
        candidate = w.copy()
        for _pass in range(2):
            for basis in V:
                candidate -= np.dot(basis, candidate) * basis
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-13 * max(1.0, float(np.linalg.norm(w))):
            break
        v = candidate / norm
    return (z, len(V), np.asarray(errors))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '\n'
               'import numpy as np\n'
               'J = np.array([\n'
               '    [-1.0, 0.8, 0.1, 0.0],\n'
               '    [0.0, -0.7, 0.5, 0.2],\n'
               '    [0.2, 0.0, -1.4, 0.6],\n'
               '    [0.0, 0.1, 0.0, -0.9],\n'
               '])\n'
               'K0 = np.array([\n'
               '    [-1.0, 0.1, 0.0, 0.0],\n'
               '    [0.0, -0.8, 0.0, 0.1],\n'
               '    [0.0, 0.0, -1.2, 0.0],\n'
               '    [0.1, 0.0, 0.0, -0.9],\n'
               '])\n'
               'f = np.array([0.4, -0.3, 0.7, -0.2])\n',
      'call': 'krylov_action(J, K0, f, 0.0, 2)',
      'gold_call': '_oracle_krylov_action(J, K0, f, 0.0, 2)'},
     {'setup': '\n'
               'import numpy as np\n'
               'J = np.array([\n'
               '    [-1.0, 0.8, 0.1, 0.0],\n'
               '    [0.0, -0.7, 0.5, 0.2],\n'
               '    [0.2, 0.0, -1.4, 0.6],\n'
               '    [0.0, 0.1, 0.0, -0.9],\n'
               '])\n'
               'K0 = np.array([\n'
               '    [-1.0, 0.1, 0.0, 0.0],\n'
               '    [0.0, -0.8, 0.0, 0.1],\n'
               '    [0.0, 0.0, -1.2, 0.0],\n'
               '    [0.1, 0.0, 0.0, -0.9],\n'
               '])\n'
               'f = np.array([0.4, -0.3, 0.7, -0.2])\n',
      'call': 'krylov_action(J, K0, f, 0.22, 4)',
      'gold_call': '_oracle_krylov_action(J, K0, f, 0.22, 4)'},
     {'setup': '\n'
               'import numpy as np\n'
               'J = np.array([\n'
               '    [-1.0, 0.8, 0.1, 0.0],\n'
               '    [0.0, -0.7, 0.5, 0.2],\n'
               '    [0.2, 0.0, -1.4, 0.6],\n'
               '    [0.0, 0.1, 0.0, -0.9],\n'
               '])\n'
               'K0 = np.array([\n'
               '    [-1.0, 0.1, 0.0, 0.0],\n'
               '    [0.0, -0.8, 0.0, 0.1],\n'
               '    [0.0, 0.0, -1.2, 0.0],\n'
               '    [0.1, 0.0, 0.0, -0.9],\n'
               '])\n'
               'f = np.array([0.4, -0.3, 0.7, -0.2])\n',
      'call': 'krylov_action(J, K0, f, 1e-12, 4)',
      'gold_call': '_oracle_krylov_action(J, K0, f, 1e-12, 4)'},
     {'setup': 'import numpy as np',
      'call': 'krylov_action(-2*np.eye(3), np.eye(3), np.array([1., 2., 3.]), 0.0, 3)',
      'gold_call': '_oracle_krylov_action(-2*np.eye(3), np.eye(3), np.array([1., 2., 3.]), 0.0, 3)'},
     {'setup': 'import numpy as np',
      'call': 'krylov_action(np.eye(3), np.eye(3), np.zeros(3), 0.0, 3)',
      'gold_call': '_oracle_krylov_action(np.eye(3), np.eye(3), np.zeros(3), 0.0, 3)'},
     {'setup': 'import numpy as np',
      'call': 'krylov_action(np.eye(3), np.eye(3), np.full(3, 1e-16), 0.0, 3)',
      'gold_call': '_oracle_krylov_action(np.eye(3), np.eye(3), np.full(3, 1e-16), 0.0, 3)'},
     {'setup': '\n'
               'import numpy as np\n'
               'J = np.array([\n'
               '    [-1.0, 0.8, 0.1, 0.0],\n'
               '    [0.0, -0.7, 0.5, 0.2],\n'
               '    [0.2, 0.0, -1.4, 0.6],\n'
               '    [0.0, 0.1, 0.0, -0.9],\n'
               '])\n'
               'K0 = np.array([\n'
               '    [-1.0, 0.1, 0.0, 0.0],\n'
               '    [0.0, -0.8, 0.0, 0.1],\n'
               '    [0.0, 0.0, -1.2, 0.0],\n'
               '    [0.1, 0.0, 0.0, -0.9],\n'
               '])\n'
               'f = np.array([0.4, -0.3, 0.7, -0.2])\n'
               '\n'
               'J[1] = J[0]\n'
               '\n'
               'def _expect_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_expect_value_error(lambda: krylov_action(J, K0, f, 1e-6, 3))',
      'gold_call': '_expect_value_error(lambda: _oracle_krylov_action(J, K0, f, 1e-6, 3))'},
     {'setup': '\n'
               'import numpy as np\n'
               'J = np.array([\n'
               '    [-1.0, 0.8, 0.1, 0.0],\n'
               '    [0.0, -0.7, 0.5, 0.2],\n'
               '    [0.2, 0.0, -1.4, 0.6],\n'
               '    [0.0, 0.1, 0.0, -0.9],\n'
               '])\n'
               'K0 = np.array([\n'
               '    [-1.0, 0.1, 0.0, 0.0],\n'
               '    [0.0, -0.8, 0.0, 0.1],\n'
               '    [0.0, 0.0, -1.2, 0.0],\n'
               '    [0.1, 0.0, 0.0, -0.9],\n'
               '])\n'
               'f = np.array([0.4, -0.3, 0.7, -0.2])\n'
               '\n'
               'def _expect_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_expect_value_error(lambda: krylov_action(J, K0, f, 1e-6, 5))',
      'gold_call': '_expect_value_error(lambda: _oracle_krylov_action(J, K0, f, 1e-6, 5))'},
     {'setup': '\nimport numpy as np\nJ = np.diag([1.0, 1e-8])\nK0 = J.copy()\nr = np.array([1.0, 1e8])\n',
      'call': 'krylov_action(J, K0, r, 0.0, 2)',
      'gold_call': '_oracle_krylov_action(J, K0, r, 0.0, 2)'}]
