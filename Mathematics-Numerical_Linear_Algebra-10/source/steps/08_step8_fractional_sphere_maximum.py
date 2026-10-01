"""
Compute a globally maximal fractional quadratic on a sphere.

An anisotropic residual uncertainty set produces a quadratic numerator and

a direction-dependent positive denominator. The global maximum must include

singular stationary cases and repeated extreme eigenvalues; a local stationary

point or a numerator-only maximization is insufficient.

Returns
-------
float, the globally maximal quotient.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fractional_sphere_maximum(A: np.ndarray, g: np.ndarray, c: float,
                             B: np.ndarray, tau: float) -> float:
    r"""Maximize a fractional quadratic over a Euclidean sphere.

    Parameters
    ----------
    A : np.ndarray
        Finite real symmetric matrix of shape $d\times d$, $d\ge1$;
        it may be indefinite and may have repeated eigenvalues.
    g : np.ndarray
        Finite real vector of length $d$.
    c : float
        Finite real constant.
    B : np.ndarray
        Finite real symmetric positive-semidefinite matrix of shape
        $d\times d$. Thus the denominator below is strictly positive.
    tau : float
        Finite nonnegative sphere radius.

    Returns
    -------
    value : float
        $\max_{\lVert z\rVert_2=\tau}
        (c+2g^\top z+z^\top Az)/(1+z^\top Bz)$.
        For $\tau=0$, return $c$. The constraint is a sphere, including
        when $A$ is negative definite. Return the global value even
        when the maximizing direction is not unique. Numerical answers
        are compared at relative and absolute tolerance $10^{-9}$.

    Raises
    ------
    ValueError
        For nonfinite inputs, incompatible or empty shapes, nonsymmetric
        matrices, non-positive-semidefinite $B$, or negative $\tau$.
        Symmetry and semidefiniteness may be assessed at relative
        tolerance $10^{-12}$ against a scale of at least one.
    """
    return value

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _sphere_quadratic_maximum(H, g, tau):
    """Dual solution of the equality-constrained trust-region problem."""
    d, U = np.linalg.eigh(H)
    h = U.T @ g
    top = float(d[-1])
    gaps = top - d
    eps = 64 * np.finfo(float).eps
    top_mask = gaps <= eps * max(1.0, float(np.max(np.abs(d))))
    lower = ~top_mask
    hard_norm = np.linalg.norm(h[lower] / gaps[lower])
    if np.linalg.norm(h[top_mask]) <= eps * max(1.0, np.linalg.norm(g)) and hard_norm <= tau:
        # Add the missing radius in the entire top eigenspace. Its
        # orientation is irrelevant to the optimal value.
        return float(top * tau**2 + np.sum(h[lower]**2 / gaps[lower]))
    lo = 0.0
    hi = max(1.0, float(np.linalg.norm(h) / tau))
    # Solve for the shift above the top eigenvalue, avoiding cancellation.
    while np.linalg.norm(h / (gaps + hi)) > tau:
        hi *= 2.0
    for _ in range(100):
        shift = (lo + hi) / 2.0
        if np.linalg.norm(h / (gaps + shift)) > tau:
            lo = shift
        else:
            hi = shift
    shift = hi
    return float((top + shift) * tau**2 + np.sum(h**2 / (gaps + shift)))


def _oracle_fractional_sphere_maximum(A, g, c, B, tau):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    g = np.asarray(g, dtype=float)
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be nonempty square")
    if B.shape != A.shape or g.shape != (A.shape[0],):
        raise ValueError("incompatible shapes")
    if not all(np.all(np.isfinite(x)) for x in (A, B, g)):
        raise ValueError("nonfinite array")
    if not np.isscalar(c) or not np.isscalar(tau) or not np.isfinite(c) or not np.isfinite(tau) or tau < 0:
        raise ValueError("invalid scalar")
    for X in (A, B):
        if np.max(np.abs(X - X.T)) > 1e-12 * max(1.0, np.max(np.abs(X))):
            raise ValueError("matrix must be symmetric")
    A = (A + A.T) / 2
    B = (B + B.T) / 2
    if np.linalg.eigvalsh(B)[0] < -1e-12 * max(1.0, np.linalg.norm(B, 2)):
        raise ValueError("B must be positive semidefinite")
    if tau == 0:
        return float(c)
    bound = abs(c) + 2 * tau * np.linalg.norm(g) + tau**2 * np.linalg.norm(A, 2)
    lo, hi = -float(bound) - 1.0, float(bound) + 1.0
    for _ in range(80):
        rho = (lo + hi) / 2
        f = c - rho + _sphere_quadratic_maximum(A - rho * B, g, tau)
        if f > 0:
            lo = rho
        else:
            hi = rho
    return float((lo + hi) / 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'A=np.array([[3.,.7],[.7,-1.]])\n'
               'B=np.array([[.2,.1],[.1,2.]])\n'
               'g=np.array([.4,-.8]); c=1.7; tau=.65',
      'call': 'fractional_sphere_maximum(A,g,c,B,tau)',
      'gold_call': '_oracle_fractional_sphere_maximum(A,g,c,B,tau)'},
     {'setup': 'import numpy as np\n'
               'A=np.diag([5.,5.,1.]); B=np.eye(3)*.3\n'
               'g=np.array([0.,0.,.5]); c=2.; tau=1.',
      'call': 'fractional_sphere_maximum(A,g,c,B,tau)',
      'gold_call': '_oracle_fractional_sphere_maximum(A,g,c,B,tau)'},
     {'setup': 'import numpy as np\n'
               'A=np.diag([5.,5.,1.]); B=np.eye(3)*.3\n'
               'g=np.array([0.,0.,.5]); c=2.; tau=.125',
      'call': 'fractional_sphere_maximum(A,g,c,B,tau)',
      'gold_call': '_oracle_fractional_sphere_maximum(A,g,c,B,tau)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[4.,1.],[1.,2.]]); B=np.diag([4.,.1])\n'
               'g=np.zeros(2); c=.3; tau=1.2',
      'call': 'fractional_sphere_maximum(A,g,c,B,tau)',
      'gold_call': '_oracle_fractional_sphere_maximum(A,g,c,B,tau)'},
     {'setup': 'import numpy as np\n'
               'A=np.diag([-4.,-2.]); B=np.zeros((2,2))\n'
               'g=np.zeros(2); c=-3.; tau=.7',
      'call': 'fractional_sphere_maximum(A,g,c,B,tau)',
      'gold_call': '_oracle_fractional_sphere_maximum(A,g,c,B,tau)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.]]); B=np.array([[.6]])\n'
               'g=np.array([-.9]); c=.4; tau=.8',
      'call': 'fractional_sphere_maximum(A,g,c,B,tau)',
      'gold_call': '_oracle_fractional_sphere_maximum(A,g,c,B,tau)'},
     {'setup': 'import numpy as np\nA=np.eye(3); B=np.eye(3); g=np.ones(3); c=-2.; tau=0.',
      'call': 'fractional_sphere_maximum(A,g,c,B,tau)',
      'gold_call': '_oracle_fractional_sphere_maximum(A,g,c,B,tau)'},
     {'setup': 'import numpy as np\n'
               'A=np.diag([5.,5.,1.]); B=np.eye(3)*.3\n'
               'g=np.array([1e-6,0.,.5]); c=2.; tau=1.',
      'call': 'fractional_sphere_maximum(A,g,c,B,tau)',
      'gold_call': '_oracle_fractional_sphere_maximum(A,g,c,B,tau)'},
     {'setup': 'import numpy as np\n'
               'U=np.linalg.qr(np.array([[1.,2.,3.],[2.,-1.,1.],[1.,1.,-2.]]))[0]\n'
               'A=U@np.diag([5.,5.,1.])@U.T; B=np.eye(3)*.3\n'
               'g=U@np.array([0.,0.,.5]); c=2.; tau=1.',
      'call': 'fractional_sphere_maximum(A,g,c,B,tau)',
      'gold_call': '_oracle_fractional_sphere_maximum(A,g,c,B,tau)'},
     {'setup': 'import numpy as np\n'
               'A=np.eye(2); B=np.eye(2); g=np.ones(2); c=1.; tau=1.\n'
               'A=np.zeros((0,0))\n'
               'def run_model():\n'
               '    try:\n'
               '        fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'A=np.eye(2); B=np.eye(2); g=np.ones(2); c=1.; tau=1.\n'
               'g=np.ones(3)\n'
               'def run_model():\n'
               '    try:\n'
               '        fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'A=np.eye(2); B=np.eye(2); g=np.ones(2); c=1.; tau=1.\n'
               'A[0,1]=1.\n'
               'def run_model():\n'
               '    try:\n'
               '        fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'A=np.eye(2); B=np.eye(2); g=np.ones(2); c=1.; tau=1.\n'
               'B[0,0]=-1.\n'
               'def run_model():\n'
               '    try:\n'
               '        fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'A=np.eye(2); B=np.eye(2); g=np.ones(2); c=1.; tau=1.\n'
               'tau=-1.\n'
               'def run_model():\n'
               '    try:\n'
               '        fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'A=np.eye(2); B=np.eye(2); g=np.ones(2); c=1.; tau=1.\n'
               'g[0]=np.nan\n'
               'def run_model():\n'
               '    try:\n'
               '        fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'A=np.eye(2); B=np.eye(2); g=np.ones(2); c=1.; tau=1.\n'
               'c=np.inf\n'
               'def run_model():\n'
               '    try:\n'
               '        fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_fractional_sphere_maximum(A,g,c,B,tau)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
