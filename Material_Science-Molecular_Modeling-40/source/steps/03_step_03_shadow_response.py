"""
Global constrained relaxation of complete fragment blocks and the residual Jacobian.

The electronic energy is quadratic within retained fragments and linearized between fragments. A single global charge constraint couples their otherwise separate stationary responses.

Returns
-------
arrays (9N,) and (9N,9N)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def shadow_response(G, chi, charge, x, groups):
    """Equilibrate a partially linearized multipole energy and return its residual Jacobian.

Parameters
----------
G : finite real symmetric array of shape (9N,9N)
    Symmetry tolerance is absolute 1e-12, relative zero; N>=1.
chi : finite real array (N,)
charge : finite real scalar
x : finite real array (9N,)
    Extended multipoles ordered as q[N], p[3N], theta[5N]; p and theta
    are atom-major. Theta uses the traceless basis of multipole_operator. Its charge
    sum need not equal charge.
groups : length-N array of nonnegative integer-valued labels
    Equal labels identify one fragment; labels need not be consecutive.

Returns
-------
(c, J) : arrays (9N,) and (9N,9N)
    Retain in S every G entry connecting DOFs on atoms in the SAME
    fragment, including all charge, dipole and quadrupole couplings; set all other entries
    of S to zero. L=G-S. For e=[chi,0_(8N)] define
      E(c,x)=e.T c + (1/2)c.T S c + (c-x/2).T L x.
    c is the unique minimizer over w.T c=charge, where w=[ones(N),
    zeros(8N)]. Each retained fragment block of S is positive definite.
    J is d(c(x)-x)/dx at fixed G, chi, charge and groups.
    The constraint applies only to charges, and globally across all
    fragments. Do not impose separate fragment charge constraints.
    Do not include the Lagrange multiplier in either returned array.
    Do not mutate inputs.

Raises
------
ValueError
    If an input is not finite and real; shapes or symmetry violate the
    conditions above; a label is negative/nonintegral; or a retained
    fragment block is not positive definite."""
    return (np.zeros(0), np.zeros((0, 0)))

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

def _matrix_inputs(G, chi, charge, x, groups):
    chi = _real(chi, 'chi')
    if chi.ndim != 1 or len(chi) < 1:
        raise ValueError('chi must be a nonempty vector')
    n = len(chi)
    G = _real(G, 'G', (9 * n, 9 * n))
    x = _real(x, 'x', (9 * n,))
    charge = _scalar(charge, 'charge')
    if not np.allclose(G, G.T, atol=1e-12, rtol=0):
        raise ValueError('G must be symmetric within absolute tolerance 1e-12')
    groups = _real(groups, 'groups', (n,))
    if np.any(groups < 0) or np.any(groups != np.floor(groups)):
        raise ValueError('groups must contain nonnegative integer-valued labels')
    labels = np.concatenate((groups, np.repeat(groups, 3), np.repeat(groups, 5)))
    mask = labels[:, None] == labels[None, :]
    S = G * mask
    return (G, chi, charge, x, labels, mask, S)

def _oracle_shadow_response(G, chi, charge, x, groups):
    """Equilibrate a partially linearized multipole energy and return its residual Jacobian.

Parameters
----------
G : finite real symmetric array of shape (9N,9N)
    Symmetry tolerance is absolute 1e-12, relative zero; N>=1.
chi : finite real array (N,)
charge : finite real scalar
x : finite real array (9N,)
    Extended multipoles ordered as q[N], p[3N], theta[5N]; p and theta
    are atom-major. Theta uses the traceless basis of multipole_operator. Its charge
    sum need not equal charge.
groups : length-N array of nonnegative integer-valued labels
    Equal labels identify one fragment; labels need not be consecutive.

Returns
-------
(c, J) : arrays (9N,) and (9N,9N)
    Retain in S every G entry connecting DOFs on atoms in the SAME
    fragment, including all charge, dipole and quadrupole couplings; set all other entries
    of S to zero. L=G-S. For e=[chi,0_(8N)] define
      E(c,x)=e.T c + (1/2)c.T S c + (c-x/2).T L x.
    c is the unique minimizer over w.T c=charge, where w=[ones(N),
    zeros(8N)]. Each retained fragment block of S is positive definite.
    J is d(c(x)-x)/dx at fixed G, chi, charge and groups.
    The constraint applies only to charges, and globally across all
    fragments. Do not impose separate fragment charge constraints.
    Do not include the Lagrange multiplier in either returned array.
    Do not mutate inputs.

Raises
------
ValueError
    If an input is not finite and real; shapes or symmetry violate the
    conditions above; a label is negative/nonintegral; or a retained
    fragment block is not positive definite."""
    G, chi, charge, x, labels, mask, S = _matrix_inputs(G, chi, charge, x, groups)
    n = len(chi)
    e = np.concatenate((chi, np.zeros(8 * n)))
    w = np.concatenate((np.ones(n), np.zeros(8 * n)))
    inv_s = np.zeros_like(G)
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        block = S[np.ix_(ids, ids)]
        try:
            np.linalg.cholesky(block)
            inv_s[np.ix_(ids, ids)] = np.linalg.solve(block, np.eye(len(ids)))
        except np.linalg.LinAlgError as exc:
            raise ValueError('Each retained fragment block must be positive definite') from exc
    v = inv_s @ w
    denom = float(w @ v)
    P = inv_s - np.outer(v, v) / denom
    L = G - S
    c = -P @ (e + L @ x) + v * (charge / denom)
    J = -P @ L - np.eye(9 * n)
    return (c, J)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(904)\n'
               'B=rng.normal(size=(27,27))*.2\n'
               'G=B@B.T+np.diag(np.linspace(1.2,2.9,27))\n'
               'chi=np.array([-.4,.3,.1]);charge=.7\n'
               'x=np.linspace(-.3,.6,27)\n'
               'groups=np.array([8,3,8])\n',
      'call': 'shadow_response(G,chi,charge,x,groups)',
      'gold_call': '_oracle_shadow_response(G,chi,charge,x,groups)'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(904)\n'
               'B=rng.normal(size=(27,27))*.2\n'
               'G=B@B.T+np.diag(np.linspace(1.2,2.9,27))\n'
               'chi=np.array([-.4,.3,.1]);charge=.7\n'
               'x=np.linspace(-.3,.6,27)\n'
               'groups=np.array([8,3,8])\n'
               'groups=np.array([0,0,0])\n',
      'call': 'shadow_response(G,chi,charge,x,groups)',
      'gold_call': '_oracle_shadow_response(G,chi,charge,x,groups)'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(904)\n'
               'B=rng.normal(size=(27,27))*.2\n'
               'G=B@B.T+np.diag(np.linspace(1.2,2.9,27))\n'
               'chi=np.array([-.4,.3,.1]);charge=.7\n'
               'x=np.linspace(-.3,.6,27)\n'
               'groups=np.array([8,3,8])\n'
               'groups=np.array([0,1,2])\n'
               'charge=-.6\n',
      'call': 'shadow_response(G,chi,charge,x,groups)',
      'gold_call': '_oracle_shadow_response(G,chi,charge,x,groups)'},
     {'setup': 'import numpy as np\n'
               'G=np.diag([2.,4.,4.,4.,12.,12.,12.,12.,12.])\n'
               'chi=np.array([.7]);x=np.array([-.2,.1,-.3,.2,.02,-.03,.01,-.01,.025]);groups=[12]\n',
      'call': 'shadow_response(G,chi,.25,x,groups)',
      'gold_call': '_oracle_shadow_response(G,chi,.25,x,groups)'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(904)\n'
               'B=rng.normal(size=(27,27))*.2\n'
               'G=B@B.T+np.diag(np.linspace(1.2,2.9,27))\n'
               'chi=np.array([-.4,.3,.1]);charge=.7\n'
               'x=np.linspace(-.3,.6,27)\n'
               'groups=np.array([8,3,8])\n'
               'G[0,1]+=1\n'
               '\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        shadow_response(G,chi,charge,x,groups)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_shadow_response(G,chi,charge,x,groups)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(904)\n'
               'B=rng.normal(size=(27,27))*.2\n'
               'G=B@B.T+np.diag(np.linspace(1.2,2.9,27))\n'
               'chi=np.array([-.4,.3,.1]);charge=.7\n'
               'x=np.linspace(-.3,.6,27)\n'
               'groups=np.array([8,3,8])\n'
               'G[0,0]=-1\n'
               '\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        shadow_response(G,chi,charge,x,groups)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_shadow_response(G,chi,charge,x,groups)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(904)\n'
               'B=rng.normal(size=(27,27))*.2\n'
               'G=B@B.T+np.diag(np.linspace(1.2,2.9,27))\n'
               'chi=np.array([-.4,.3,.1]);charge=.7\n'
               'x=np.linspace(-.3,.6,27)\n'
               'groups=np.array([8,3,8])\n'
               'groups=[0,.5,1]\n'
               '\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        shadow_response(G,chi,charge,x,groups)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_shadow_response(G,chi,charge,x,groups)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
