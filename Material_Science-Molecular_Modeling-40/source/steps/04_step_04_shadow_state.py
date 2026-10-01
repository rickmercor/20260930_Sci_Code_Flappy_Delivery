"""
Relaxed shadow energy and fixed-auxiliary nuclear forces.

Constrained electronic stationarity removes response derivatives from the first nuclear derivative. Geometry-dependent interactions in both retained and complementary blocks still contribute to the force.

Returns
-------
scalar and arrays (N,3), (9N,), (9N,9N)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def shadow_state(R, u, alpha, gamma, chi, charge, x, groups):
    """Evaluate the relaxed shadow potential, nuclear forces and electronic response.

Parameters
----------
R is a finite real (N,3) array, N>=1. u, alpha and gamma are finite
positive (N,) arrays of charge hardness, dipole polarizability and
quadrupole polarizability. The 9N ordering is charges, atom-major
Cartesian dipoles, then atom-major five-component quadrupoles in the
orthonormal traceless basis specified by multipole_operator.
All units are atomic units. Boundaries are nonperiodic. Site parameters
do not depend on coordinates. Inputs are never mutated.
chi is finite real (N,), charge is a finite real scalar, x is finite
real (9N,); groups obeys shadow_response's fragment-label contract.

Returns
-------
(energy, forces, c, J) : numerical tuple
    Shapes (), (N,3), (9N,), (9N,9N). Assemble G from the Gaussian
    multipole model; obtain S,L,c and J using shadow_response's
    definition. energy=E(c(x),x) with e=[chi,zeros(8N)]. There is
    no charge-independent potential. forces[i,k] is minus the partial
    derivative of this relaxed energy with respect to R[i,k] at FIXED x.
    Account for coordinate dependence of BOTH S and L. Fragment
    membership remains fixed under differentiation. These are forces
    of the shadow potential, not forces of the fully equilibrated
    regular Born-Oppenheimer potential. Return J for residual c(x)-x.

Raises
------
ValueError
    For any invalid shape, nonfinite/nonreal input, nonpositive u/alpha/gamma,
    invalid group label, or non-positive-definite retained block, as
    specified by multipole_operator and shadow_response."""
    return (0.0, np.zeros((0, 3)), np.zeros(0), np.zeros((0, 0)))

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

def _oracle_shadow_state(R, u, alpha, gamma, chi, charge, x, groups):
    """Evaluate the relaxed shadow potential, nuclear forces and electronic response.

Parameters
----------
R is a finite real (N,3) array, N>=1. u, alpha and gamma are finite
positive (N,) arrays of charge hardness, dipole polarizability and
quadrupole polarizability. The 9N ordering is charges, atom-major
Cartesian dipoles, then atom-major five-component quadrupoles in the
orthonormal traceless basis specified by multipole_operator.
All units are atomic units. Boundaries are nonperiodic. Site parameters
do not depend on coordinates. Inputs are never mutated.
chi is finite real (N,), charge is a finite real scalar, x is finite
real (9N,); groups obeys shadow_response's fragment-label contract.

Returns
-------
(energy, forces, c, J) : numerical tuple
    Shapes (), (N,3), (9N,), (9N,9N). Assemble G from the Gaussian
    multipole model; obtain S,L,c and J using shadow_response's
    definition. energy=E(c(x),x) with e=[chi,zeros(8N)]. There is
    no charge-independent potential. forces[i,k] is minus the partial
    derivative of this relaxed energy with respect to R[i,k] at FIXED x.
    Account for coordinate dependence of BOTH S and L. Fragment
    membership remains fixed under differentiation. These are forces
    of the shadow potential, not forces of the fully equilibrated
    regular Born-Oppenheimer potential. Return J for residual c(x)-x.

Raises
------
ValueError
    For any invalid shape, nonfinite/nonreal input, nonpositive u/alpha/gamma,
    invalid group label, or non-positive-definite retained block, as
    specified by multipole_operator and shadow_response."""
    G, dG = _oracle_multipole_operator(R, u, alpha, gamma)
    G, chi, charge, x, labels, mask, S = _matrix_inputs(G, chi, charge, x, groups)
    c, J = _oracle_shadow_response(G, chi, charge, x, groups)
    n = len(chi)
    L = G - S
    energy = float(chi @ c[:n] + 0.5 * c @ S @ c + (c - 0.5 * x) @ L @ x)
    dS = dG * mask
    dL = dG - dS
    forces = -0.5 * np.einsum('a,ijab,b->ij', c, dS, c)
    forces -= np.einsum('a,ijab,b->ij', c - 0.5 * x, dL, x)
    return (energy, forces, c, J)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'chi=np.array([-.5,.25,.1]);charge=.3\n'
               'x=np.linspace(-.2,.35,27)\n'
               'groups=np.array([0,0,1])\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'shadow_state(R,u,alpha,gamma,chi,charge,x,groups)',
      'gold_call': '_oracle_shadow_state(R,u,alpha,gamma,chi,charge,x,groups)'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'chi=np.array([-.5,.25,.1]);charge=.3\n'
               'x=np.linspace(-.2,.35,27)\n'
               'groups=np.array([0,0,1])\n'
               'groups=np.array([4,4,4])\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'shadow_state(R,u,alpha,gamma,chi,charge,x,groups)',
      'gold_call': '_oracle_shadow_state(R,u,alpha,gamma,chi,charge,x,groups)'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'chi=np.array([-.5,.25,.1]);charge=.3\n'
               'x=np.linspace(-.2,.35,27)\n'
               'groups=np.array([0,0,1])\n'
               'groups=np.array([0,1,2])\n'
               'x[3:]=0\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'shadow_state(R,u,alpha,gamma,chi,charge,x,groups)',
      'gold_call': '_oracle_shadow_state(R,u,alpha,gamma,chi,charge,x,groups)'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[.1,.2,.3]]);u=[1.3];alpha=[.2];chi=[-.2];x=np.array([.6,.1,.2,-.3,.04,-.02,.03,.01,-.01])\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'shadow_state(R,u,alpha,gamma,chi,.5,x,[0])',
      'gold_call': '_oracle_shadow_state(R,u,alpha,gamma,chi,.5,x,[0])'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'chi=np.array([-.5,.25,.1]);charge=.3\n'
               'x=np.linspace(-.2,.35,27)\n'
               'groups=np.array([0,0,1])\n'
               'x=x[:-1]\n'
               '\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        shadow_state(R,u,alpha,gamma,chi,charge,x,groups)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_shadow_state(R,u,alpha,gamma,chi,charge,x,groups)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
