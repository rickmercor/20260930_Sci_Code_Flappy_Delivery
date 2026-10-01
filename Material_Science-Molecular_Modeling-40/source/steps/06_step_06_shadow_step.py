"""
One coupled nuclear and auxiliary multipole update.

Nuclear forces and auxiliary restoring actions are evaluated at the old state. The second nuclear half kick uses the new geometry and auxiliary multipoles.

Returns
-------
(N,3), (N,3), (H,9N), scalar, array (9N,), integer and array (rank,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def shadow_step(R, velocity, history, K0, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, tol, max_rank):
    """Advance nuclear and extended multipole states by one coupled shadow-MD step.

Parameters
----------
R is a finite real (N,3) array, N>=1. u, alpha and gamma are finite
positive (N,) arrays of charge hardness, dipole polarizability and
quadrupole polarizability. The 9N ordering is charges, atom-major
Cartesian dipoles, then atom-major five-component quadrupoles in the
orthonormal traceless basis specified by multipole_operator.
All units are atomic units. Boundaries are nonperiodic. Site parameters
do not depend on coordinates. Inputs are never mutated.
velocity is finite (N,3); chi and masses are finite (N,),
with masses strictly positive. charge is a finite scalar. groups is a
length-N array of nonnegative integer-valued fragment labels (equal
labels, including nonconsecutive labels, identify the same fragment).
dt>0, 0<kappa<4, diss>=0 are finite scalars. coeff is a finite vector of
length H>=2 whose sum is zero within absolute tolerance 1e-12.
tol is finite, 0<=tol<1; max_rank is an integer in [1,9N].
At each kernel call use the noise, SVD, rank, and breakdown conventions
of krylov_action; do not replace the truncated action by a full solve.
history has shape (H,9N), newest first: [x(t),x(t-dt),...].
K0 is a finite real (9N,9N) matrix satisfying krylov_action's
singular-value condition. The history entries are independent extended
states, not relaxed multipoles; their charge sums need not equal charge.

Returns
-------
(R_new, velocity_new, history_new, energy_new, c_new, rank, errors)
    Evaluate shadow_state at OLD R and history[0] and obtain force,
    c and J. Let z be krylov_action(J,K0,c-history[0],tol,max_rank).
    Perform the nuclear velocity half kick and position drift with
    old forces and masses. At the same old state use
      x_new=2*history[0]-history[1]-kappa*z+diss*(coeff@history).
    Place x_new at the front of history_new and drop the oldest row.
    Evaluate shadow_state at R_new and x_new, then complete the velocity
    half kick using NEW forces. Keep history_new[0]=x_new; never replace
    it with c_new. Return new-state energy and c, but OLD-state kernel
    rank and error history. kappa already equals dt^2*omega^2.
    Shapes: (N,3),(N,3),(H,9N),(),(9N,),(),(rank,).

Raises
------
ValueError
    For invalid shapes, nonfinite/nonreal entries, nonpositive masses,
    u, alpha, gamma or dt, kappa outside (0,4), diss<0, H<2, coefficient sum
    outside absolute 1e-12 of zero, invalid tol/max_rank/groups, or any
    retained block/J/K0 violating the conditions of shadow_state or
    krylov_action."""
    return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 0)), 0.0, np.zeros(0), 0, np.zeros(0))

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

def _sites(R, u, alpha, gamma):
    R = _real(R, 'R')
    if R.ndim != 2 or R.shape[1] != 3 or R.shape[0] < 1:
        raise ValueError('R must have shape (N,3), N>=1')
    n = len(R)
    u, alpha = (_real(u, 'u', (n,)), _real(alpha, 'alpha', (n,)))
    gamma = _real(gamma, 'gamma', (n,))
    if np.any(u <= 0) or np.any(alpha <= 0) or np.any(gamma <= 0):
        raise ValueError('u, alpha and gamma must be positive')
    return (R, u, alpha, gamma)

def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if value < minimum:
        raise ValueError(name + ' is too small')
    return int(value)

def _controls(dt, kappa, diss, coeff, tol, max_rank, dimension):
    dt = _scalar(dt, 'dt', positive=True)
    kappa = _scalar(kappa, 'kappa', positive=True)
    if kappa >= 4:
        raise ValueError('kappa must be less than 4')
    diss = _scalar(diss, 'diss', nonnegative=True)
    coeff = _real(coeff, 'coeff')
    if coeff.ndim != 1 or len(coeff) < 2:
        raise ValueError('coeff must have length at least 2')
    if abs(float(np.sum(coeff))) > 1e-12:
        raise ValueError('coeff must sum to zero within 1e-12')
    tol = _scalar(tol, 'tol', nonnegative=True)
    if tol >= 1:
        raise ValueError('tol must be less than 1')
    max_rank = _integer(max_rank, 'max_rank', 1)
    if max_rank > dimension:
        raise ValueError('max_rank exceeds the multipole dimension')
    return (dt, kappa, diss, coeff, tol, max_rank)

def _oracle_shadow_step(R, velocity, history, K0, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, tol, max_rank):
    """Advance nuclear and extended multipole states by one coupled shadow-MD step.

Parameters
----------
R is a finite real (N,3) array, N>=1. u, alpha and gamma are finite
positive (N,) arrays of charge hardness, dipole polarizability and
quadrupole polarizability. The 9N ordering is charges, atom-major
Cartesian dipoles, then atom-major five-component quadrupoles in the
orthonormal traceless basis specified by multipole_operator.
All units are atomic units. Boundaries are nonperiodic. Site parameters
do not depend on coordinates. Inputs are never mutated.
velocity is finite (N,3); chi and masses are finite (N,),
with masses strictly positive. charge is a finite scalar. groups is a
length-N array of nonnegative integer-valued fragment labels (equal
labels, including nonconsecutive labels, identify the same fragment).
dt>0, 0<kappa<4, diss>=0 are finite scalars. coeff is a finite vector of
length H>=2 whose sum is zero within absolute tolerance 1e-12.
tol is finite, 0<=tol<1; max_rank is an integer in [1,9N].
At each kernel call use the noise, SVD, rank, and breakdown conventions
of krylov_action; do not replace the truncated action by a full solve.
history has shape (H,9N), newest first: [x(t),x(t-dt),...].
K0 is a finite real (9N,9N) matrix satisfying krylov_action's
singular-value condition. The history entries are independent extended
states, not relaxed multipoles; their charge sums need not equal charge.

Returns
-------
(R_new, velocity_new, history_new, energy_new, c_new, rank, errors)
    Evaluate shadow_state at OLD R and history[0] and obtain force,
    c and J. Let z be krylov_action(J,K0,c-history[0],tol,max_rank).
    Perform the nuclear velocity half kick and position drift with
    old forces and masses. At the same old state use
      x_new=2*history[0]-history[1]-kappa*z+diss*(coeff@history).
    Place x_new at the front of history_new and drop the oldest row.
    Evaluate shadow_state at R_new and x_new, then complete the velocity
    half kick using NEW forces. Keep history_new[0]=x_new; never replace
    it with c_new. Return new-state energy and c, but OLD-state kernel
    rank and error history. kappa already equals dt^2*omega^2.
    Shapes: (N,3),(N,3),(H,9N),(),(9N,),(),(rank,).

Raises
------
ValueError
    For invalid shapes, nonfinite/nonreal entries, nonpositive masses,
    u, alpha, gamma or dt, kappa outside (0,4), diss<0, H<2, coefficient sum
    outside absolute 1e-12 of zero, invalid tol/max_rank/groups, or any
    retained block/J/K0 violating the conditions of shadow_state or
    krylov_action."""
    R, u, alpha, gamma = _sites(R, u, alpha, gamma)
    n = len(R)
    velocity = _real(velocity, 'velocity', (n, 3))
    masses = _real(masses, 'masses', (n,))
    chi = _real(chi, 'chi', (n,))
    if np.any(masses <= 0):
        raise ValueError('masses must be positive')
    dt, kappa, diss, coeff, tol, max_rank = _controls(dt, kappa, diss, coeff, tol, max_rank, 9 * n)
    history = _real(history, 'history', (len(coeff), 9 * n))
    K0 = _real(K0, 'K0', (9 * n, 9 * n))
    charge = _scalar(charge, 'charge')
    _, force, c, J = _oracle_shadow_state(R, u, alpha, gamma, chi, charge, history[0], groups)
    z, rank, errors = _oracle_krylov_action(J, K0, c - history[0], tol, max_rank)
    half_velocity = velocity + 0.5 * dt * force / masses[:, None]
    new_R = R + dt * half_velocity
    new_x = 2 * history[0] - history[1] - kappa * z + diss * (coeff @ history)
    new_history = np.concatenate((new_x[None, :], history[:-1]), axis=0)
    energy, new_force, new_c, _ = _oracle_shadow_state(new_R, u, alpha, gamma, chi, charge, new_x, groups)
    new_velocity = half_velocity + 0.5 * dt * new_force / masses[:, None]
    return (new_R, new_velocity, new_history, energy, new_c, rank, errors)

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
               'velocity=np.array([[.02,-.01,.03],[-.01,.02,.01],[.01,-.03,.02]])\n'
               'masses=np.array([12.,16.,14.])\n'
               'history=np.stack([x,x+.01*np.arange(27),x-.004*np.arange(27)])\n'
               'K0=-np.eye(27);dt=.03;kappa=1.6;diss=.01;coeff=np.array([-2.,3.,-1.]);tol=1e-5;max_rank=3\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'shadow_step(R,velocity,history,K0,u,alpha,gamma,chi,masses,charge,groups,dt,kappa,diss,coeff,tol,max_rank)',
      'gold_call': '_oracle_shadow_step(R,velocity,history,K0,u,alpha,gamma,chi,masses,charge,groups,dt,kappa,diss,coeff,tol,max_rank)'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'chi=np.array([-.5,.25,.1]);charge=.3\n'
               'x=np.linspace(-.2,.35,27)\n'
               'groups=np.array([0,0,1])\n'
               'velocity=np.array([[.02,-.01,.03],[-.01,.02,.01],[.01,-.03,.02]])\n'
               'masses=np.array([12.,16.,14.])\n'
               'history=np.stack([x,x+.01*np.arange(27),x-.004*np.arange(27)])\n'
               'K0=-np.eye(27);dt=.03;kappa=1.6;diss=.01;coeff=np.array([-2.,3.,-1.]);tol=1e-5;max_rank=3\n'
               'diss=0.;max_rank=27;tol=1e-11\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'shadow_step(R,velocity,history,K0,u,alpha,gamma,chi,masses,charge,groups,dt,kappa,diss,coeff,tol,max_rank)',
      'gold_call': '_oracle_shadow_step(R,velocity,history,K0,u,alpha,gamma,chi,masses,charge,groups,dt,kappa,diss,coeff,tol,max_rank)'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'chi=np.array([-.5,.25,.1]);charge=.3\n'
               'x=np.linspace(-.2,.35,27)\n'
               'groups=np.array([0,0,1])\n'
               'velocity=np.array([[.02,-.01,.03],[-.01,.02,.01],[.01,-.03,.02]])\n'
               'masses=np.array([12.,16.,14.])\n'
               'history=np.stack([x,x+.01*np.arange(27),x-.004*np.arange(27)])\n'
               'K0=-np.eye(27);dt=.03;kappa=1.6;diss=.01;coeff=np.array([-2.,3.,-1.]);tol=1e-5;max_rank=3\n'
               'groups=np.array([5,5,5]);history[:]=x\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'shadow_step(R,velocity,history,K0,u,alpha,gamma,chi,masses,charge,groups,dt,kappa,diss,coeff,tol,max_rank)',
      'gold_call': '_oracle_shadow_step(R,velocity,history,K0,u,alpha,gamma,chi,masses,charge,groups,dt,kappa,diss,coeff,tol,max_rank)'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'chi=np.array([-.5,.25,.1]);charge=.3\n'
               'x=np.linspace(-.2,.35,27)\n'
               'groups=np.array([0,0,1])\n'
               'velocity=np.array([[.02,-.01,.03],[-.01,.02,.01],[.01,-.03,.02]])\n'
               'masses=np.array([12.,16.,14.])\n'
               'history=np.stack([x,x+.01*np.arange(27),x-.004*np.arange(27)])\n'
               'K0=-np.eye(27);dt=.03;kappa=1.6;diss=.01;coeff=np.array([-2.,3.,-1.]);tol=1e-5;max_rank=3\n'
               'charge=-.4;groups=np.array([3,7,3]);max_rank=1\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'shadow_step(R,velocity,history,K0,u,alpha,gamma,chi,masses,charge,groups,dt,kappa,diss,coeff,tol,max_rank)',
      'gold_call': '_oracle_shadow_step(R,velocity,history,K0,u,alpha,gamma,chi,masses,charge,groups,dt,kappa,diss,coeff,tol,max_rank)'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'chi=np.array([-.5,.25,.1]);charge=.3\n'
               'x=np.linspace(-.2,.35,27)\n'
               'groups=np.array([0,0,1])\n'
               'velocity=np.array([[.02,-.01,.03],[-.01,.02,.01],[.01,-.03,.02]])\n'
               'masses=np.array([12.,16.,14.])\n'
               'history=np.stack([x,x+.01*np.arange(27),x-.004*np.arange(27)])\n'
               'K0=-np.eye(27);dt=.03;kappa=1.6;diss=.01;coeff=np.array([-2.,3.,-1.]);tol=1e-5;max_rank=3\n'
               'coeff=np.array([1.,1.,1.])\n'
               '\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        '
               'shadow_step(R,velocity,history,K0,u,alpha,gamma,chi,masses,charge,groups,dt,kappa,diss,coeff,tol,max_rank)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        '
               '_oracle_shadow_step(R,velocity,history,K0,u,alpha,gamma,chi,masses,charge,groups,dt,kappa,diss,coeff,tol,max_rank)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'chi=np.array([-.5,.25,.1]);charge=.3\n'
               'x=np.linspace(-.2,.35,27)\n'
               'groups=np.array([0,0,1])\n'
               'velocity=np.array([[.02,-.01,.03],[-.01,.02,.01],[.01,-.03,.02]])\n'
               'masses=np.array([12.,16.,14.])\n'
               'history=np.stack([x,x+.01*np.arange(27),x-.004*np.arange(27)])\n'
               'K0=-np.eye(27);dt=.03;kappa=1.6;diss=.01;coeff=np.array([-2.,3.,-1.]);tol=1e-5;max_rank=3\n'
               'masses[0]=0\n'
               '\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        '
               'shadow_step(R,velocity,history,K0,u,alpha,gamma,chi,masses,charge,groups,dt,kappa,diss,coeff,tol,max_rank)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        '
               '_oracle_shadow_step(R,velocity,history,K0,u,alpha,gamma,chi,masses,charge,groups,dt,kappa,diss,coeff,tol,max_rank)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
