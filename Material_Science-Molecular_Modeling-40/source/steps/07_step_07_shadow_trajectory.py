"""
Initialization and finite shadow trajectory with physical diagnostics.

Physical multipoles are relaxed at each recorded auxiliary state. Intrinsic quadrupoles alter the response and forces, while the total dipole remains the first moment about the fixed origin.

Returns
-------
float array (steps+1,12)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def shadow_trajectory(R, velocity, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, steps, tol, max_rank):
    """Initialize and propagate a finite deterministic block-retaining shadow trajectory.

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
steps is an integer (not bool) >=0. Initial G must be positive definite.

Returns
-------
record : float array (steps+1,12)
    At initial R, minimize the REGULAR energy e.T c+0.5*c.T G*c
    with global charge constraint. Fill ALL H history rows with this
    initial c0. Calculate J0 from shadow_response at this state and
    retain K0=J0^(-1) for the WHOLE trajectory; never refresh it.
    Call shadow_step exactly steps times. Record the initial state and
    each completed step, using the RELAXED c at that state's R and x.
    Columns: [time, shadow_energy, nuclear_kinetic_energy,
              total_energy, ||c-x||_2, sum(q), D_x, D_y, D_z,
              preceding_step_kernel_rank, preceding_step_final_error,
              ||theta||_2].
    D=sum_i(q_i*R_i+p_i) is the total physical dipole about the fixed
    coordinate origin; intrinsic quadrupoles have zero charge and dipole
    moments and do not enter D directly. Kinetic energy is sum_i masses_i*|velocity_i|^2/2.
    At row 0, rank and final_error are 0; a rank-zero step also has
    final_error 0. Extended DOFs have no kinetic contribution in this
    mass-zero model. No equilibration iterations or thermostat are added.
    steps=0 returns only the initialized row. Do not mutate inputs.

Raises
------
ValueError
    For any invalid dynamics parameter described by shadow_step; if steps
    is not an integer >=0; if initial G is not positive definite/J0 is
    singular; or if a later retained block/J/K0 violates the stated
    positive-definiteness or singular-value conditions."""
    return np.zeros((0, 12))

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

def _oracle_shadow_trajectory(R, velocity, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, steps, tol, max_rank):
    """Initialize and propagate a finite deterministic block-retaining shadow trajectory.

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
steps is an integer (not bool) >=0. Initial G must be positive definite.

Returns
-------
record : float array (steps+1,12)
    At initial R, minimize the REGULAR energy e.T c+0.5*c.T G*c
    with global charge constraint. Fill ALL H history rows with this
    initial c0. Calculate J0 from shadow_response at this state and
    retain K0=J0^(-1) for the WHOLE trajectory; never refresh it.
    Call shadow_step exactly steps times. Record the initial state and
    each completed step, using the RELAXED c at that state's R and x.
    Columns: [time, shadow_energy, nuclear_kinetic_energy,
              total_energy, ||c-x||_2, sum(q), D_x, D_y, D_z,
              preceding_step_kernel_rank, preceding_step_final_error,
              ||theta||_2].
    D=sum_i(q_i*R_i+p_i) is the total physical dipole about the fixed
    coordinate origin; intrinsic quadrupoles have zero charge and dipole
    moments and do not enter D directly. Kinetic energy is sum_i masses_i*|velocity_i|^2/2.
    At row 0, rank and final_error are 0; a rank-zero step also has
    final_error 0. Extended DOFs have no kinetic contribution in this
    mass-zero model. No equilibration iterations or thermostat are added.
    steps=0 returns only the initialized row. Do not mutate inputs.

Raises
------
ValueError
    For any invalid dynamics parameter described by shadow_step; if steps
    is not an integer >=0; if initial G is not positive definite/J0 is
    singular; or if a later retained block/J/K0 violates the stated
    positive-definiteness or singular-value conditions."""
    R, u, alpha, gamma = _sites(R, u, alpha, gamma)
    n = len(R)
    R = R.copy()
    velocity = _real(velocity, 'velocity', (n, 3)).copy()
    chi, masses = (_real(chi, 'chi', (n,)), _real(masses, 'masses', (n,)))
    if np.any(masses <= 0):
        raise ValueError('masses must be positive')
    charge = _scalar(charge, 'charge')
    dt, kappa, diss, coeff, tol, max_rank = _controls(dt, kappa, diss, coeff, tol, max_rank, 9 * n)
    steps = _integer(steps, 'steps', 0)
    G, _ = _oracle_multipole_operator(R, u, alpha, gamma)
    w = np.concatenate((np.ones(n), np.zeros(8 * n)))
    b = np.concatenate((-chi, np.zeros(8 * n), [charge]))
    augmented = np.zeros((9 * n + 1, 9 * n + 1))
    augmented[:-1, :-1] = G
    augmented[:-1, -1] = augmented[-1, :-1] = w
    try:
        np.linalg.cholesky(G)
        initial_c = np.linalg.solve(augmented, b)[:-1]
        _, J0 = _oracle_shadow_response(G, chi, charge, initial_c, groups)
        K0 = np.linalg.solve(J0, np.eye(9 * n))
    except np.linalg.LinAlgError as exc:
        raise ValueError('Initial G must be positive definite and J0 nonsingular') from exc
    history = np.repeat(initial_c[None, :], len(coeff), axis=0)
    record = np.empty((steps + 1, 12))
    rank, errors = (0, np.zeros(0))
    for k in range(steps + 1):
        energy, _, c, _ = _oracle_shadow_state(R, u, alpha, gamma, chi, charge, history[0], groups)
        kinetic = float(0.5 * np.sum(masses[:, None] * velocity ** 2))
        dipole = c[:n] @ R + c[n:4 * n].reshape(n, 3).sum(axis=0)
        record[k] = np.r_[k * dt, energy, kinetic, energy + kinetic, np.linalg.norm(c - history[0]), np.sum(c[:n]), dipole, rank, errors[-1] if len(errors) else 0.0, np.linalg.norm(c[4 * n:])]
        if k < steps:
            R, velocity, history, _, _, rank, errors = _oracle_shadow_step(R, velocity, history, K0, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, tol, max_rank)
    return record

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               "cfg={'R': [[0, 0, 0], [0.62, 0.17, -0.13], [1.7, -0.25, 0.4], [-0.45, 1.35, 0.65], [0.8, 0.55, "
               "1.65]], 'velocity': [[0.18, -0.05, 0.08], [-0.12, 0.1, -0.04], [0.04, -0.11, 0.05], [-0.08, "
               "0.06, -0.12], [0.025, 0.045, 0.055]], 'u': [1.8, 1.2, 1.6, 1.4, 1.1], 'alpha': [0.22, 0.19, "
               "0.25, 0.17, 0.21], 'chi': [-0.5, 0.35, 0.15, -0.25, 0.45], 'masses': [12, 14, 16, 10, 19], "
               "'charge': 0.3, 'dt': 0.02, 'kappa': 1.82, 'diss': 0.018, 'coeff': [-6, 14, -8, -3, 4, -1], "
               "'steps': 40, 'tol': 1e-06, 'max_rank': 2, 'groups': [0, 0, 1, 1, 2]}\n"
               '\n'
               'cfg["gamma"]=[.08,.06,.09,.07,.05] if len(cfg["u"])==5 else [.07]*len(cfg["u"])\n'
               '# The step-1 correction acts on an exactly relaxed initial state, so its beta is\n'
               '# round-off near the 1e-14 cutoff; exclude only that row\'s rank and error.\n'
               'def _without_step1_kernel_stats(record):\n'
               '    record = np.asarray(record, dtype=float)\n'
               '    keep = np.ones(record.shape, dtype=bool)\n'
               '    if record.shape[0] > 1:\n'
               '        keep[1, 9:11] = False\n'
               '    return np.concatenate((np.asarray(record.shape, dtype=float), record[keep]))\n',
      'call': '_without_step1_kernel_stats(shadow_trajectory(**cfg))',
      'gold_call': '_without_step1_kernel_stats(_oracle_shadow_trajectory(**cfg))'},
     {'setup': 'import numpy as np\n'
               "cfg={'R': [[0, 0, 0], [1.1, -0.2, 0.4], [-0.4, 1.3, 0.5]], 'velocity': [[0.01, -0.02, 0.01], "
               "[-0.015, 0.012, 0.005], [0.006, 0.004, -0.01]], 'u': [1.4, 1.1, 1.7], 'alpha': [0.2, 0.25, "
               "0.18], 'chi': [-0.3, 0.2, 0.1], 'masses': [12, 16, 14], 'charge': -0.2, 'groups': [0, 0, 1], "
               "'dt': 0.025, 'kappa': 1.6, 'diss': 0.012, 'coeff': [-2, 3, 0, -1], 'steps': 6, 'tol': 1e-07, "
               "'max_rank': 4}\n"
               '\n'
               'cfg["gamma"]=[.08,.06,.09,.07,.05] if len(cfg["u"])==5 else [.07]*len(cfg["u"])\n',
      'call': 'shadow_trajectory(**cfg)',
      'gold_call': '_oracle_shadow_trajectory(**cfg)'},
     {'setup': 'import numpy as np\n'
               "cfg={'R': [[0, 0, 0], [1.1, -0.2, 0.4], [-0.4, 1.3, 0.5]], 'velocity': [[0.01, -0.02, 0.01], "
               "[-0.015, 0.012, 0.005], [0.006, 0.004, -0.01]], 'u': [1.4, 1.1, 1.7], 'alpha': [0.2, 0.25, "
               "0.18], 'chi': [-0.3, 0.2, 0.1], 'masses': [12, 16, 14], 'charge': -0.2, 'groups': [0, 0, 1], "
               "'dt': 0.025, 'kappa': 1.6, 'diss': 0.012, 'coeff': [-2, 3, 0, -1], 'steps': 0, 'tol': 1e-07, "
               "'max_rank': 4}\n"
               '\n'
               'cfg["gamma"]=[.08,.06,.09,.07,.05] if len(cfg["u"])==5 else [.07]*len(cfg["u"])\n',
      'call': 'shadow_trajectory(**cfg)',
      'gold_call': '_oracle_shadow_trajectory(**cfg)'},
     {'setup': 'import numpy as np\n'
               "cfg={'R': [[0.2, -0.1, 0.3]], 'velocity': [[0.03, -0.02, 0.01]], 'u': [1.3], 'alpha': [0.25], "
               "'chi': [-0.2], 'masses': [12], 'charge': 0.4, 'groups': [4], 'dt': 0.03, 'kappa': 1.6, 'diss': "
               "0.01, 'coeff': [1, -2, 1], 'steps': 4, 'tol': 1e-07, 'max_rank': 4}\n"
               '\n'
               'cfg["gamma"]=[.08,.06,.09,.07,.05] if len(cfg["u"])==5 else [.07]*len(cfg["u"])\n',
      'call': 'shadow_trajectory(**cfg)',
      'gold_call': '_oracle_shadow_trajectory(**cfg)'},
     {'setup': 'import numpy as np\n'
               "cfg={'R': [[0, 0, 0], [1.1, -0.2, 0.4], [-0.4, 1.3, 0.5]], 'velocity': [[0.01, -0.02, 0.01], "
               "[-0.015, 0.012, 0.005], [0.006, 0.004, -0.01]], 'u': [1.4, 1.1, 1.7], 'alpha': [0.2, 0.25, "
               "0.18], 'chi': [-0.3, 0.2, 0.1], 'masses': [12, 16, 14], 'charge': -0.2, 'groups': [0, 1, 2], "
               "'dt': 0.025, 'kappa': 1.6, 'diss': 0.0, 'coeff': [-2, 3, 0, -1], 'steps': 3, 'tol': 1e-07, "
               "'max_rank': 4}\n"
               '\n'
               'cfg["gamma"]=[.08,.06,.09,.07,.05] if len(cfg["u"])==5 else [.07]*len(cfg["u"])\n',
      'call': 'shadow_trajectory(**cfg)',
      'gold_call': '_oracle_shadow_trajectory(**cfg)'},
     {'setup': "cfg={'R': [[0, 0, 0], [1.1, -0.2, 0.4], [-0.4, 1.3, 0.5]], 'velocity': [[0.01, -0.02, 0.01], "
               "[-0.015, 0.012, 0.005], [0.006, 0.004, -0.01]], 'u': [1.4, 1.1, 1.7], 'alpha': [0.2, 0.25, "
               "0.18], 'chi': [-0.3, 0.2, 0.1], 'masses': [12, 16, 14], 'charge': -0.2, 'groups': [0, 0, 1], "
               "'dt': 0.025, 'kappa': 1.6, 'diss': 0.012, 'coeff': [-2, 3, 0, -1], 'steps': -1, 'tol': 1e-07, "
               "'max_rank': 4}\n"
               '\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        shadow_trajectory(**cfg)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_shadow_trajectory(**cfg)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'cfg["gamma"]=[.08,.06,.09,.07,.05] if len(cfg["u"])==5 else [.07]*len(cfg["u"])\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               "cfg={'R': [[0, 0, 0], [0.62, 0.17, -0.13], [1.7, -0.25, 0.4], [-0.45, 1.35, 0.65], [0.8, 0.55, "
               "1.65]], 'velocity': [[0.18, -0.05, 0.08], [-0.12, 0.1, -0.04], [0.04, -0.11, 0.05], [-0.08, "
               "0.06, -0.12], [0.025, 0.045, 0.055]], 'u': [1.8, 1.2, 1.6, 1.4, 1.1], 'alpha': [0.22, 0.19, "
               "0.25, 0.17, 0.21], 'chi': [-0.5, 0.35, 0.15, -0.25, 0.45], 'masses': [12, 14, 16, 10, 19], "
               "'charge': 0.3, 'dt': 0.02, 'kappa': 1.82, 'diss': 0.018, 'coeff': [-6, 14, -8, -3, 4, -1], "
               "'steps': 40, 'tol': 1e-06, 'max_rank': 2, 'groups': [0, 0, 1, 1, 2]}\n"
               '\n'
               'cfg["gamma"]=[.08,.06,.09,.07,.05] if len(cfg["u"])==5 else [.07]*len(cfg["u"])\n'
               '\n'
               'angle=.37\n'
               'axis=np.array([1.,2.,-1.]);axis/=np.linalg.norm(axis)\n'
               'K=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])\n'
               'O=np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)\n'
               'cfg["R"]=(np.asarray(cfg["R"])@O.T).tolist()\n'
               'cfg["velocity"]=(np.asarray(cfg["velocity"])@O.T).tolist()\n'
               '# The step-1 correction acts on an exactly relaxed initial state, so its beta is\n'
               '# round-off near the 1e-14 cutoff; exclude only that row\'s rank and error.\n'
               'def _without_step1_kernel_stats(record):\n'
               '    record = np.asarray(record, dtype=float)\n'
               '    keep = np.ones(record.shape, dtype=bool)\n'
               '    if record.shape[0] > 1:\n'
               '        keep[1, 9:11] = False\n'
               '    return np.concatenate((np.asarray(record.shape, dtype=float), record[keep]))\n',
      'call': '_without_step1_kernel_stats(shadow_trajectory(**cfg))',
      'gold_call': '_without_step1_kernel_stats(_oracle_shadow_trajectory(**cfg))'}]
