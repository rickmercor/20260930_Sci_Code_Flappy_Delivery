"""
Finite-window unnormalized physical-dipole correlation

The time average uses the initial relaxed physical dipole as its reference and all completed trajectory samples. It is not a normalized correlation coefficient.

Returns
-------
native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve(R, velocity, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, steps, tol, max_rank):
    """Return the finite-window, time-averaged initial-dipole correlation.

Parameters
----------
Same arguments and validation domain as shadow_trajectory, including
groups, steps>=0, finite dt>0 and fixed K0 initialization.
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

Returns
-------
answer : native Python float
    Obtain shadow_trajectory and take C_k=D(k*dt).dot(D(0)) from its
    RELAXED physical dipoles (columns 6:9). For steps M>0 return the
    composite trapezoidal integral of C(t), divided by M*dt:
      (C_0/2 + sum(C_1,...,C_(M-1)) + C_M/2)/M.
    For M=0 return C_0, the zero-window limit. Do not normalize by
    |D(0)|^2, discard the initial sample, or use extended x as c.
    Units are squared atomic dipole units. No input is mutated.

Raises
------
ValueError
    If steps is not an integer >=0; any array has an invalid shape or
    nonfinite/nonreal entry; a group label is negative/nonintegral;
    masses/u/alpha/gamma/dt are nonpositive; kappa is outside (0,4); diss<0;
    coeff has length <2 or sum outside absolute 1e-12 of zero; tol is
    outside [0,1); max_rank is not an integer in [1,9N]; initial G is
    not positive definite or J0 is singular; a retained fragment block
    is not positive definite; or a J/K0 passed to krylov_action has
    smallest/largest singular-value ratio <=1e-14."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import math
import numpy as np
from scipy.special import gammainc, gammaln
import math
import numpy as np

def _oracle_solve(R, velocity, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, steps, tol, max_rank):
    """Return the finite-window, time-averaged initial-dipole correlation.

Parameters
----------
Same arguments and validation domain as shadow_trajectory, including
groups, steps>=0, finite dt>0 and fixed K0 initialization.
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

Returns
-------
answer : native Python float
    Obtain shadow_trajectory and take C_k=D(k*dt).dot(D(0)) from its
    RELAXED physical dipoles (columns 6:9). For steps M>0 return the
    composite trapezoidal integral of C(t), divided by M*dt:
      (C_0/2 + sum(C_1,...,C_(M-1)) + C_M/2)/M.
    For M=0 return C_0, the zero-window limit. Do not normalize by
    |D(0)|^2, discard the initial sample, or use extended x as c.
    Units are squared atomic dipole units. No input is mutated.

Raises
------
ValueError
    If steps is not an integer >=0; any array has an invalid shape or
    nonfinite/nonreal entry; a group label is negative/nonintegral;
    masses/u/alpha/gamma/dt are nonpositive; kappa is outside (0,4); diss<0;
    coeff has length <2 or sum outside absolute 1e-12 of zero; tol is
    outside [0,1); max_rank is not an integer in [1,9N]; initial G is
    not positive definite or J0 is singular; a retained fragment block
    is not positive definite; or a J/K0 passed to krylov_action has
    smallest/largest singular-value ratio <=1e-14."""
    record = _oracle_shadow_trajectory(R, velocity, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, steps, tol, max_rank)
    correlation = record[:, 6:9] @ record[0, 6:9]
    if len(correlation) == 1:
        return float(correlation[0])
    return float((0.5 * correlation[0] + np.sum(correlation[1:-1]) + 0.5 * correlation[-1]) / (len(correlation) - 1))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
cfg = {
    'R': [[0, 0, 0], [0.62, 0.17, -0.13], [1.7, -0.25, 0.4],
          [-0.45, 1.35, 0.65], [0.8, 0.55, 1.65]],
    'velocity': [[0.18, -0.05, 0.08], [-0.12, 0.1, -0.04],
                 [0.04, -0.11, 0.05], [-0.08, 0.06, -0.12],
                 [0.025, 0.045, 0.055]],
    'u': [1.8, 1.2, 1.6, 1.4, 1.1],
    'alpha': [0.22, 0.19, 0.25, 0.17, 0.21],
    'gamma': [0.08, 0.06, 0.09, 0.07, 0.05],
    'chi': [-0.5, 0.35, 0.15, -0.25, 0.45],
    'masses': [12, 14, 16, 10, 19],
    'charge': 0.3,
    'groups': [0, 0, 1, 1, 2],
    'dt': 0.02,
    'kappa': 1.82,
    'diss': 0.018,
    'coeff': [-6, 14, -8, -3, 4, -1],
    'steps': 40,
    'tol': 1e-6,
    'max_rank': 2
}
""",
            "call": "solve(**cfg)",
            "gold_call": "_oracle_solve(**cfg)"
        },
        {
            "setup": """
import numpy as np
cfg = {
    'R': [[0, 0, 0], [1.1, -0.2, 0.4], [-0.4, 1.3, 0.5]],
    'velocity': [[0.01, -0.02, 0.01], [-0.015, 0.012, 0.005],
                 [0.006, 0.004, -0.01]],
    'u': [1.4, 1.1, 1.7],
    'alpha': [0.2, 0.25, 0.18],
    'gamma': [0.07, 0.07, 0.07],
    'chi': [-0.3, 0.2, 0.1],
    'masses': [12, 16, 14],
    'charge': -0.2,
    'groups': [0, 0, 1],
    'dt': 0.025,
    'kappa': 1.6,
    'diss': 0.012,
    'coeff': [-2, 3, 0, -1],
    'steps': 6,
    'tol': 1e-7,
    'max_rank': 4
}
""",
            "call": "solve(**cfg)",
            "gold_call": "_oracle_solve(**cfg)"
        },
        {
            "setup": """
import numpy as np
cfg = {
    'R': [[0, 0, 0], [1.1, -0.2, 0.4], [-0.4, 1.3, 0.5]],
    'velocity': [[0.01, -0.02, 0.01], [-0.015, 0.012, 0.005],
                 [0.006, 0.004, -0.01]],
    'u': [1.4, 1.1, 1.7],
    'alpha': [0.2, 0.25, 0.18],
    'gamma': [0.07, 0.07, 0.07],
    'chi': [-0.3, 0.2, 0.1],
    'masses': [12, 16, 14],
    'charge': -0.2,
    'groups': [0, 0, 1],
    'dt': 0.025,
    'kappa': 1.6,
    'diss': 0.012,
    'coeff': [-2, 3, 0, -1],
    'steps': 0,
    'tol': 1e-7,
    'max_rank': 4
}
""",
            "call": "solve(**cfg)",
            "gold_call": "_oracle_solve(**cfg)"
        },
        {
            "setup": """
import numpy as np
cfg = {
    'R': [[0.2, -0.1, 0.3]],
    'velocity': [[0.03, -0.02, 0.01]],
    'u': [1.3],
    'alpha': [0.25],
    'gamma': [0.07],
    'chi': [-0.2],
    'masses': [12],
    'charge': 0.4,
    'groups': [4],
    'dt': 0.03,
    'kappa': 1.6,
    'diss': 0.01,
    'coeff': [1, -2, 1],
    'steps': 4,
    'tol': 1e-7,
    'max_rank': 4
}
""",
            "call": "solve(**cfg)",
            "gold_call": "_oracle_solve(**cfg)"
        },
        {
            "setup": """
import numpy as np
cfg = {
    'R': [[0, 0, 0], [1.1, -0.2, 0.4], [-0.4, 1.3, 0.5]],
    'velocity': [[0.01, -0.02, 0.01], [-0.015, 0.012, 0.005],
                 [0.006, 0.004, -0.01]],
    'u': [1.4, 1.1, 1.7],
    'alpha': [0.2, 0.25, 0.18],
    'gamma': [0.07, 0.07, 0.07],
    'chi': [-0.3, 0.2, 0.1],
    'masses': [12, 16, 14],
    'charge': -0.2,
    'groups': [0, 1, 2],
    'dt': 0.025,
    'kappa': 1.6,
    'diss': 0.0,
    'coeff': [-2, 3, 0, -1],
    'steps': 3,
    'tol': 1e-7,
    'max_rank': 4
}
""",
            "call": "solve(**cfg)",
            "gold_call": "_oracle_solve(**cfg)"
        },
        {
            "setup": """
cfg = {
    'R': [[0, 0, 0], [1.1, -0.2, 0.4], [-0.4, 1.3, 0.5]],
    'velocity': [[0.01, -0.02, 0.01], [-0.015, 0.012, 0.005],
                 [0.006, 0.004, -0.01]],
    'u': [1.4, 1.1, 1.7],
    'alpha': [0.2, 0.25, 0.18],
    'gamma': [0.07, 0.07, 0.07],
    'chi': [-0.3, 0.2, 0.1],
    'masses': [12, 16, 14],
    'charge': -0.2,
    'groups': [0, 0, 1],
    'dt': 0.025,
    'kappa': 1.6,
    'diss': 0.012,
    'coeff': [-2, 3, 0, -1],
    'steps': -1,
    'tol': 1e-7,
    'max_rank': 4
}

def run_model():
    try:
        solve(**cfg)
    except ValueError:
        return 1
    return 0

def run_gold():
    try:
        _oracle_solve(**cfg)
    except ValueError:
        return 1
    return 0
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """
import numpy as np
cfg = {
    'R': [[0, 0, 0], [0.62, 0.17, -0.13], [1.7, -0.25, 0.4],
          [-0.45, 1.35, 0.65], [0.8, 0.55, 1.65]],
    'velocity': [[0.18, -0.05, 0.08], [-0.12, 0.1, -0.04],
                 [0.04, -0.11, 0.05], [-0.08, 0.06, -0.12],
                 [0.025, 0.045, 0.055]],
    'u': [1.8, 1.2, 1.6, 1.4, 1.1],
    'alpha': [0.22, 0.19, 0.25, 0.17, 0.21],
    'gamma': [0.08, 0.06, 0.09, 0.07, 0.05],
    'chi': [-0.5, 0.35, 0.15, -0.25, 0.45],
    'masses': [12, 14, 16, 10, 19],
    'charge': 0.3,
    'groups': [0, 0, 1, 1, 2],
    'dt': 0.02,
    'kappa': 1.82,
    'diss': 0.018,
    'coeff': [-6, 14, -8, -3, 4, -1],
    'steps': 40,
    'tol': 1e-6,
    'max_rank': 2
}
angle = 0.37
axis = np.array([1., 2., -1.])
axis /= np.linalg.norm(axis)
K = np.array([
    [0., -axis[2], axis[1]],
    [axis[2], 0., -axis[0]],
    [-axis[1], axis[0], 0.]
])
O = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
cfg['R'] = (np.asarray(cfg['R']) @ O.T).tolist()
cfg['velocity'] = (np.asarray(cfg['velocity']) @ O.T).tolist()
""",
            "call": "solve(**cfg)",
            "gold_call": "_oracle_solve(**cfg)"
        }
    ]
