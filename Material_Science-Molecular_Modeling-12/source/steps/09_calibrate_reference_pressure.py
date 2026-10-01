"""
Calibrate the reference particle pressure pi00 of the LTh model against the structure-aware equation of state: find the value of pi00, all other entries of params held fixed, at which the pressure of the self-consistent HNC structure at (T, rho) on the grid (n_grid, dr) equals p_target, solving the scalar equation to |P - p_target| < 1e-13 with any root finder (the root is simple and lies close to the input pi00). Return the calibrated pi00, the pressure at the input pi00, and the corrected density n and the internal energy per particle U/N of the calibrated model. Raise ValueError if p_target is not positive.

The source's first-estimate parameters neglect the structure; simulations at those parameters give a pressure a few percent below the target, and the source fine-tunes pi00 by adding the difference. The HNC route predicts a pressure above the target at the liquid reference state, so its calibrated pi00 lies below the first estimate, and by more the smaller the cutoff; at the supercritical state the sign of the correction reverses. Because the pressure responds non-linearly to pi00 through the structure, the exact calibration differs from the one-step estimate target minus prediction.

Returns
-------
A (4,) float64 array [calibrated pi00, P at the input pi00, n of the calibrated model, U/N of the calibrated model] in reduced units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def calibrate_reference_pressure(temperature, rho, rcut, fcut, params, p_target, n_grid, dr):
    """Calibrate the reference particle pressure pi00 of the LTh model against the structure-
    aware equation of state: find the value of pi00, all other entries of params held fixed,
    at which the pressure of the self-consistent HNC structure at (T, rho) on the grid
    (n_grid, dr) equals p_target, solving the scalar equation to |P - p_target| < 1e-13 with
    any root finder (the root is simple and lies close to the input pi00). A (4,) float64
    array [calibrated pi00, P at the input pi00, n of the calibrated model, U/N of the
    calibrated model] in reduced units."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pressure(temperature, rho, rcut, fcut, params, n_grid, dr):
    g = _oracle_self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr)
    return _oracle_eos_state(g, temperature, rho, rcut, fcut, params, dr)

def _oracle_calibrate_reference_pressure(temperature, rho, rcut, fcut, params, p_target, n_grid, dr):
    params = [float(x) for x in params]
    if p_target <= 0:
        raise ValueError("target pressure must be positive")
    def resid(pi00):
        q = list(params); q[2] = pi00
        return float(_pressure(temperature, rho, rcut, fcut, q, n_grid, dr)[2]) - p_target
    p0 = params[2]
    f0 = resid(p0)
    p_init = f0 + p_target
    p1 = p0 - f0                                        # unit-slope first guess
    f1 = resid(p1)
    for it in range(100):
        if abs(f1) < 1.0e-13:
            break
        if f1 == f0:
            raise RuntimeError("secant stalled")
        p2 = p1 - f1*(p1 - p0)/(f1 - f0)
        p0, f0 = p1, f1
        p1, f1 = p2, resid(p2)
    else:
        raise RuntimeError("reference-pressure calibration did not converge")
    q = list(params); q[2] = p1
    st = _pressure(temperature, rho, rcut, fcut, q, n_grid, dr)
    return np.array([p1, p_init, float(st[1]), float(st[6])])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n# normal: benchmark liquid calibration\nparams = np.array([0.01105407273373585, 1.0, 0.11605782726626415, 1.0111776310806926, 29.34480813105209, 11.008324924443508])\ntemperature = 0.01105407273373585\nrho = 1.0\nrcut = 2.1564\nfcut = 1.33\np_target = 0.1271119\nn_grid = 100\ndr = 0.025\n',
         'call': 'calibrate_reference_pressure(temperature, rho, rcut, fcut, params, p_target, n_grid, dr)',
         'gold_call': '_oracle_calibrate_reference_pressure(temperature, rho, rcut, fcut, params, p_target, n_grid, dr)'},
        {'setup': 'import numpy as np\n# boundary: minimum permitted HNC grid length n_grid = 8\nparams = np.array([0.1, 1.0, 0.0, 1000.0, 0.0, 1.0])\ntemperature = 0.1\nrho = 0.1\nrcut = 0.5\nfcut = 1.0\np_target = 0.00794165813986\nn_grid = 8\ndr = 0.06666666666666667\n',
         'call': 'calibrate_reference_pressure(temperature, rho, rcut, fcut, params, p_target, n_grid, dr)',
         'gold_call': '_oracle_calibrate_reference_pressure(temperature, rho, rcut, fcut, params, p_target, n_grid, dr)'},
        {'setup': 'import numpy as np\n# edge: dilute strongly structured state with a target pressure displaced from the initial EOS value\nparams = np.array([0.1, 1.0, 0.0, 10.0, 0.0, 1.0])\ntemperature = 0.1\nrho = 0.1\nrcut = 1.0\nfcut = 2.0\np_target = 0.00732378684218\nn_grid = 24\ndr = 0.05\n',
         'call': 'calibrate_reference_pressure(temperature, rho, rcut, fcut, params, p_target, n_grid, dr)',
         'gold_call': '_oracle_calibrate_reference_pressure(temperature, rho, rcut, fcut, params, p_target, n_grid, dr)'},
    ]
