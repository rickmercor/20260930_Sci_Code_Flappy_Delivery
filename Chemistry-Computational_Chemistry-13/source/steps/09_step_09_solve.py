"""
Integrate the complete single-Hessian dipole correlation on the stated grid.

The reduced spectrum is a damped Fourier transform of the nuclear autocorrelation. The initial zero-point energy sets the absorption frequency origin.

Returns
-------
float — normalized spectral ordinate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve(data):
    r"""Return the normalized finite-grid spectral ordinate

    J = dt/(3*pi*C(0)) * Re sum_j w_j*C(t_j)
        * exp(i*(omega + E0/hbar)*t_j - eta*t_j**2),
    where t_j = j*dt, j=0,...,nsteps.

    Simpson weights: 1 at endpoints, 4 at odd interior indices,
    2 at even interior indices. No additional frequency prefactor,
    tail correction, or clipping.

    Use earlier steps in this order:
    initial_frame; potential_jet at q_reference; centre_action with
    zero initial momentum and action; width_path with the constant
    reference Hessian; dipole_coefficients; correlation_path.
    correlation_path calls overlap_generator and fock_overlap.

    The potential uses coordinates q-origin.
    The dipole uses coordinates q-q_initial.

    Parameters
    ----------
    data : dict
        Required fields:
        mass, ground_hessian: real SPD arrays, shape (D,D).
        q_initial, origin, q_reference: real arrays, shape (D,).
        potential_powers: nonnegative integer-valued array, shape (R,D).
        potential_coefficients: real monomial coefficients, shape (R,).
        dipole_powers: nonnegative integer-valued array, shape (U,D),
            U>=1, total degree of each row <=6.
        dipole_coefficients: real or complex array, shape (U,3).
        hbar, dt: finite positive scalars.
        eta: finite nonnegative scalar.
        omega: finite real scalar.
        nsteps: even integer >=2.

        All arrays are finite. Dimensions and canonical conditions
        follow the earlier step contracts. The time grid must resolve
        the determinant phase as specified in width_path.

    Returns
    -------
    ordinate : float
        Normalized spectral ordinate as a native Python float.

    Raises
    ------
    ValueError
        For a non-dict or missing fields; invalid spectral scalars
        or grid; invalid arrays under the earlier step contracts;
        or zero initial dipole-state norm.
    """
    return ordinate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve(data):
    import numpy as np

    required = {
        "mass",
        "ground_hessian",
        "q_initial",
        "origin",
        "potential_powers",
        "potential_coefficients",
        "q_reference",
        "dipole_powers",
        "dipole_coefficients",
        "hbar",
        "dt",
        "nsteps",
        "omega",
        "eta",
    }

    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError("Missing benchmark fields")

    hbar = data["hbar"]
    dt = data["dt"]
    nsteps = data["nsteps"]
    omega = data["omega"]
    eta = data["eta"]

    if (
        not all(np.isfinite(v) for v in (hbar, dt, omega, eta))
        or hbar <= 0
        or dt <= 0
        or eta < 0
        or not isinstance(nsteps, (int, np.integer))
        or nsteps < 2
        or nsteps % 2 != 0
    ):
        raise ValueError("Invalid spectral grid")

    Q0, P0, energy = _oracle_initial_frame(
        data["mass"],
        data["ground_hessian"],
        hbar,
    )

    _, _, reference_hessian = _oracle_potential_jet(
        data["q_reference"],
        data["origin"],
        data["potential_powers"],
        data["potential_coefficients"],
    )

    qs, ps, actions = _oracle_centre_action(
        data["mass"],
        data["q_initial"],
        np.zeros(Q0.shape[0]),
        data["origin"],
        data["potential_powers"],
        data["potential_coefficients"],
        dt,
        nsteps,
    )

    times = dt * np.arange(nsteps + 1)

    Qs, Ps, logdets = _oracle_width_path(
        data["mass"],
        reference_hessian,
        Q0,
        P0,
        times,
    )

    labels, coefficients = _oracle_dipole_coefficients(
        Q0,
        data["dipole_powers"],
        data["dipole_coefficients"],
        hbar,
    )

    correlation = _oracle_correlation_path(
        qs,
        ps,
        actions,
        Qs,
        Ps,
        logdets,
        labels,
        coefficients,
        hbar,
    )

    norm = float(correlation[0].real)
    if norm <= 0:
        raise ValueError("The initial dipole state has zero norm")

    integrand = correlation * np.exp(
        1j * (omega + energy / hbar) * times
        - eta * times ** 2
    )

    weights = np.ones(nsteps + 1)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0

    return float(
        (dt / 3.0)
        * np.real(weights @ integrand)
        / (np.pi * norm)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'case_1',
            "setup": """import numpy as np
data={'mass': [[1.2, 0.12, -0.08], [0.12, 1.5, 0.1], [-0.08, 0.1, 0.9]], 'ground_hessian': [[1.4, 0.23, -0.14], [0.23, 2.1, 0.17], [-0.14, 0.17, 0.95]], 'q_initial': [-0.35, 0.22, 0.18], 'origin': [0.1, -0.15, 0.08], 'q_reference': [0.06, -0.04, 0.12], 'potential_powers': [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3], [2, 1, 0], [1, 1, 1], [4, 0, 0], [0, 4, 0], [0, 0, 4], [2, 2, 0], [0, 2, 2], [2, 0, 2]], 'potential_coefficients': [1.6, 0.5, 0.925, 0.65, 0.21, -0.16, 0.27, 0.06, -0.04, 0.03, 0.08, -0.055, 0.018, 0.024, 0.02, 0.012, 0.016, 0.01], 'dipole_powers': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 2], [3, 0, 0], [1, 1, 1], [0, 2, 1]], 'dipole_coefficients': [[0.32, -0.18, 0.11], [0.75, 0.12, -0.24], [-0.28, 0.6, 0.16], [0.18, -0.32, 0.55], [0.2, -0.1, 0.08], [-0.17, 0.22, -0.12], [0.09, -0.16, 0.13], [-0.11, 0.07, 0.2], [0.065, -0.04, 0.015], [-0.045, 0.035, 0.055], [0.03, -0.05, 0.025]], 'hbar': 0.7, 'dt': 0.04, 'nsteps': 300, 'omega': 2.2, 'eta': 0.045}
""",
            "call": 'solve(data)',
            "gold_call": '_oracle_solve(data)',
        },
        {
            "name": 'case_2',
            "setup": """import numpy as np
data={'mass': [[1.2, 0.12, -0.08], [0.12, 1.5, 0.1], [-0.08, 0.1, 0.9]], 'ground_hessian': [[1.4, 0.23, -0.14], [0.23, 2.1, 0.17], [-0.14, 0.17, 0.95]], 'q_initial': [-0.35, 0.22, 0.18], 'origin': [0.1, -0.15, 0.08], 'q_reference': [0.06, -0.04, 0.12], 'potential_powers': [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3], [2, 1, 0], [1, 1, 1], [4, 0, 0], [0, 4, 0], [0, 0, 4], [2, 2, 0], [0, 2, 2], [2, 0, 2]], 'potential_coefficients': [1.6, 0.5, 0.925, 0.65, 0.21, -0.16, 0.27, 0.06, -0.04, 0.03, 0.08, -0.055, 0.018, 0.024, 0.02, 0.012, 0.016, 0.01], 'dipole_powers': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 2], [3, 0, 0], [1, 1, 1], [0, 2, 1]], 'dipole_coefficients': [[0.32, -0.18, 0.11], [0.75, 0.12, -0.24], [-0.28, 0.6, 0.16], [0.18, -0.32, 0.55], [0.2, -0.1, 0.08], [-0.17, 0.22, -0.12], [0.09, -0.16, 0.13], [-0.11, 0.07, 0.2], [0.065, -0.04, 0.015], [-0.045, 0.035, 0.055], [0.03, -0.05, 0.025]], 'hbar': 0.7, 'dt': 0.04, 'nsteps': 300, 'omega': 2.2, 'eta': 0.045}
data["nsteps"]=2
""",
            "call": 'solve(data)',
            "gold_call": '_oracle_solve(data)',
        },
        {
            "name": 'case_3',
            "setup": """import numpy as np
data={'mass': [[1.2, 0.12, -0.08], [0.12, 1.5, 0.1], [-0.08, 0.1, 0.9]], 'ground_hessian': [[1.4, 0.23, -0.14], [0.23, 2.1, 0.17], [-0.14, 0.17, 0.95]], 'q_initial': [-0.35, 0.22, 0.18], 'origin': [0.1, -0.15, 0.08], 'q_reference': [0.06, -0.04, 0.12], 'potential_powers': [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3], [2, 1, 0], [1, 1, 1], [4, 0, 0], [0, 4, 0], [0, 0, 4], [2, 2, 0], [0, 2, 2], [2, 0, 2]], 'potential_coefficients': [1.6, 0.5, 0.925, 0.65, 0.21, -0.16, 0.27, 0.06, -0.04, 0.03, 0.08, -0.055, 0.018, 0.024, 0.02, 0.012, 0.016, 0.01], 'dipole_powers': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 2], [3, 0, 0], [1, 1, 1], [0, 2, 1]], 'dipole_coefficients': [[0.32, -0.18, 0.11], [0.75, 0.12, -0.24], [-0.28, 0.6, 0.16], [0.18, -0.32, 0.55], [0.2, -0.1, 0.08], [-0.17, 0.22, -0.12], [0.09, -0.16, 0.13], [-0.11, 0.07, 0.2], [0.065, -0.04, 0.015], [-0.045, 0.035, 0.055], [0.03, -0.05, 0.025]], 'hbar': 0.7, 'dt': 0.04, 'nsteps': 300, 'omega': 2.2, 'eta': 0.045}
data["dipole_powers"]=[[0,0,0]]
data["dipole_coefficients"]=[[1.,0.,0.]]
data["nsteps"]=40
""",
            "call": 'solve(data)',
            "gold_call": '_oracle_solve(data)',
        },
        {
            "name": 'case_4',
            "setup": """import numpy as np
data={'mass': [[1.2, 0.12, -0.08], [0.12, 1.5, 0.1], [-0.08, 0.1, 0.9]], 'ground_hessian': [[1.4, 0.23, -0.14], [0.23, 2.1, 0.17], [-0.14, 0.17, 0.95]], 'q_initial': [-0.35, 0.22, 0.18], 'origin': [0.1, -0.15, 0.08], 'q_reference': [0.06, -0.04, 0.12], 'potential_powers': [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3], [2, 1, 0], [1, 1, 1], [4, 0, 0], [0, 4, 0], [0, 0, 4], [2, 2, 0], [0, 2, 2], [2, 0, 2]], 'potential_coefficients': [1.6, 0.5, 0.925, 0.65, 0.21, -0.16, 0.27, 0.06, -0.04, 0.03, 0.08, -0.055, 0.018, 0.024, 0.02, 0.012, 0.016, 0.01], 'dipole_powers': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 2], [3, 0, 0], [1, 1, 1], [0, 2, 1]], 'dipole_coefficients': [[0.32, -0.18, 0.11], [0.75, 0.12, -0.24], [-0.28, 0.6, 0.16], [0.18, -0.32, 0.55], [0.2, -0.1, 0.08], [-0.17, 0.22, -0.12], [0.09, -0.16, 0.13], [-0.11, 0.07, 0.2], [0.065, -0.04, 0.015], [-0.045, 0.035, 0.055], [0.03, -0.05, 0.025]], 'hbar': 0.7, 'dt': 0.04, 'nsteps': 300, 'omega': 2.2, 'eta': 0.045}
data["hbar"]=1.0
data["omega"]=1.7
data["nsteps"]=60
""",
            "call": 'solve(data)',
            "gold_call": '_oracle_solve(data)',
        },
        {
            "name": 'case_5',
            "setup": """import numpy as np
data={'mass': [[1.2, 0.12, -0.08], [0.12, 1.5, 0.1], [-0.08, 0.1, 0.9]], 'ground_hessian': [[1.4, 0.23, -0.14], [0.23, 2.1, 0.17], [-0.14, 0.17, 0.95]], 'q_initial': [-0.35, 0.22, 0.18], 'origin': [0.1, -0.15, 0.08], 'q_reference': [0.06, -0.04, 0.12], 'potential_powers': [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3], [2, 1, 0], [1, 1, 1], [4, 0, 0], [0, 4, 0], [0, 0, 4], [2, 2, 0], [0, 2, 2], [2, 0, 2]], 'potential_coefficients': [1.6, 0.5, 0.925, 0.65, 0.21, -0.16, 0.27, 0.06, -0.04, 0.03, 0.08, -0.055, 0.018, 0.024, 0.02, 0.012, 0.016, 0.01], 'dipole_powers': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 2], [3, 0, 0], [1, 1, 1], [0, 2, 1]], 'dipole_coefficients': [[0.32, -0.18, 0.11], [0.75, 0.12, -0.24], [-0.28, 0.6, 0.16], [0.18, -0.32, 0.55], [0.2, -0.1, 0.08], [-0.17, 0.22, -0.12], [0.09, -0.16, 0.13], [-0.11, 0.07, 0.2], [0.065, -0.04, 0.015], [-0.045, 0.035, 0.055], [0.03, -0.05, 0.025]], 'hbar': 0.7, 'dt': 0.04, 'nsteps': 300, 'omega': 2.2, 'eta': 0.045}
data["nsteps"]=3

def _status(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError("Expected ValueError")
""",
            "call": '_status(solve, data)',
            "gold_call": '_status(_oracle_solve, data)',
        },
    ]
