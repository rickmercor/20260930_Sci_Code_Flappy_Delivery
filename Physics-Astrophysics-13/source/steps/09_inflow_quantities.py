"""
Step 09: inflow field strength, inflow density and Alfven speed.

Convention (pinned)



Displace every supplied sample point by plus delta and by minus delta along the

unit vector (0, 1, 0), giving two displaced polylines. For each of the two

signs form the step-08 curve average of the magnetic field, as a VECTOR, and of

the scaled density, both over the displaced points; then average the two signs.

Because the displacement is a constant vector, the two displaced polylines have

the same segment lengths as the supplied one, so the arclength weights are

unchanged.



  B_in   the magnitude of the resulting mean vector, in Gauss;

  rho_s  the resulting mean scaled density, i.e. the mass density divided by

         the reference density 1.0e-14 g/cm^3, so it is of order unity;

  V_A    B_in divided by the square root of 4 pi times the PHYSICAL inflow

         density, that is 4 pi rho_s 1.0e-14. Gaussian (CGS) units, so

         the factor 4 pi is present and V_A comes out in cm/s.



Both fields come from the step-01 public function and the averaging from the

step-08 public function.



Inputs



points : array_like

    Shape (N, 3), N >= 2, all finite.

delta : float

    Displacement magnitude, strictly positive and finite.



Returns



tuple of float

    (B_in, rho_s, V_A).

Returns
-------
A tuple of three floats: the inflow field strength in Gauss, the scaled inflow density in units of 1.0e-14 g/cm^3, and the Alfven speed in cm/s.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def inflow_quantities(points, delta):
    """Return the inflow field strength, scaled inflow density and Alfven speed.

    Parameters
    ----------
    points : array_like
        Shape (N, 3), N >= 2, all finite.
    delta : float
        Strictly positive finite displacement along the unit vector (0, 1, 0).

    Returns
    -------
    B_in : float
        Magnitude of the sign-averaged curve average of the magnetic field.
    rho_s : float
        Sign-averaged curve average of the scaled density.
    V_A : float
        B_in divided by the square root of 4 * pi * rho_s * 1.0e-14.

    Raises
    ------
    ValueError
        If points is not a finite (N, 3) array with N >= 2, if delta is not a
        positive finite real, or if the resulting scaled density is not
        positive.
    """
    return B_in, rho_s, V_A  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

RHO_REF = 1.0e-14


def _oracle_inflow_quantities(points, delta):
    """Reference implementation of inflow_quantities (deterministic).

    Composes the step-01 and step-08 oracle functions directly, so this
    reference value never depends on a submitted implementation.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("points must have shape (N, 3) with N >= 2")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")
    try:
        delta = float(delta)
    except (TypeError, ValueError):
        raise ValueError("delta must be a real number")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be a positive finite real")

    shift = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    b_means = []
    r_means = []
    for sign in (1.0, -1.0):
        q = p + sign * delta * shift
        b_avg, _ = _oracle_line_average(
            p, np.asarray(_oracle_field_samples(q, "B"), dtype=np.float64))
        r_avg, _ = _oracle_line_average(
            p, np.asarray(_oracle_field_samples(q, "rho"), dtype=np.float64))
        b_means.append(np.asarray(b_avg, dtype=np.float64))
        r_means.append(float(r_avg))

    b_vec = 0.5 * (b_means[0] + b_means[1])
    rho_s = 0.5 * (r_means[0] + r_means[1])
    if not (rho_s > 0.0):
        raise ValueError("inflow density must be positive")
    b_in = float(np.linalg.norm(b_vec))
    v_a = b_in / math.sqrt(4.0 * math.pi * rho_s * RHO_REF)
    return b_in, rho_s, v_a

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for inflow_quantities."""
    return [
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\nzs = np.linspace(-0.8, 0.8, 81)\npts = _oracle_quasi_x_line(zs, sd)\n', 'call': 'inflow_quantities(pts, 0.05)', 'gold_call': '_oracle_inflow_quantities(pts, 0.05)'},
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\nzs = np.linspace(-0.8, 0.8, 81)\npts = _oracle_quasi_x_line(zs, sd)\n', 'call': 'inflow_quantities(pts, 0.4)', 'gold_call': '_oracle_inflow_quantities(pts, 0.4)'},
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\npts = _oracle_quasi_x_line(np.array([-0.5, -0.04]), sd)\n', 'call': 'inflow_quantities(pts, 0.25)', 'gold_call': '_oracle_inflow_quantities(pts, 0.25)'},
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\npts = _oracle_quasi_x_line(np.array([-0.8, -0.7, -0.3, -0.04, 0.6]), sd)\n', 'call': 'inflow_quantities(pts, 0.01)', 'gold_call': '_oracle_inflow_quantities(pts, 0.01)'},
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\npts = _oracle_quasi_x_line(np.array([-0.5, -0.04]), sd)\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(inflow_quantities, pts, 0.0)', 'gold_call': '_status(_oracle_inflow_quantities, pts, 0.0)'},
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\npts = _oracle_quasi_x_line(np.array([-0.04]), sd)\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(inflow_quantities, pts, 0.05)', 'gold_call': '_status(_oracle_inflow_quantities, pts, 0.05)'},
    ]
