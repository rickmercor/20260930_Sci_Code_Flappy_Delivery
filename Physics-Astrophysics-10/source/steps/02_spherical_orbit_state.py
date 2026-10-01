"""
Map spherical orbit coordinates to a Cartesian Kepler state.

For longitude $\phi$, latitude $\theta$, distance $r$, radial speed $v_r$, tangential speed $v_\Omega$, inclination $i$, and $\kappa\in\{-1,+1\}$, use



$$

\hat r=(\cos\theta\cos\phi,\cos\theta\sin\phi,\sin\theta),\quad

\hat A=(-\sin\phi,\cos\phi,0),\quad \hat D=\hat r\times\hat A,

$$



$$

\psi=\kappa\arccos\!\left(\frac{\cos i}{\cos\theta}\right),\qquad

\mathbf r_0=r\hat r,\qquad

\mathbf v_0=v_r\hat r+v_\Omega(\cos\psi\,\hat A+\sin\psi\,\hat D).

$$



Reject nonfinite values, $r\le0$, $v_\Omega\le0$, $\kappa\notin\{-1,+1\}$, or $|\cos i/\cos\theta|>1$ apart from floating-point roundoff.  The graded instance uses $(\phi,\theta,r,v_r,v_\Omega,i,\kappa)=(0.640,0.061,42.60,-0.00014,0.00418,0.0614,-1)$ in radians, AU, and AU/day.

Returns
-------
`np.ndarray` of shape `(7,)`, ordered as Cartesian position in AU, Cartesian velocity in AU/day, and the signed angle $\psi$ in radians.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Map spherical orbit coordinates to a Cartesian Kepler state."""

import numpy as np


def spherical_orbit_state(phi: float, theta: float, distance_au: float, radial_speed_au_day: float, tangential_speed_au_day: float, inclination_rad: float, kappa: int) -> np.ndarray:
    """Return the Cartesian state implied by a signed spherical orbit.

    Parameters are longitude, latitude, distance, radial and tangential speed,
    inclination, and the ascending/descending branch sign ``kappa``.
    """
    return np.array([], dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_spherical_orbit_state(phi: float, theta: float, distance_au: float, radial_speed_au_day: float, tangential_speed_au_day: float, inclination_rad: float, kappa: int) -> np.ndarray:
    """Construct the state using the spherical A--D tangent basis."""
    import numpy as np

    values = np.array([phi, theta, distance_au, radial_speed_au_day, tangential_speed_au_day, inclination_rad], dtype=float)
    if not np.all(np.isfinite(values)) or distance_au <= 0 or tangential_speed_au_day <= 0 or kappa not in (-1, 1):
        raise ValueError("invalid spherical-orbit parameters")
    ratio = np.cos(inclination_rad) / np.cos(theta)
    if abs(ratio) > 1.0 + 1e-12:
        raise ValueError("inclination is incompatible with latitude")
    psi = float(kappa * np.arccos(np.clip(ratio, -1.0, 1.0)))
    rhat = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)])
    ahat = np.array([-np.sin(phi), np.cos(phi), 0.0])
    dhat = np.cross(rhat, ahat)
    position = distance_au * rhat
    velocity = radial_speed_au_day * rhat + tangential_speed_au_day * (np.cos(psi) * ahat + np.sin(psi) * dhat)
    return np.concatenate([position, velocity, [psi]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and branch cases."""
    return [
        {"setup": "pass", "call": "spherical_orbit_state(0.4, 0.05, 41.0, -0.0002, 0.0041, 0.15, -1)", "gold_call": "_oracle_spherical_orbit_state(0.4, 0.05, 41.0, -0.0002, 0.0041, 0.15, -1)"},
        {"setup": "pass", "call": "spherical_orbit_state(0.0, 0.0, 38.0, 0.0, 0.0038, 0.0, 1)", "gold_call": "_oracle_spherical_orbit_state(0.0, 0.0, 38.0, 0.0, 0.0038, 0.0, 1)"},
        {"setup": "pass", "call": "spherical_orbit_state(-0.7, -0.08, 52.0, 0.0001, 0.0032, 0.23, 1)", "gold_call": "_oracle_spherical_orbit_state(-0.7, -0.08, 52.0, 0.0001, 0.0032, 0.23, 1)"},
    ]
