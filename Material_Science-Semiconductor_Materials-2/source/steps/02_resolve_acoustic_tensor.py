"""
Contract an in-plane deformation-potential tensor into LA and TA band-edge shifts.

For in-plane acoustic phonons of a two-dimensional hexagonal crystal, the longitudinal (LA) and transverse (TA) polarizations are parallel and perpendicular to the phonon wave vector. Let the wave vector point along \(\theta\), measured counterclockwise from the \(x\) axis of the supplied in-plane deformation-potential tensor \((\Xi_{xx},\Xi_{yy},\Xi_{xy})\), and write \(s=\sin\theta\) and \(c=\cos\theta\). The band-edge shifts per unit strain amplitude are



\[

D_{LA}=\Xi_{xx}c^2+\Xi_{yy}s^2+2\Xi_{xy}sc,\qquad

D_{TA}=(\Xi_{yy}-\Xi_{xx})sc+\Xi_{xy}(c^2-s^2).

\]



The three tensor components are an input of this step, and the angle array may have any shape; it is flattened in C order.

Returns
-------
A NumPy array with shape `(2, n_angle)` whose first row is \(D_{LA}\) and whose second row is \(D_{TA}\), both in eV, in the flattened angle order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def resolve_acoustic_tensor(phonon_angle_rad: 'np.ndarray', dp_tensor_ev: 'Sequence[float]') -> 'np.ndarray':
    """Return the longitudinal and transverse band-edge shifts per unit strain.

    For a phonon wave vector at angle theta, measured counterclockwise from
    the x axis of the supplied tensor, the in-plane longitudinal and
    transverse acoustic polarizations of a two-dimensional hexagonal crystal
    are parallel and perpendicular to the wave vector. With s = sin(theta)
    and c = cos(theta), the contracted deformation potentials are

        D_LA = Xi_xx c^2 + Xi_yy s^2 + 2 Xi_xy s c,
        D_TA = (Xi_yy - Xi_xx) s c + Xi_xy (c^2 - s^2).

    Parameters
    ----------
    phonon_angle_rad : numpy.ndarray
        Phonon wave-vector directions theta in radians, measured
        counterclockwise from the tensor's x axis; any nonempty array of
        finite values, which is flattened in C order.
    dp_tensor_ev : sequence of float
        The three in-plane tensor components (Xi_xx, Xi_yy, Xi_xy) in eV.

    Returns
    -------
    numpy.ndarray
        Shape (2, n_angle), where n_angle is the number of supplied angles.
        Row 0 is D_LA and row 1 is D_TA at each direction in the flattened
        order, both in eV.

    Raises
    ------
    ValueError
        For an angle array that is empty or holds a value that is not a
        finite number, or a tensor that is not three finite numbers.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Contract an in-plane deformation-potential tensor into LA and TA band-edge shifts."""
import numpy as np

def _oracle_resolve_acoustic_tensor(phonon_angle_rad: 'np.ndarray', dp_tensor_ev: 'Sequence[float]') -> 'np.ndarray':
    """Reference longitudinal and transverse contraction of an in-plane tensor."""
    try:
        angle = np.asarray(phonon_angle_rad, dtype=float).ravel()
    except (TypeError, ValueError) as error:
        raise ValueError('phonon_angle_rad must hold finite numbers') from error
    if angle.size == 0 or not np.all(np.isfinite(angle)):
        raise ValueError('phonon_angle_rad must be a nonempty array of finite values')
    try:
        tensor = np.asarray(dp_tensor_ev, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('dp_tensor_ev must hold three finite numbers') from error
    if tensor.shape != (3,) or not np.all(np.isfinite(tensor)):
        raise ValueError('dp_tensor_ev must hold three finite numbers')
    xi_xx, xi_yy, xi_xy = tensor
    sine = np.sin(angle)
    cosine = np.cos(angle)
    longitudinal_ev = xi_xx * cosine ** 2 + xi_yy * sine ** 2 + 2.0 * xi_xy * sine * cosine
    transverse_ev = (xi_yy - xi_xx) * sine * cosine + xi_xy * (cosine ** 2 - sine ** 2)
    return np.vstack((longitudinal_ev, transverse_ev))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and rejection test specifications."""
    return [
        {
            "setup": "angle = np.linspace(0.1, 6.1, 7); params = (angle, (6.10, 3.90, 0.85))",
            "call": "resolve_acoustic_tensor(*params)",
            "gold_call": "_oracle_resolve_acoustic_tensor(*params)",
        },
        {
            "setup": "angle = np.array([0.0, np.pi/2, np.pi/6, 3*np.pi/2, np.pi/4]); params = (angle, [5.3, 5.3, 0.0])",
            "call": "resolve_acoustic_tensor(*params)",
            "gold_call": "_oracle_resolve_acoustic_tensor(*params)",
        },
        {
            "setup": "angle = np.array([[np.pi/2, 2.0, 3.5], [5*np.pi/6, -1.1, 0.3]]); params = (angle, np.array([-2.5, 4.4, -1.7]))",
            "call": "resolve_acoustic_tensor(*params)",
            "gold_call": "_oracle_resolve_acoustic_tensor(*params)",
        },
        {
            "setup": "angle = np.array([0.7]); params = (angle, (0.0, 0.0, 3.2))",
            "call": "resolve_acoustic_tensor(*params)",
            "gold_call": "_oracle_resolve_acoustic_tensor(*params)",
        },
        {
            "setup": "def bad_argument_sets():\n    good = (np.array([0.5, 2.0]), (6.1, 3.9, 0.85))\n    bad = [list(good) for _ in range(10)]\n    bad[0][0] = np.array([[0.5, np.nan]])\n    bad[1][0] = np.array([0.5, np.inf])\n    bad[2][1] = (6.1, 3.9)\n    bad[3][1] = (6.1, np.nan, 0.85)\n    bad[4][0] = np.array([])\n    bad[5][0] = np.array([0.5, np.nan])\n    bad[6][1] = ((6.1, 3.9, 0.85),)\n    bad[7][1] = (6.1, 3.9, np.inf)\n    bad[8][1] = 'not a tensor'\n    bad[9][0] = ['a', 'b']\n    return bad\ndef rejected_public():\n    count = 0.0\n    for args in bad_argument_sets():\n        try:\n            resolve_acoustic_tensor(*args)\n        except ValueError:\n            count += 1.0\n    return count\ndef rejected_oracle():\n    count = 0.0\n    for args in bad_argument_sets():\n        try:\n            _oracle_resolve_acoustic_tensor(*args)\n        except ValueError:\n            count += 1.0\n    return count",
            "call": "rejected_public()",
            "gold_call": "rejected_oracle()",
        },
    ]
