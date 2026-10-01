"""
Build the paper's eight-directional magnetization design.

For every supplied sample angle alpha, construct the eight in-plane magnetization directions mu=[0,45,90,135,180,225,270,315] degrees in that order. The sample angles are already in radians and must be copied without wrapping or sorting. With the paper's axes, M_T=cos(mu), M_L=sin(mu), and M_P=0. Return alpha with each direction so downstream tensor rotations and signal rows cannot lose alignment. Preserve alpha-major then direction-major order: all eight rows for sample_angles[0] precede all eight for sample_angles[1]. Accept a nonempty finite one-dimensional array only, raising ValueError otherwise, and return float64 values without rounding cardinal-direction trigonometric results.

Returns
-------
design : np.ndarray, shape (A,8,4), float Columns alpha,M_T,M_L,M_P in fixed direction order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_eight_directional_design(sample_angles: np.ndarray) -> np.ndarray:
    '''Return sample angles and eight in-plane directions.

    Parameters
    ----------
    sample_angles : np.ndarray, shape (A,)
        Nonempty finite sample angles in radians; preserve their order.

    Returns
    -------
    design : np.ndarray, shape (A,8,4), float
        Columns alpha,M_T,M_L,M_P in fixed direction order.

    Raises
    ------
    ValueError
        Sample angles are empty, nonfinite, or not one-dimensional.'''
    return np.empty((0, 8, 4), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _c_build_eight_directional_design(sample_angles: np.ndarray) -> np.ndarray:
    alpha = np.asarray(sample_angles, dtype=float)
    if alpha.ndim != 1 or alpha.size == 0 or (not np.all(np.isfinite(alpha))):
        raise ValueError('sample_angles must be a nonempty finite 1-D array')
    mu = np.arange(8, dtype=float) * (np.pi / 4.0)
    magnetization = np.stack((np.cos(mu), np.sin(mu), np.zeros(8)), axis=1)
    out = np.empty((alpha.size, 8, 4), dtype=float)
    out[:, :, 0] = alpha[:, None]
    out[:, :, 1:] = magnetization[None, :, :]
    return out

def _oracle_build_eight_directional_design(sample_angles):
    return _c_build_eight_directional_design(sample_angles)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    exception_setup = 'def _value_error_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n'
    return [
        {
            "setup": '# Case: normal\n',
            "call": 'build_eight_directional_design(np.array([-0.4,0.2,1.1]))',
            "gold_call": '_oracle_build_eight_directional_design(np.array([-0.4,0.2,1.1]))',
        },
        {
            "setup": '# Case: boundary\n# Coverage: boundary (single zero sample angle)\n',
            "call": 'build_eight_directional_design(np.array([0.0]))',
            "gold_call": '_oracle_build_eight_directional_design(np.array([0.0]))',
        },
        {
            "setup": '# Case: edge\n# Coverage: edge (duplicate unwrapped sample angles)\n',
            "call": 'build_eight_directional_design(np.array([2*np.pi,-2*np.pi,2*np.pi]))',
            "gold_call": '_oracle_build_eight_directional_design(np.array([2*np.pi,-2*np.pi,2*np.pi]))',
        },
        {
            "setup": '# Case: edge\n\n' + exception_setup,
            "call": '_value_error_code(build_eight_directional_design, np.array([], dtype=float))',
            "gold_call": '_value_error_code(_oracle_build_eight_directional_design, np.array([], dtype=float))',
        },
        {
            "setup": '# Case: edge\n\n' + exception_setup,
            "call": '_value_error_code(build_eight_directional_design, np.zeros((2, 2), dtype=float))',
            "gold_call": '_value_error_code(_oracle_build_eight_directional_design, np.zeros((2, 2), dtype=float))',
        },
        {
            "setup": '# Case: edge\n\n' + exception_setup,
            "call": '_value_error_code(build_eight_directional_design, np.array([0.0, np.nan]))',
            "gold_call": '_value_error_code(_oracle_build_eight_directional_design, np.array([0.0, np.nan]))',
        },
    ]
