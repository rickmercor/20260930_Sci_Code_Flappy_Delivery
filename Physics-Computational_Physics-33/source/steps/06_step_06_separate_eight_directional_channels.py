"""
Separate the four eight-directional Kerr channels.

Apply the four linear combinations in the source paper to a scan whose direction axis is mu=[0,45,...,315] degrees. Return channels in order: (M_L+M_L^3), M_L*M_T, (M_T^2-M_L^2), and M_T^3. The combinations are respectively (Phi_90-Phi_270)/2; (Phi_45+Phi_225-Phi_135-Phi_315)/2; (Phi_0+Phi_180-Phi_90-Phi_270)/2; and (Phi_0-Phi_180)/2. Apply them independently to both polarizations and both real/imag components. Require a nonempty finite scan of shape (A,8,2,2); raise ValueError otherwise.

Returns
-------
channels : np.ndarray, shape (A,4,2,2), float Channel, polarization s/p, and real/imag axes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def separate_eight_directional_channels(kerr_scan: np.ndarray) -> np.ndarray:
    '''Return the four separated channel packets.

    Parameters
    ----------
    kerr_scan : np.ndarray, shape (A,8,2,2)
        Nonempty finite scan in the prescribed direction order, with s/p
        polarization and final real/imag axes.

    Returns
    -------
    channels : np.ndarray, shape (A,4,2,2), float
        Channel, polarization s/p, and real/imag axes.

    Raises
    ------
    ValueError
        The scan is empty, nonfinite, or has an invalid shape.'''
    return np.empty((0, 4, 2, 2), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _c_separate_eight_directional_channels__packet(values):
    values = np.asarray(values, dtype=complex)
    return np.stack((values.real, values.imag), axis=-1).astype(float)

def _c_separate_eight_directional_channels__unpacket(values, shape):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError('malformed real/imag packet')
    return array[..., 0] + 1j * array[..., 1]

def _c_separate_eight_directional_channels(kerr_scan: np.ndarray) -> np.ndarray:
    scan = np.asarray(kerr_scan, dtype=float)
    if scan.ndim != 4 or scan.shape[1:] != (8, 2, 2) or scan.shape[0] == 0:
        raise ValueError('kerr_scan must have nonempty shape (A,8,2,2)')
    values = _c_separate_eight_directional_channels__unpacket(scan, scan.shape)
    channels = np.stack(((values[:, 2] - values[:, 6]) / 2.0, (values[:, 1] + values[:, 5] - values[:, 3] - values[:, 7]) / 2.0, (values[:, 0] + values[:, 4] - values[:, 2] - values[:, 6]) / 2.0, (values[:, 0] - values[:, 4]) / 2.0), axis=1)
    return _c_separate_eight_directional_channels__packet(channels)

def _oracle_separate_eight_directional_channels(kerr_scan):
    return _c_separate_eight_directional_channels(kerr_scan)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    exception_setup = 'def _value_error_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n'
    return [
        {
            "setup": '# Case: normal\nx=np.sin(np.arange(5*8*2*2,dtype=float).reshape(5,8,2,2)/9)',
            "call": 'separate_eight_directional_channels(x)',
            "gold_call": '_oracle_separate_eight_directional_channels(x)',
        },
        {
            "setup": '# Case: boundary\n# Coverage: boundary (one all-zero scan row)\nx=np.zeros((1,8,2,2),dtype=float)',
            "call": 'separate_eight_directional_channels(x)',
            "gold_call": '_oracle_separate_eight_directional_channels(x)',
        },
        {
            "setup": '# Case: edge\n# Coverage: edge (sparse signed scan)\nx=np.zeros((2,8,2,2),dtype=float)\nx[0,7,1,1]=-3.25\nx[1,0,0,0]=4.5',
            "call": 'separate_eight_directional_channels(x)',
            "gold_call": '_oracle_separate_eight_directional_channels(x)',
        },
        {
            "setup": '# Case: edge\n\n' + exception_setup,
            "call": '_value_error_code(separate_eight_directional_channels, np.zeros((2, 7, 2, 2)))',
            "gold_call": '_value_error_code(_oracle_separate_eight_directional_channels, np.zeros((2, 7, 2, 2)))',
        },
        {
            "setup": '# Case: edge\n\n' + exception_setup,
            "call": '_value_error_code(separate_eight_directional_channels, np.zeros((0, 8, 2, 2)))',
            "gold_call": '_value_error_code(_oracle_separate_eight_directional_channels, np.zeros((0, 8, 2, 2)))',
        },
        {
            "setup": '# Case: edge\nx=np.zeros((1,8,2,2)); x[0,0,0,0]=np.nan\n' + exception_setup,
            "call": '_value_error_code(separate_eight_directional_channels, x)',
            "gold_call": '_value_error_code(_oracle_separate_eight_directional_channels, x)',
        },
    ]
