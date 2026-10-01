"""
Recover the polarization-invariant amplitude and fine-structure contrast.

Columns are delay samples. The measured two-channel field amplitudes satisfy channels=mixing@[A_parallel,A_cross]. Apply the source polarization sum and contrast to the recovered amplitudes. The polarization sum is Sigma=Psi0. All denominators A_parallel+2*A_cross are nonzero.

Returns
-------
return result  # real ndarray, shape (2,T), rows [Sigma, polarization contrast rho], both dimensionless.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def recover_intrinsic_echo(channels: np.ndarray, mixing: np.ndarray) -> np.ndarray:
    """Recover the polarization-invariant amplitude and fine-structure contrast.

    Parameters
    ----------
    channels : real ndarray, shape (2,T)
        Measured dimensionless heterodyne field amplitudes; row order parallel, cross.
    mixing : real ndarray, shape (2,2)
        Known nonsingular dimensionless detector response, measured=mixing@physical.

    Returns
    -------
    real ndarray, shape (2,T), rows [Sigma, polarization contrast rho], both dimensionless.

    Raises
    ------
    ValueError
        If inputs fall outside the documented numerical domain.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def _oracle_recover_intrinsic_echo(channels: np.ndarray, mixing: np.ndarray) -> np.ndarray:
    if np.ndim(channels)!=2 or channels.shape[0]!=2 or np.shape(mixing)!=(2,2) or not np.all(np.isfinite(channels)) or not np.all(np.isfinite(mixing)):raise ValueError("invalid detector data")
    try:intrinsic=np.linalg.solve(mixing,channels)
    except np.linalg.LinAlgError as exc:raise ValueError("singular detector response") from exc
    denominator=intrinsic[0]+2*intrinsic[1]
    if np.any(denominator==0):raise ValueError("undefined polarization contrast")
    return np.stack([denominator/2,(intrinsic[0]-3*intrinsic[1])/denominator])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nchannels=np.array([[2.4, 1.68], [0.16000000000000003, 0.24]])\nmixing=np.array([[1.2, 0.0], [0.0, 0.8]])\n', 'call': 'recover_intrinsic_echo(channels,mixing)', 'gold_call': '_oracle_recover_intrinsic_echo(channels,mixing)'}, {'setup': 'import numpy as np\nchannels=np.array([[1.1260000000000001, 1.46], [0.028999999999999998, 0.158]])\nmixing=np.array([[1.0, 0.2], [-0.08, 0.9]])\n', 'call': 'recover_intrinsic_echo(channels,mixing)', 'gold_call': '_oracle_recover_intrinsic_echo(channels,mixing)'}, {'setup': 'import numpy as np\nchannels=np.array([[0.95, -0.2700000000000001], [-0.07400000000000002, 0.538]])\nmixing=np.array([[0.8, 0.1], [0.03, 1.1]])\n', 'call': 'recover_intrinsic_echo(channels,mixing)', 'gold_call': '_oracle_recover_intrinsic_echo(channels,mixing)'}, {'setup': 'import numpy as np\nchannels=np.array([[0.9, 0.6], [0.3, 0.2]])\nmixing=np.array([[1.0, 0.0], [0.0, 1.0]])\n', 'call': 'recover_intrinsic_echo(channels,mixing)', 'gold_call': '_oracle_recover_intrinsic_echo(channels,mixing)'}, {'setup': 'import numpy as np\nchannels=np.array([[1.2, 0.7], [0.0, 0.0]])\nmixing=np.array([[1.0, 0.0], [0.0, 1.0]])\n', 'call': 'recover_intrinsic_echo(channels,mixing)', 'gold_call': '_oracle_recover_intrinsic_echo(channels,mixing)'}, {'setup': 'import numpy as np\nchannels=np.array([[0.008, 0.016], [0.2, 0.4]])\nmixing=np.array([[1.0, 0.04], [0.02, 1.0]])\n', 'call': 'recover_intrinsic_echo(channels,mixing)', 'gold_call': '_oracle_recover_intrinsic_echo(channels,mixing)'}]
