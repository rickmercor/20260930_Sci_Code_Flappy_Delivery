"""
Evaluate the source amplitude kernel used by calibration.

Use the main-paper small-S echo amplitude Psi0(t;Omega), retaining its written exponential factors and its terms at Omega and 2*Omega. t is the interval between the two excitation pulses. This step is the reduced calibration law; prediction uses the full complex four-level coefficient.

Returns
-------
return result  # real ndarray, shape (T,), dimensionless Psi0 amplitude in input time order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def weak_polaron_echo(times: np.ndarray, omega: float, huang_rhys: float, gamma0: float, gamma_ph: float) -> np.ndarray:
    """Evaluate the source amplitude kernel used by calibration.

    Parameters
    ----------
    times : nonnegative real ndarray, shape (T,)
        Pulse separations in ps.
    omega : nonnegative float
        Phonon angular frequency in ps^-1.
    huang_rhys : float in [0,0.2]
        Dimensionless S in the small-coupling calibration model.
    gamma0, gamma_ph : nonnegative floats
        Optical zero-phonon and phonon population-decay rates in ps^-1.

    Returns
    -------
    real ndarray, shape (T,), dimensionless Psi0 amplitude in input time order.

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

def _oracle_weak_polaron_echo(times: np.ndarray, omega: float, huang_rhys: float, gamma0: float, gamma_ph: float) -> np.ndarray:
    if np.ndim(times)!=1 or not np.all(np.isfinite(times)) or np.any(times<0) or not np.all(np.isfinite([omega,huang_rhys,gamma0,gamma_ph])) or min(omega,gamma0,gamma_ph)<0 or not 0<=huang_rhys<=.2:raise ValueError("invalid calibration-kernel data")
    t=np.asarray(times);s=huang_rhys
    return np.exp(-(1+s)*gamma0*t)*(1+s*np.exp(-(gamma_ph-s*gamma0)*t)
        +s*np.cos(omega*t)*np.exp(-.5*(gamma_ph-2*s*gamma0)*t)
          *(2+np.exp(-s*gamma0*t)+np.exp(-gamma_ph*t))
        +s*np.cos(2*omega*t)*np.exp(-(gamma_ph-s*gamma0)*t))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ntimes=np.array([0.2, 1.3, 3.7, 8.1])\nomega=4.8\nhuang_rhys=0.0\ngamma0=0.003\ngamma_ph=0.14\n', 'call': 'weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)', 'gold_call': '_oracle_weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)'}, {'setup': 'import numpy as np\ntimes=np.array([0.0])\nomega=4.8\nhuang_rhys=0.08\ngamma0=0.003\ngamma_ph=0.14\n', 'call': 'weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)', 'gold_call': '_oracle_weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)'}, {'setup': 'import numpy as np\ntimes=np.array([0.2, 1.3, 3.7, 8.1])\nomega=4.8\nhuang_rhys=0.08\ngamma0=0.003\ngamma_ph=0.0\n', 'call': 'weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)', 'gold_call': '_oracle_weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)'}, {'setup': 'import numpy as np\ntimes=np.array([0.2, 1.3, 3.7, 8.1])\nomega=4.8\nhuang_rhys=0.08\ngamma0=0.0\ngamma_ph=0.14\n', 'call': 'weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)', 'gold_call': '_oracle_weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)'}, {'setup': 'import numpy as np\ntimes=np.array([0.3272492347489368, 0.9817477042468103])\nomega=4.8\nhuang_rhys=0.08\ngamma0=0.003\ngamma_ph=0.14\n', 'call': 'weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)', 'gold_call': '_oracle_weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)'}, {'setup': 'import numpy as np\ntimes=np.array([0.1636246173744684, 0.4908738521234052])\nomega=4.8\nhuang_rhys=0.08\ngamma0=0.003\ngamma_ph=0.14\n', 'call': 'weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)', 'gold_call': '_oracle_weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)'}, {'setup': 'import numpy as np\ntimes=np.array([70.0, 130.0, 210.0])\nomega=4.8\nhuang_rhys=0.08\ngamma0=0.003\ngamma_ph=0.14\n', 'call': 'weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)', 'gold_call': '_oracle_weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)'}, {'setup': 'import numpy as np\ntimes=np.array([0.2, 1.3, 3.7, 8.1])\nomega=4.8\nhuang_rhys=0.08\ngamma0=0.003\ngamma_ph=0.14\n', 'call': 'weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)', 'gold_call': '_oracle_weak_polaron_echo(times,omega,huang_rhys,gamma0,gamma_ph)'}]
