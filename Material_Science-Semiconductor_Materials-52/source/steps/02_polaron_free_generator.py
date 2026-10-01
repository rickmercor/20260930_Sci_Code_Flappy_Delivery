"""
Build the field-free Lindblad generator.

State order is [0,0′,X,X′], energies/hbar=[0,omega,detuning,detuning+omega]. Use separate population-transfer jump operators for each optical transition and each phonon-emission transition. The source inflow is diagonal. The pure-dephasing rate is zero. Matrix vectorization is column-major (Fortran order).

Returns
-------
return result  # complex ndarray, shape (16,16), acting on vec_F(rho); entries are in ps^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def polaron_free_generator(omega: float, detuning: float, rates: np.ndarray, gamma_ph: float) -> np.ndarray:
    """Build the field-free Lindblad generator.

    Parameters
    ----------
    omega : nonnegative float
        Phonon angular frequency in ps^-1.
    detuning : real float
        Zero-phonon optical angular-frequency detuning in ps^-1.
    rates : nonnegative real ndarray, shape (3,)
        [gamma0, gamma_prime, gamma1]: population rates of X→0, of each of X→0′ and X′→0, and of X′→0′, in ps^-1.
    gamma_ph : nonnegative float
        Phonon population-decay rate in ps^-1.

    Returns
    -------
    complex ndarray, shape (16,16), acting on vec_F(rho); entries are in ps^-1.

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

def _oracle_polaron_free_generator(omega: float, detuning: float, rates: np.ndarray, gamma_ph: float) -> np.ndarray:
    if np.shape(rates)!=(3,) or not np.all(np.isfinite(rates)) or np.any(rates<0) or not np.all(np.isfinite([omega,detuning,gamma_ph])) or min(omega,gamma_ph)<0:raise ValueError("invalid generator data")
    e=np.array([0.,omega,detuning,detuning+omega])
    h=np.diag(e);eye=np.eye(4)
    generator=-1j*(np.kron(eye,h)-np.kron(h.T,eye))
    # States [0, 0', X, X']; separate decay channels as in the source inflow.
    for dest,origin,rate in [(0,2,rates[0]),(1,2,rates[1]),(0,3,rates[1]),
                             (1,3,rates[2]),(0,1,gamma_ph),(2,3,gamma_ph)]:
        jump=np.zeros((4,4));jump[dest,origin]=np.sqrt(rate)
        loss=jump.T@jump
        generator+=np.kron(jump,jump)-.5*(np.kron(eye,loss)+np.kron(loss.T,eye))
    return generator

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nomega=4.8\ndetuning=0.3\nrates=np.array([0.0, 0.0, 0.0])\ngamma_ph=0.0\n', 'call': 'polaron_free_generator(omega,detuning,rates,gamma_ph)', 'gold_call': '_oracle_polaron_free_generator(omega,detuning,rates,gamma_ph)'}, {'setup': 'import numpy as np\nomega=4.8\ndetuning=0.3\nrates=np.array([0.1, 0.0, 0.0])\ngamma_ph=0.0\n', 'call': 'polaron_free_generator(omega,detuning,rates,gamma_ph)', 'gold_call': '_oracle_polaron_free_generator(omega,detuning,rates,gamma_ph)'}, {'setup': 'import numpy as np\nomega=4.8\ndetuning=0.3\nrates=np.array([0.0, 0.1, 0.0])\ngamma_ph=0.0\n', 'call': 'polaron_free_generator(omega,detuning,rates,gamma_ph)', 'gold_call': '_oracle_polaron_free_generator(omega,detuning,rates,gamma_ph)'}, {'setup': 'import numpy as np\nomega=4.8\ndetuning=0.3\nrates=np.array([0.0, 0.0, 0.1])\ngamma_ph=0.0\n', 'call': 'polaron_free_generator(omega,detuning,rates,gamma_ph)', 'gold_call': '_oracle_polaron_free_generator(omega,detuning,rates,gamma_ph)'}, {'setup': 'import numpy as np\nomega=4.8\ndetuning=0.3\nrates=np.array([0.0, 0.0, 0.0])\ngamma_ph=0.13\n', 'call': 'polaron_free_generator(omega,detuning,rates,gamma_ph)', 'gold_call': '_oracle_polaron_free_generator(omega,detuning,rates,gamma_ph)'}, {'setup': 'import numpy as np\nomega=4.8\ndetuning=0.3\nrates=np.array([0.003, 0.00024, 0.0025392])\ngamma_ph=0.13\n', 'call': 'polaron_free_generator(omega,detuning,rates,gamma_ph)', 'gold_call': '_oracle_polaron_free_generator(omega,detuning,rates,gamma_ph)'}, {'setup': 'import numpy as np\nomega=0.0\ndetuning=0.3\nrates=np.array([0.003, 0.00024, 0.0025392])\ngamma_ph=0.13\n', 'call': 'polaron_free_generator(omega,detuning,rates,gamma_ph)', 'gold_call': '_oracle_polaron_free_generator(omega,detuning,rates,gamma_ph)'}, {'setup': 'import numpy as np\nomega=4.8\ndetuning=0.0\nrates=np.array([0.003, 0.00024, 0.0025392])\ngamma_ph=0.13\n', 'call': 'polaron_free_generator(omega,detuning,rates,gamma_ph)', 'gold_call': '_oracle_polaron_free_generator(omega,detuning,rates,gamma_ph)'}, {'setup': 'import numpy as np\nomega=4.8\ndetuning=-1.7\nrates=np.array([0.003, 0.00024, 0.0025392])\ngamma_ph=0.13\n', 'call': 'polaron_free_generator(omega,detuning,rates,gamma_ph)', 'gold_call': '_oracle_polaron_free_generator(omega,detuning,rates,gamma_ph)'}, {'setup': 'import numpy as np\nomega=4.8\ndetuning=0.3\nrates=np.array([0.003, 0.00024, 0.0025392])\ngamma_ph=7.0\n', 'call': 'polaron_free_generator(omega,detuning,rates,gamma_ph)', 'gold_call': '_oracle_polaron_free_generator(omega,detuning,rates,gamma_ph)'}]
