"""
Compute the complex rephasing pathway coefficient for one phonon mode.

Initial rho=|0><0|. Pulse 1 at time zero contributes order one; pulse 2 at time tau contributes order two. u is time after pulse 2. Select the Fourier phase harmonic exp(i*(2*phase2-phase1)) and retain the full four-level S dependence. The measured polarization is rho[X,0]+rho[X′,0′]+sqrt(S)*(rho[X′,0]+rho[X,0′]). A five-point phase grid per pulse exactly resolves this harmonic of the stated Taylor coefficients. The numerical coefficient uses unit optical dipole and the pulse convention of the preceding step.

Returns
-------
return result  # complex scalar, dimensionless coefficient of theta1*theta2^2*exp(i*(2*phase2-phase1)).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def single_mode_rephasing_echo(huang_rhys: float, omega: float, gamma0: float, gamma_ph: float, tau: float, u: float, detuning: float) -> complex:
    """Compute the complex rephasing pathway coefficient for one phonon mode.

    Parameters
    ----------
    huang_rhys : float in [0,1]
        Dimensionless S.
    omega : nonnegative float
        Phonon angular frequency in ps^-1.
    gamma0, gamma_ph : nonnegative floats
        Optical zero-phonon and phonon population-decay rates in ps^-1.
    tau, u : nonnegative floats
        Pulse separation and time after pulse 2, in ps.
    detuning : real float
        Zero-phonon optical angular-frequency detuning in ps^-1.

    Returns
    -------
    complex scalar, dimensionless coefficient of theta1*theta2^2*exp(i*(2*phase2-phase1)).

    Raises
    ------
    ValueError
        If inputs fall outside the documented numerical domain.
    """
    return 0j

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def _oracle_single_mode_rephasing_echo(huang_rhys: float, omega: float, gamma0: float, gamma_ph: float, tau: float, u: float, detuning: float) -> complex:
    if not np.all(np.isfinite([tau,u])) or min(tau,u)<0:raise ValueError("delays must be finite and nonnegative")
    rates=_oracle_polaron_optical_rates(huang_rhys,gamma0)
    generator=_oracle_polaron_free_generator(omega,detuning,rates,gamma_ph)
    first=np.zeros((16,16),dtype=complex);second=first.copy()
    for j in range(5):
        phase=2*np.pi*j/5
        first+=np.exp(1j*phase)*_oracle_polaron_pulse_coefficient(huang_rhys,phase,1)/5
        second+=np.exp(-2j*phase)*_oracle_polaron_pulse_coefficient(huang_rhys,phase,2)/5
    rho0=np.zeros(16,dtype=complex);rho0[0]=1
    rho=(expm(generator*u)@second@expm(generator*tau)@first@rho0).reshape((4,4),order='F')
    s=np.sqrt(huang_rhys)
    return complex(rho[2,0]+rho[3,1]+s*(rho[3,0]+rho[2,1]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nhuang_rhys=0.083\nomega=4.8\ngamma0=0.0032258064516129032\ngamma_ph=0.137\ntau=3.1\nu=3.1\ndetuning=0.7\n', 'call': 'np.array([single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)', 'gold_call': 'np.array([_oracle_single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)'}, {'setup': 'import numpy as np\nhuang_rhys=0.083\nomega=4.8\ngamma0=0.0032258064516129032\ngamma_ph=0.137\ntau=3.1\nu=3.47\ndetuning=0.7\n', 'call': 'np.array([single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)', 'gold_call': 'np.array([_oracle_single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)'}, {'setup': 'import numpy as np\nhuang_rhys=0.0\nomega=4.8\ngamma0=0.0032258064516129032\ngamma_ph=0.137\ntau=3.1\nu=2.7\ndetuning=0.7\n', 'call': 'np.array([single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)', 'gold_call': 'np.array([_oracle_single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)'}, {'setup': 'import numpy as np\nhuang_rhys=0.083\nomega=4.8\ngamma0=0.0\ngamma_ph=0.0\ntau=3.1\nu=3.1\ndetuning=0.7\n', 'call': 'np.array([single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)', 'gold_call': 'np.array([_oracle_single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)'}, {'setup': 'import numpy as np\nhuang_rhys=0.083\nomega=4.8\ngamma0=0.0032258064516129032\ngamma_ph=0.0\ntau=3.1\nu=3.6\ndetuning=0.7\n', 'call': 'np.array([single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)', 'gold_call': 'np.array([_oracle_single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)'}, {'setup': 'import numpy as np\nhuang_rhys=0.083\nomega=4.8\ngamma0=0.0032258064516129032\ngamma_ph=4.0\ntau=3.1\nu=3.1\ndetuning=0.7\n', 'call': 'np.array([single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)', 'gold_call': 'np.array([_oracle_single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)'}, {'setup': 'import numpy as np\nhuang_rhys=0.083\nomega=4.8\ngamma0=0.0032258064516129032\ngamma_ph=0.137\ntau=0.0\nu=0.8\ndetuning=0.7\n', 'call': 'np.array([single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)', 'gold_call': 'np.array([_oracle_single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)'}, {'setup': 'import numpy as np\nhuang_rhys=0.083\nomega=4.8\ngamma0=0.0032258064516129032\ngamma_ph=0.137\ntau=3.1\nu=0.0\ndetuning=0.7\n', 'call': 'np.array([single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)', 'gold_call': 'np.array([_oracle_single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)'}, {'setup': 'import numpy as np\nhuang_rhys=0.083\nomega=4.8\ngamma0=0.0032258064516129032\ngamma_ph=0.137\ntau=40.0\nu=40.0\ndetuning=0.7\n', 'call': 'np.array([single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)', 'gold_call': 'np.array([_oracle_single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)'}, {'setup': 'import numpy as np\nhuang_rhys=0.083\nomega=7.75\ngamma0=0.0032258064516129032\ngamma_ph=0.137\ntau=3.1\nu=2.4\ndetuning=0.7\n', 'call': 'np.array([single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)', 'gold_call': 'np.array([_oracle_single_mode_rephasing_echo(huang_rhys,omega,gamma0,gamma_ph,tau,u,detuning)], dtype=np.complex128)'}]
