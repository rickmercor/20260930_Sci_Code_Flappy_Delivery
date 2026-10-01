"""
Obtain one pulse-area coefficient of the impulsive density map.

Let R be the optical raising matrix whose block (rows X,X′; columns 0,0′) is [[1,sqrt(S)],[sqrt(S),1]], with all other entries zero. The dimensionless pulse interaction is exp(i*phase)R+exp(-i*phase)R†, and its unitary is exp(-i*theta*interaction). Return the coefficient of theta^order in the density-map Taylor series. This includes the Taylor factorial. The source pulse convention has positive off-diagonal overlaps and unit diagonal pulse vertices; its population rates retain their separately specified overlap factors.

Returns
-------
return result  # complex ndarray, shape (16,16), dimensionless pulse coefficient acting on vec_F(rho).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def polaron_pulse_coefficient(huang_rhys: float, phase: float, order: int) -> np.ndarray:
    """Obtain one pulse-area coefficient of the impulsive density map.

    Parameters
    ----------
    huang_rhys : float in [0,1]
        Dimensionless S.
    phase : real float
        Optical carrier phase in radians.
    order : integer in {0,1,2}
        Power of dimensionless pulse area in the Taylor series.

    Returns
    -------
    complex ndarray, shape (16,16), dimensionless pulse coefficient acting on vec_F(rho).

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

def _oracle_polaron_pulse_coefficient(huang_rhys: float, phase: float, order: int) -> np.ndarray:
    if not np.all(np.isfinite([huang_rhys,phase])) or not 0<=huang_rhys<=1 or order not in (0,1,2):raise ValueError("invalid pulse data")
    s=np.sqrt(huang_rhys)
    raising=np.zeros((4,4),dtype=complex);raising[2:,:2]=np.array([[1.,s],[s,1.]])
    interaction=np.exp(1j*phase)*raising+np.exp(-1j*phase)*raising.conj().T
    action=-1j*(np.kron(np.eye(4),interaction)-np.kron(interaction.T,np.eye(4)))
    if order==0:return np.eye(16,dtype=complex)
    if order==1:return action
    if order==2:return action@action/2
    raise ValueError('order must be 0, 1 or 2')

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nhuang_rhys=0.08\nphase=0.31\norder=0\n', 'call': 'polaron_pulse_coefficient(huang_rhys,phase,order)', 'gold_call': '_oracle_polaron_pulse_coefficient(huang_rhys,phase,order)'}, {'setup': 'import numpy as np\nhuang_rhys=0.08\nphase=0.0\norder=1\n', 'call': 'polaron_pulse_coefficient(huang_rhys,phase,order)', 'gold_call': '_oracle_polaron_pulse_coefficient(huang_rhys,phase,order)'}, {'setup': 'import numpy as np\nhuang_rhys=0.08\nphase=0.0\norder=2\n', 'call': 'polaron_pulse_coefficient(huang_rhys,phase,order)', 'gold_call': '_oracle_polaron_pulse_coefficient(huang_rhys,phase,order)'}, {'setup': 'import numpy as np\nhuang_rhys=0.08\nphase=1.5707963267948966\norder=1\n', 'call': 'polaron_pulse_coefficient(huang_rhys,phase,order)', 'gold_call': '_oracle_polaron_pulse_coefficient(huang_rhys,phase,order)'}, {'setup': 'import numpy as np\nhuang_rhys=0.08\nphase=1.5707963267948966\norder=2\n', 'call': 'polaron_pulse_coefficient(huang_rhys,phase,order)', 'gold_call': '_oracle_polaron_pulse_coefficient(huang_rhys,phase,order)'}, {'setup': 'import numpy as np\nhuang_rhys=0.0\nphase=0.23\norder=1\n', 'call': 'polaron_pulse_coefficient(huang_rhys,phase,order)', 'gold_call': '_oracle_polaron_pulse_coefficient(huang_rhys,phase,order)'}, {'setup': 'import numpy as np\nhuang_rhys=0.0\nphase=-0.42\norder=2\n', 'call': 'polaron_pulse_coefficient(huang_rhys,phase,order)', 'gold_call': '_oracle_polaron_pulse_coefficient(huang_rhys,phase,order)'}, {'setup': 'import numpy as np\nhuang_rhys=0.12\nphase=0.73\norder=1\n', 'call': 'polaron_pulse_coefficient(huang_rhys,phase,order)', 'gold_call': '_oracle_polaron_pulse_coefficient(huang_rhys,phase,order)'}, {'setup': 'import numpy as np\nhuang_rhys=0.08\nphase=3.141592653589793\norder=1\n', 'call': 'polaron_pulse_coefficient(huang_rhys,phase,order)', 'gold_call': '_oracle_polaron_pulse_coefficient(huang_rhys,phase,order)'}, {'setup': 'import numpy as np\nhuang_rhys=0.64\nphase=0.27\norder=2\n', 'call': 'polaron_pulse_coefficient(huang_rhys,phase,order)', 'gold_call': '_oracle_polaron_pulse_coefficient(huang_rhys,phase,order)'}]
