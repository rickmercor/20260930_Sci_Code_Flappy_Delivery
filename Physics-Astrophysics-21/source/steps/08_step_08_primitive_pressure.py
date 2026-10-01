"""
Recover ideal-gas pressure from an admissible relativistic local state.

The orthonormal conservative components are (D,m_r,m_z,E), the speed of light
is one, and the ideal-gas adiabatic index is supplied. Pressure is the positive
primitive-state pressure compatible with these conserved variables.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def primitive_pressure(local_state, gamma=5/3):
    """Return the physical pressure of one flat-frame conservative state.

    Parameters
    ----------
    local_state : array_like, shape (4,)
        Finite (D,m_r,m_z,E) with D>0 and E>sqrt(D^2+m_r^2+m_z^2).
    gamma : float
        Finite ideal-gas adiabatic index in (1,2].

    Returns
    -------
    float
        Positive pressure in the same units as E. The input is not modified.
        Invalid data raise ValueError.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

from scipy.optimize import brentq

def _oracle_primitive_pressure(local_state, gamma=5/3):
    u = np.asarray(local_state,dtype=float)
    if (u.shape != (4,) or not np.isfinite(u).all() or not np.isfinite(gamma)
            or not 1 < gamma <= 2):
        raise ValueError('Invalid local state or adiabatic index')
    scale = u[3]
    if u[0] <= 0 or scale <= 0:
        raise ValueError('State is not strictly admissible')
    d,mr,mz,e = u/scale
    if e <= np.linalg.norm(np.array([d,mr,mz])):
        raise ValueError('State is not strictly admissible')
    momentum_squared = mr*mr+mz*mz
    def residual(p):
        ep = e+p
        return p/(gamma-1)-e+momentum_squared/ep+d*np.sqrt(1-momentum_squared/ep**2)
    return float(scale*brentq(residual,0.,(gamma-1)*e,
                             xtol=np.nextafter(0.,1.),rtol=1e-14))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'w = _oracle_relative_physical_scaling(_s7_absolute(),0); u = _oracle_regular_local_state(w,0)[2,1]',
         'call':'primitive_pressure(u)', 'gold_call':'1.2212246706514617136238315514'},
        {'setup':'u = np.array([1.,0.,0.,3.])', 'call':'primitive_pressure(u)',
         'gold_call':'4/3'},
        {'setup':'u = np.array([1.,0.,0.,3.])*1e-20', 'call':'primitive_pressure(u)*1e20',
         'gold_call':'4/3'},
        {'setup':'rho=0.7; p=0.04; gamma=1.4; v=np.array([0.8,0.3]); lorentz=1/np.sqrt(1-np.dot(v,v)); h=1+gamma/(gamma-1)*p/rho; u=np.r_[rho*lorentz,rho*h*lorentz**2*v,rho*h*lorentz**2-p]',
         'call':'primitive_pressure(u,gamma=gamma)', 'gold_call':'p'},
        {'setup':'u = np.array([1.,2.,0.,1.])\ndef _s8_invalid():\n    try:\n        primitive_pressure(u)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0',
         'call':'_s8_invalid()', 'gold_call':'1'},
    ]
