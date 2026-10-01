"""
Affine uncertainty in a conditioned merger rate history

To test how formation history positivity conditions uncertain merger histories, use three shared dimensionless coefficients $$\boldsymbol u\in[-1,1]^3$$ and an affine family $$R(z,\boldsymbol u)=r_0(z)+\sum_{j=1}^3u_jr_j(z)$$. Define $$w(x)=[1-\cos(\pi\,\mathrm{clip}(x,0,1))]/2$$, $$\ell=w(z/0.1)$$, $$b=w[(z-1.5)/0.05]$$ and $$E=[(1+z)/1.2]^\kappa$$. The central history is $$r_0=\ell[(1-b)R_pE+bR_{\rm asym}]$$. With supplied amplitudes $$a_0,a_1,a_2$$, set $$r_1=\ell(1-b)R_pE a_0$$, $$r_2=\ell(1-b)R_pE a_1\exp[-(z-0.35)^2/(2\times0.12^2)]$$ and $$r_3=\ell(1-b)R_pE a_2\exp[-(z-0.95)^2/(2\times0.17^2)]$$. The modes represent uncertain normalization and localized redshift structure. This is an authored finite-dimensional approximation to a rate posterior; its coefficient distribution is supplied in the later conditioning steps, and these modes are not already transformed by a Gaussian covariance.

Returns
-------
np.ndarray of shape (4,N), central merger rate followed by its three affine modes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def affine_merger_family(z_grid: np.ndarray, pivot_rate: float, r_asym: float,
                         amplitudes: np.ndarray, kappa: float = 3.2) -> np.ndarray:
    r'''Construct the central history and three affine uncertainty modes.

    Parameters
    ----------
    z_grid : np.ndarray
        Nonempty finite real one-dimensional redshift grid, all entries >= 0.
    pivot_rate : float
        Finite positive central rate at z=0.2, in Gpc^-3 yr^-1.
    r_asym : float
        Finite nonnegative common high-redshift boundary, in Gpc^-3 yr^-1.
    amplitudes : np.ndarray
        Finite nonnegative real shape (3,), with sum strictly below one. The three dimensionless amplitudes scale the normalization, lower-redshift feature and higher-redshift feature, in that order. These bounds ensure a nonnegative merger history for every coefficient in [-1,1]^3.
    kappa : float
        Finite redshift exponent, default 3.2.

    Returns
    -------
    family : np.ndarray
        Finite shape (4,N), in Gpc^-3 yr^-1. Row 0 is the central history; rows 1–3 multiply the three latent coefficients. These are affine coefficients, not derivative or Taylor coefficients.

    Raises
    ------
    ValueError
        If the finite real grid, scalar, shape or amplitude-domain contracts fail, or if the evaluated family is nonfinite.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_affine_merger_family(z_grid: np.ndarray, pivot_rate: float, r_asym: float,
                                 amplitudes: np.ndarray, kappa: float = 3.2) -> np.ndarray:
    if not np.isrealobj(z_grid) or not np.isrealobj(amplitudes):
        raise ValueError("grid and amplitudes must be real")
    try:
        z = np.asarray(z_grid, dtype=float)
        amp = np.asarray(amplitudes, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("grid and amplitudes must be finite real arrays") from exc
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("redshifts must be nonempty, finite and nonnegative")
    if amp.shape != (3,) or not np.all(np.isfinite(amp)) or np.any(amp < 0) or amp.sum() >= 1:
        raise ValueError("three nonnegative amplitudes with sum below one required")
    vals = (pivot_rate, r_asym, kappa)
    if any(not np.isscalar(v) or not np.isrealobj(v) for v in vals):
        raise ValueError("rate and exponent parameters must be real scalars")
    try:
        rp, ra, exponent = map(float, vals)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("rate and exponent parameters must be finite real scalars") from exc
    if not np.all(np.isfinite([rp,ra,exponent])) or rp <= 0 or ra < 0:
        raise ValueError("positive pivot and nonnegative boundary required")
    result = np.zeros((4,z.size))
    result[0] = ra
    active = z < 1.55
    za = z[active]
    low = .5*(1-np.cos(np.pi*np.clip(za/.1,0,1)))
    blend = .5*(1-np.cos(np.pi*np.clip((za-1.5)/.05,0,1)))
    with np.errstate(over='ignore',invalid='ignore'):
        core = low*(1-blend)*rp*np.exp(exponent*(np.log1p(za)-np.log(1.2)))
        result[0,active] = core + low*blend*ra
        result[1,active] = core*amp[0]
        result[2,active] = core*amp[1]*np.exp(-.5*((za-.35)/.12)**2)
        result[3,active] = core*amp[2]*np.exp(-.5*((za-.95)/.17)**2)
    if not np.all(np.isfinite(result)):
        raise ValueError("evaluated history family must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent input copies in explicit scientific case specifications."""
    return [{'setup': 'import numpy as np\n'
               'z=np.array([0, 0.05, 0.1, 0.2, 0.35, 0.95, 1.5, 1.525, 1.55, 3],dtype=float)\n'
               'a=np.array([0.24, 0.2, 0.2],dtype=float)\n'
               'rp,ra,k=29,300,3.2\n',
      'call': 'affine_merger_family(z.copy(),rp,ra,a.copy(),k)',
      'gold_call': '_oracle_affine_merger_family(z.copy(),rp,ra,a.copy(),k)',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'z=np.array([0.1, 0.35, 0.95, 1.53],dtype=float)\n'
               'a=np.array([0.1, 0.4, 0.05],dtype=float)\n'
               'rp,ra,k=22.5,100,2.5\n',
      'call': 'affine_merger_family(z.copy(),rp,ra,a.copy(),k)',
      'gold_call': '_oracle_affine_merger_family(z.copy(),rp,ra,a.copy(),k)',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'z=np.array([0.95, 0.35, 0.2, 0, 1.6],dtype=float)\n'
               'a=np.array([0, 0, 0],dtype=float)\n'
               'rp,ra,k=37.5,0,3.2\n',
      'call': 'affine_merger_family(z.copy(),rp,ra,a.copy(),k)',
      'gold_call': '_oracle_affine_merger_family(z.copy(),rp,ra,a.copy(),k)',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'z=np.array([0.05, 0.15, 0.4, 0.7, 1, 1.54],dtype=float)\n'
               'a=np.array([0, 0.2, 0.3],dtype=float)\n'
               'rp,ra,k=18,200,4\n',
      'call': 'affine_merger_family(z.copy(),rp,ra,a.copy(),k)',
      'gold_call': '_oracle_affine_merger_family(z.copy(),rp,ra,a.copy(),k)',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'z=np.array([0.2],dtype=float)\n'
               'a=np.array([0.3, 0, 0],dtype=float)\n'
               'rp,ra,k=29,300,3.2\n',
      'call': 'affine_merger_family(z.copy(),rp,ra,a.copy(),k)',
      'gold_call': '_oracle_affine_merger_family(z.copy(),rp,ra,a.copy(),k)',
      'tol': 1e-06}]
