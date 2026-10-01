"""
Given a spherical spin-unpolarized density, its reduced gradient and iso-orbital indicator on a radial grid, evaluate W_inf with the full primary-paper ePC meta-GGA interpolation between the two branches from the previous step.

Along the density-fixed adiabatic connection the

integrand W_alpha[n] behaves at large coupling strength alpha as

W_alpha -> W_inf[n] + W'_inf[n] alpha^{-1/2} + ... . In the

strictly-correlated-electron (SCE) picture W_inf is the SCE electron-electron

interaction energy minus the Hartree energy. It is negative, and it is the

leading term of the interaction energy when the coupling strength is very

large. Exact SCE values are costly and exist only for a few systems, so

interaction-strength interpolation functionals use semilocal models of W_inf

built on the point-charge-plus-continuum (PC) picture. Gradient-corrected

(GGA) PC models depend only on the reduced gradient s. A meta-GGA also uses

the iso-orbital indicator z, so it can tell one-orbital regions from slowly

varying regions.

Returns
-------
a float, W_inf in hartree. Raise ``ValueError`` if array lengths differ, r is not strictly increasing, density is negative, or z lies outside [0,1].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def epc_w_inf(r: "np.ndarray", n: "np.ndarray", s: "np.ndarray", z: "np.ndarray") -> float:
    """Return W_inf[n] in hartree from the published ePC meta-GGA.

    Inputs share a strictly increasing radial grid. ``n`` is nonnegative,
    ``s`` is the reduced gradient, and ``z`` is the iso-orbital indicator in
    [0,1]. Use ``epc_w_inf_branches`` for the two paper-defined branches and
    recover from the primary paper the z-interpolation that forms the full
    enhancement factor. Use A=-1.451 and integrate
    4*pi*r^2*A*n^(4/3)*F by the trapezoidal rule; n<1e-30 contributes zero.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _radial_trapezoid(r, f):
    return float(np.sum(0.5*(f[1:]+f[:-1])*np.diff(r)))


def _oracle_epc_w_inf(r: np.ndarray, n: np.ndarray, s: np.ndarray, z: np.ndarray) -> float:
    r=np.asarray(r,dtype=float); n=np.asarray(n,dtype=float); s=np.asarray(s,dtype=float); z=np.asarray(z,dtype=float)
    if not (len(r)==len(n)==len(s)==len(z)):
        raise ValueError("Arrays must have the same length.")
    if np.any(np.diff(r)<=0.0) or np.any(n<0.0) or np.any((z<0.0)|(z>1.0)):
        raise ValueError("Invalid radial grid, density or z.")
    f0,f1=_oracle_epc_w_inf_branches(s)
    f=f0+(z*f1-f0)*z**6.65
    keep=n>=1e-30
    integrand=np.zeros_like(r)
    integrand[keep]=4.0*np.pi*r[keep]**2*(-1.451)*n[keep]**(4.0/3.0)*f[keep]
    return _radial_trapezoid(r,integrand)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    return [{'setup': 'import numpy as np; r=np.geomspace(1e-6,60.0*4**2,20001)/8.0; r_gold=r.copy(); ing=p_shell_semilocal_ingredients(4,8.0,r); ing_gold=_oracle_p_shell_semilocal_ingredients(4,8.0,r_gold)', 'call': 'epc_w_inf(r,ing[2],ing[6],ing[7])', 'gold_call': '_oracle_epc_w_inf(r_gold,ing_gold[2],ing_gold[6],ing_gold[7])'}, {'setup': 'import numpy as np; r=np.geomspace(1e-6,60.0*4**2,20001)/4.5; r_gold=r.copy(); ing=p_shell_semilocal_ingredients(4,4.5,r); ing_gold=_oracle_p_shell_semilocal_ingredients(4,4.5,r_gold)', 'call': 'epc_w_inf(r,ing[2],ing[6],ing[7])', 'gold_call': '_oracle_epc_w_inf(r_gold,ing_gold[2],ing_gold[6],ing_gold[7])'}, {'setup': 'import numpy as np; r=np.geomspace(1e-6,60.0*3**2,20001)/5.0; r_gold=r.copy(); ing=p_shell_semilocal_ingredients(3,5.0,r); ing_gold=_oracle_p_shell_semilocal_ingredients(3,5.0,r_gold)', 'call': 'epc_w_inf(r,ing[2],ing[6],ing[7])', 'gold_call': '_oracle_epc_w_inf(r_gold,ing_gold[2],ing_gold[6],ing_gold[7])'}, {'setup': 'import numpy as np; r=np.geomspace(1e-6,60.0*4**2,4001)/12.0; r_gold=r.copy(); ing=p_shell_semilocal_ingredients(4,12.0,r); ing_gold=_oracle_p_shell_semilocal_ingredients(4,12.0,r_gold)', 'call': 'epc_w_inf(r,ing[2],ing[6],ing[7])', 'gold_call': '_oracle_epc_w_inf(r_gold,ing_gold[2],ing_gold[6],ing_gold[7])'}]
