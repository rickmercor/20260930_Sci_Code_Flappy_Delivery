"""
Given a spherical spin-unpolarized density, reduced gradient and iso-orbital indicator on a radial grid, evaluate W'_inf with the full primary-paper ePC interpolation between the branches from the previous step.

Along the density-fixed adiabatic connection the

integrand W_alpha[n] behaves at large coupling as

W_alpha -> W_inf[n] + W'_inf[n] alpha^{-1/2} + ... . In the

strictly-correlated-electron (SCE) picture W_inf is the SCE interaction energy

minus the Hartree energy, and W'_inf is the energy of the zero-point

oscillations of the electrons around the SCE configurations,

W'_inf = (1/(4N)) integral n(r) sum_mu omega_mu(r) d^3r, with omega_mu the

normal-mode frequencies. W'_inf is positive for any many-electron density and

zero for a one-electron density. Exact SCE values are costly and exist only

for a few systems, so interaction-strength interpolation functionals use

semilocal models of W'_inf built on the point-charge-plus-continuum (PC)

picture.

Returns
-------
a float, W'_inf in hartree. Raise ``ValueError`` if array lengths differ, r is not strictly increasing, density is negative, or z lies outside [0,1].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def epc_wprime_inf(r: "np.ndarray", n: "np.ndarray", s: "np.ndarray", z: "np.ndarray") -> float:
    """Return W'_inf[n] in hartree from the published ePC meta-GGA.

    Inputs share a strictly increasing radial grid. The density is
    spin-unpolarized (zeta=0). Use ``epc_wprime_branches`` and recover from
    the primary paper the z-dependent interpolation between its two branches.
    Use C=1.535 and integrate 4*pi*r^2*C*n^(3/2)*F' by the trapezoidal rule;
    n<1e-30 contributes zero.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _radial_trapezoid(r, f):
    return float(np.sum(0.5*(f[1:]+f[:-1])*np.diff(r)))


def _oracle_epc_wprime_inf(r: np.ndarray, n: np.ndarray, s: np.ndarray, z: np.ndarray) -> float:
    r=np.asarray(r,dtype=float); n=np.asarray(n,dtype=float); s=np.asarray(s,dtype=float); z=np.asarray(z,dtype=float)
    if not (len(r)==len(n)==len(s)==len(z)):
        raise ValueError("Arrays must have the same length.")
    if np.any(np.diff(r)<=0.0) or np.any(n<0.0) or np.any((z<0.0)|(z>1.0)):
        raise ValueError("Invalid radial grid, density or z.")
    f0,f1=_oracle_epc_wprime_branches(s,0.0)
    f=f0+(z**11*f1-f0)*z**2
    keep=n>=1e-30
    integrand=np.zeros_like(r)
    integrand[keep]=4.0*np.pi*r[keep]**2*1.535*n[keep]**1.5*f[keep]
    return _radial_trapezoid(r,integrand)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    return [{'setup': 'import numpy as np; r=np.geomspace(1e-6,60.0*4**2,20001)/7.0; r_gold=r.copy(); ing=p_shell_semilocal_ingredients(4,7.0,r); ing_gold=_oracle_p_shell_semilocal_ingredients(4,7.0,r_gold)', 'call': 'epc_wprime_inf(r,ing[2],ing[6],ing[7])', 'gold_call': '_oracle_epc_wprime_inf(r_gold,ing_gold[2],ing_gold[6],ing_gold[7])'}, {'setup': 'import numpy as np; r=np.geomspace(1e-6,60.0*4**2,20001)/3.5; r_gold=r.copy(); ing=p_shell_semilocal_ingredients(4,3.5,r); ing_gold=_oracle_p_shell_semilocal_ingredients(4,3.5,r_gold)', 'call': 'epc_wprime_inf(r,ing[2],ing[6],ing[7])', 'gold_call': '_oracle_epc_wprime_inf(r_gold,ing_gold[2],ing_gold[6],ing_gold[7])'}, {'setup': 'import numpy as np; r=np.geomspace(1e-6,60.0*3**2,20001)/8.0; r_gold=r.copy(); ing=p_shell_semilocal_ingredients(3,8.0,r); ing_gold=_oracle_p_shell_semilocal_ingredients(3,8.0,r_gold)', 'call': 'epc_wprime_inf(r,ing[2],ing[6],ing[7])', 'gold_call': '_oracle_epc_wprime_inf(r_gold,ing_gold[2],ing_gold[6],ing_gold[7])'}, {'setup': 'import numpy as np; r=np.geomspace(1e-6,60.0*2**2,20001)/9.0; r_gold=r.copy(); ing=p_shell_semilocal_ingredients(2,9.0,r); ing_gold=_oracle_p_shell_semilocal_ingredients(2,9.0,r_gold)', 'call': 'epc_wprime_inf(r,ing[2],ing[6],ing[7])', 'gold_call': '_oracle_epc_wprime_inf(r_gold,ing_gold[2],ing_gold[6],ing_gold[7])'}, {'setup': 'import numpy as np; r=np.geomspace(1e-6,60.0*3**2,4001)/11.0; r_gold=r.copy(); ing=p_shell_semilocal_ingredients(3,11.0,r); ing_gold=_oracle_p_shell_semilocal_ingredients(3,11.0,r_gold)', 'call': 'epc_wprime_inf(r,ing[2],ing[6],ing[7])', 'gold_call': '_oracle_epc_wprime_inf(r_gold,ing_gold[2],ing_gold[6],ing_gold[7])'}]
