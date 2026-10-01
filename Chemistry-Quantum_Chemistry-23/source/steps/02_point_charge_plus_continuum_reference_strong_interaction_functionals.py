"""
Given a spherical, spin-unpolarized density and its radial

derivative on a radial grid, return the two strong-interaction functionals

W_inf[n] and W'_inf[n] in the original gradient-corrected

point-charge-plus-continuum (PC) model.

Along the density-fixed adiabatic connection the

integrand W_alpha[n] behaves at large coupling strength alpha as

W_alpha -> W_inf[n] + W'_inf[n] alpha^{-1/2} + ... . In the

strictly-correlated-electron (SCE) picture W_inf is the SCE interaction

energy minus the Hartree energy (negative), and W'_inf is the energy of the

zero-point oscillations of the electrons around the SCE configurations

(positive for any many-electron density). The PC model of Seidl, Perdew and

Kurth estimates both from the electrostatics of a point charge in a cell of

uniform neutralizing background. It gives semilocal gradient expansions with

a local term and one gradient correction each.

Returns
-------
a tuple ``(W_inf, W_prime_inf)`` of two floats in hartree. Raise ``ValueError`` if the arrays differ in length or if ``r`` is not strictly increasing.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pc_strong_interaction(r: "np.ndarray", n: "np.ndarray", dn: "np.ndarray") -> tuple[float, float]:

    """Return (W_inf, W'_inf) in hartree from the gradient-corrected PC model.



    Inputs are arrays on a common, strictly increasing radial grid ``r``

    (bohr): the spherical, spin-summed density ``n`` and its radial

    derivative ``dn``. Use the original gradient-corrected PC model with
    A=-1.451, B=5.317e-3, C=1.535 and D=-0.02558. Evaluate each radial
    integral with the trapezoidal rule on the supplied grid points. Grid

    points where n < 1e-30 contribute zero. Use the published integrands, with no cap on the gradient terms.



    Expected return: a tuple ``(W_inf, W_prime_inf)`` of two floats in

    hartree. Raise ``ValueError`` if the arrays differ in length or if

    ``r`` is not strictly increasing.

    """

    return (0.0, 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np





def _radial_trapezoid(r, f):

    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(r)))





def _oracle_pc_strong_interaction(r: np.ndarray, n: np.ndarray, dn: np.ndarray) -> tuple[float, float]:

    r = np.asarray(r, dtype=float)

    n = np.asarray(n, dtype=float)

    dn = np.asarray(dn, dtype=float)

    if not (len(r) == len(n) == len(dn)):

        raise ValueError("Arrays must have the same length.")

    if np.any(np.diff(r) <= 0.0):

        raise ValueError("The grid must be strictly increasing.")

    A, B, C, D = -1.451, 5.317e-3, 1.535, -0.02558

    keep = n >= 1e-30

    f_w = np.zeros_like(r)

    f_wp = np.zeros_like(r)

    nk, g2 = n[keep], dn[keep] ** 2

    w4 = 4.0 * np.pi * r[keep] ** 2

    f_w[keep] = w4 * (A * nk ** (4.0 / 3.0) + B * g2 / nk ** (4.0 / 3.0))

    f_wp[keep] = w4 * (C * nk**1.5 + D * g2 / nk ** (7.0 / 6.0))

    return (_radial_trapezoid(r, f_w), _radial_trapezoid(r, f_wp))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    return [{'setup': 'import numpy as np; r = np.geomspace(1e-6, 240.0, 20001) / 7.0; r_gold=r.copy(); ing = p_shell_semilocal_ingredients(2, 7.0, r); ing_gold=_oracle_p_shell_semilocal_ingredients(2, 7.0, r_gold)', 'call': 'pc_strong_interaction(r, ing[2], ing[3])', 'gold_call': '_oracle_pc_strong_interaction(r_gold, ing_gold[2], ing_gold[3])'}, {'setup': 'import numpy as np; r = np.geomspace(1e-6, 240.0, 20001) / 3.0; r_gold=r.copy(); ing = p_shell_semilocal_ingredients(2, 3.0, r); ing_gold=_oracle_p_shell_semilocal_ingredients(2, 3.0, r_gold)', 'call': 'pc_strong_interaction(r, ing[2], ing[3])', 'gold_call': '_oracle_pc_strong_interaction(r_gold, ing_gold[2], ing_gold[3])'}, {'setup': 'import numpy as np; r = np.geomspace(1e-6, 240.0, 20001) / 9.5; r_gold=r.copy(); ing = p_shell_semilocal_ingredients(2, 9.5, r); ing_gold=_oracle_p_shell_semilocal_ingredients(2, 9.5, r_gold)', 'call': 'pc_strong_interaction(r, ing[2], ing[3])', 'gold_call': '_oracle_pc_strong_interaction(r_gold, ing_gold[2], ing_gold[3])'}, {'setup': 'import numpy as np; r = np.geomspace(1e-5, 240.0, 1501) / 11.0; r_gold=r.copy(); ing = p_shell_semilocal_ingredients(2, 11.0, r); ing_gold=_oracle_p_shell_semilocal_ingredients(2, 11.0, r_gold)', 'call': 'pc_strong_interaction(r, ing[2], ing[3])', 'gold_call': '_oracle_pc_strong_interaction(r_gold, ing_gold[2], ing_gold[3])'}, {'setup': 'import numpy as np; r = np.linspace(1e-3, 60.0, 30001); r_gold=r.copy(); ing = p_shell_semilocal_ingredients(2, 5.0, r); ing_gold=_oracle_p_shell_semilocal_ingredients(2, 5.0, r_gold)', 'call': 'pc_strong_interaction(r, ing[2], ing[3])', 'gold_call': '_oracle_pc_strong_interaction(r_gold, ing_gold[2], ing_gold[3])'}, {'setup': 'import numpy as np; r=np.geomspace(1e-6,960.0,20001)/10.0; r_gold=r.copy(); ing=p_shell_semilocal_ingredients(4,10.0,r); ing_gold=_oracle_p_shell_semilocal_ingredients(4,10.0,r_gold)', 'call': 'pc_strong_interaction(r,ing[2],ing[3])', 'gold_call': '_oracle_pc_strong_interaction(r_gold,ing_gold[2],ing_gold[3])'}]
