"""
Step 01: Spin-polarized LSD exchange-correlation energy and potentials (Slater plus VWN5). Spin-polarized local density approximation for exchange and correlation.

In the local spin density approximation the exchange-correlation energy of an inhomogeneous electron gas is built from the energy per electron of a uniform gas with the same local densities of up-spin and down-spin electrons, and each spin channel gets its own potential, the derivative of that energy with respect to its own density. Exchange couples only electrons of like spin, so the exchange energy of a polarized gas follows exactly from that of the unpolarized gas by spin scaling. Correlation has no such exact relation. The standard spin-dependent form is the Vosko, Wilk and Nusair (VWN5) interpolation, which combines their fits to the Ceperley-Alder correlation energies of the paramagnetic and ferromagnetic gases with a separate fit for the spin stiffness, so that the correlation energy is exact to second order in the relative spin polarization and exact at full polarization; it is the correlation functional of the NIST reference LSD calculations for atoms. Simpler interpolations that use only the paramagnetic and ferromagnetic limits give nearly the same total energies of open-shell atoms but noticeably different minority-spin potentials, and an unpolarized functional applied to the total density ignores the spin splitting altogether. The two spin potentials differ most where the relative polarization is large, as in the valence tail of an atom with an unpaired electron, and they keep a small finite value wherever the density is positive.

Returns
-------
numpy.ndarray of shape (3,) + rho_up.shape: LSD exchange-correlation energy per electron (row 0) and up-spin and down-spin potentials (rows 1 and 2) in hartree, Slater exchange plus VWN5
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lsd_exchange_correlation(rho_up: "np.ndarray", rho_down: "np.ndarray") -> "np.ndarray":
    '''Exchange-correlation energy per electron and spin potentials of the LSD approximation (Slater exchange plus VWN5).

    Parameters
    ----------
    rho_up : np.ndarray
        Up-spin electron density in bohr^-3, any shape, every entry finite and >= 0.
    rho_down : np.ndarray
        Down-spin electron density in bohr^-3, same shape as rho_up, every entry finite and >= 0.

    Returns
    -------
    result : np.ndarray
        Array of shape (3,) + rho_up.shape and dtype float in hartree. result[0] is the exchange-correlation energy per
        electron eps_xc(rho_up, rho_down), and result[1] and result[2] are the up-spin and down-spin potentials
        v_xc,sigma = d(rho eps_xc)/d rho_sigma with rho = rho_up + rho_down. Exchange is Slater exchange with exact spin
        scaling; correlation is the spin-polarized Vosko-Wilk-Nusair (VWN5) form built from the paramagnetic and
        ferromagnetic Ceperley-Alder fits and the spin-stiffness fit. Where rho is 0 all three entries are 0; every
        positive total density, including fully polarized points where one spin density is 0, is evaluated from the
        functional itself.

    Raises
    ------
    ValueError
        If the two arrays differ in shape or any entry is negative or not finite.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _vwn_fit(x, A, x0, b, c):
    import numpy as np
    X = x * x + b * x + c
    X0 = x0 * x0 + b * x0 + c
    Q = np.sqrt(4.0 * c - b * b)
    at = np.arctan(Q / (2.0 * x + b))
    g = A * (np.log(x * x / X) + 2.0 * b / Q * at
             - b * x0 / X0 * (np.log((x - x0) ** 2 / X) + 2.0 * (b + 2.0 * x0) / Q * at))
    dX = 2.0 * x + b
    dat = -2.0 * Q / (dX * dX + Q * Q)
    dg = A * (2.0 / x - dX / X + 2.0 * b / Q * dat
              - b * x0 / X0 * (2.0 / (x - x0) - dX / X + 2.0 * (b + 2.0 * x0) / Q * dat))
    return g, dg

def _oracle_lsd_exchange_correlation(rho_up: "np.ndarray", rho_down: "np.ndarray") -> "np.ndarray":
    import numpy as np
    ru = np.asarray(rho_up, dtype=float)
    rd = np.asarray(rho_down, dtype=float)
    if ru.shape != rd.shape:
        raise ValueError("spin densities must have the same shape")
    if not (np.all(np.isfinite(ru)) and np.all(np.isfinite(rd))) or np.any(ru < 0.0) or np.any(rd < 0.0):
        raise ValueError("spin densities must be finite and non-negative")
    out = np.zeros((3,) + ru.shape)
    pos = (ru + rd) > 0.0
    u = ru[pos]
    d = rd[pos]
    n = u + d
    one_plus = 2.0 * u / n
    one_minus = 2.0 * d / n
    z = one_plus - 1.0
    cx = (6.0 / np.pi) ** (1.0 / 3.0)
    ex_energy = -0.75 * cx * (u * np.cbrt(u) + d * np.cbrt(d))
    x = np.sqrt(np.cbrt(3.0 / (4.0 * np.pi)) / np.cbrt(n))
    eP, dP = _vwn_fit(x, 0.0310907, -0.10498, 3.72744, 12.9352)
    eF, dF = _vwn_fit(x, 0.01554535, -0.32500, 7.06042, 18.0578)
    eA, dA = _vwn_fit(x, -1.0 / (6.0 * np.pi ** 2), -0.0047584, 1.13107, 13.0045)
    fden = 2.0 ** (4.0 / 3.0) - 2.0
    f = (one_plus * np.cbrt(one_plus) + one_minus * np.cbrt(one_minus) - 2.0) / fden
    df = (4.0 / 3.0) * (np.cbrt(one_plus) - np.cbrt(one_minus)) / fden
    fpp0 = 8.0 / (9.0 * fden)
    z3 = z ** 3
    z4 = z3 * z
    ec = eP + eA * f / fpp0 * (1.0 - z4) + (eF - eP) * f * z4
    dec_dx = dP + dA * f / fpp0 * (1.0 - z4) + (dF - dP) * f * z4
    dec_dz = eA / fpp0 * (df * (1.0 - z4) - 4.0 * z3 * f) + (eF - eP) * (df * z4 + 4.0 * z3 * f)
    common = ec - x / 6.0 * dec_dx
    out[0][pos] = ex_energy / n + ec
    out[1][pos] = -cx * np.cbrt(u) + common + one_minus * dec_dz
    out[2][pos] = -cx * np.cbrt(d) + common - one_plus * dec_dz
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: partially polarized densities across the core, valence and tail of an open-shell atom ---
        {"setup": "import numpy as np\nru = np.array([20.0, 1.9, 0.20, 0.018, 1.1e-3, 3.0e-5])\nrd = np.array([18.5, 1.2, 0.05, 0.002, 4.0e-5, 1.0e-8])\n",
         "call": "lsd_exchange_correlation(ru.copy(), rd.copy())",
         "gold_call": "_oracle_lsd_exchange_correlation(ru.copy(), rd.copy())", "tol": 1e-8},
        # --- Normal: a two-dimensional array with the spins swapped in one row ---
        {"setup": "import numpy as np\nru = np.array([[0.9, 0.3, 0.02], [0.1, 0.6, 5.0]])\nrd = np.array([[0.1, 0.6, 5.0], [0.9, 0.3, 0.02]])\n",
         "call": "lsd_exchange_correlation(ru.copy(), rd.copy())",
         "gold_call": "_oracle_lsd_exchange_correlation(ru.copy(), rd.copy())", "tol": 1e-8},
        # --- Boundary: unpolarized points reduce to the paramagnetic functional ---
        {"setup": "import numpy as np\nru = np.array([2.5, 0.04, 7.0e-6])\nrd = ru.copy()\n",
         "call": "lsd_exchange_correlation(ru.copy(), rd.copy())",
         "gold_call": "_oracle_lsd_exchange_correlation(ru.copy(), rd.copy())", "tol": 1e-8},
        # --- Boundary: fully polarized points, including the empty spin channel, and very low densities ---
        {"setup": "import numpy as np\nru = np.array([0.7, 0.0, 1e-12, 1e-40])\nrd = np.array([0.0, 0.3, 0.0, 1e-44])\n",
         "call": "lsd_exchange_correlation(ru.copy(), rd.copy())",
         "gold_call": "_oracle_lsd_exchange_correlation(ru.copy(), rd.copy())", "tol": 1e-8},
        # --- Edge: zero total density ---
        {"setup": "import numpy as np\nru = np.array([0.0, 0.4])\nrd = np.array([0.0, 0.0])\n",
         "call": "lsd_exchange_correlation(ru.copy(), rd.copy())",
         "gold_call": "_oracle_lsd_exchange_correlation(ru.copy(), rd.copy())", "tol": 1e-8},
        # --- Error: a negative spin density ---
        {"setup": "import numpy as np\ndef _probe(fn):\n    try:\n        fn(np.array([0.2, 0.1]), np.array([0.1, -1e-9]))\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(lsd_exchange_correlation)", "gold_call": "_probe(_oracle_lsd_exchange_correlation)"},
    ]
