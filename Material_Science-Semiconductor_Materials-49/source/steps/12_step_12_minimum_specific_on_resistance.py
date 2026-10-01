"""
Minimum drift specific on-resistance of the optimised punch-through design with the two carrier coefficients kept separate.

The drift contribution to specific on-resistance is the layer thickness divided by the product of elementary charge, electron mobility and the density of carriers actually available to conduct. At each trial doping the critical field comes from the two-carrier breakdown condition with the electron and hole coefficients kept separate, the voltage rating then fixes the terminal-to-peak field ratio and the thickness, the conducting carrier density is the smaller free electron density of the neutral layer, and the mobility is set by scattering from the full chemical doping. The doping is chosen to minimise this resistance, with the doping-dependent mobility and the free electron density inside the objective, along the one-parameter family of designs that block the rated voltage; first locate the minimum with the source's collapsed coefficients as a starting point, then locate the two-carrier minimum to a relative precision in the resistance of at least 1e-10. Report the answer in milliohm centimetre squared.

Returns
-------
float, the drift specific on-resistance in mOhm cm^2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def minimum_specific_on_resistance(a_n: float, b_n: float, a_p: float, b_p: float, eps_r: float, bv_target: float,
                                   n_c: float, degeneracy: float, ea_shallow: float, ea_deep: float, kt: float) -> float:
    '''Minimum drift specific on-resistance of the optimised punch-through design.

    Parameters
    ----------
    a_n, a_p : float
        Ionization prefactors in cm^-1.
    b_n, b_p : float
        Ionization field scales in V/cm.
    eps_r : float
        Relative permittivity.
    bv_target : float
        Target blocking voltage in V.
    n_c : float
        Conduction band effective density of states in cm^-3.
    degeneracy : float
        Donor degeneracy factor.
    ea_shallow, ea_deep : float
        Activation energies of the two donor sites in eV.
    kt : float
        Thermal energy in eV.

    Returns
    -------
    float
        Specific on-resistance in mOhm cm^2.

    Raises
    ------
    ValueError
        If any of a_n, b_n, a_p, b_p, eps_r, bv_target, n_c, degeneracy, kt is not finite or is not strictly positive.
        Activation energies must be finite and non-negative.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize_scalar

def _oracle_minimum_specific_on_resistance(a_n: float, b_n: float, a_p: float, b_p: float, eps_r: float, bv_target: float,
                                           n_c: float, degeneracy: float, ea_shallow: float, ea_deep: float, kt: float) -> float:
    for _v in (a_n, b_n, a_p, b_p, eps_r, bv_target, n_c, degeneracy, kt):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('a_n, b_n, a_p, b_p, eps_r, bv_target, n_c, degeneracy, kt must be finite and strictly positive')
    for _v in (ea_shallow, ea_deep):
        if not np.isfinite(_v) or _v < 0.0:
            raise ValueError('activation energies must be finite and non-negative')
    q = 1.602e-19
    a_eff, b_eff = _oracle_effective_ionization_parameters(a_n, b_n, a_p, b_p)

    def _design_resistance(nd: float, ec: float) -> float:
        try:
            eta = _oracle_field_ratio_at_rating(eps_r, nd, ec, bv_target)
        except ValueError:
            return np.inf
        w = _oracle_punchthrough_depletion_width(eps_r, nd, ec, eta)
        n = _oracle_free_electron_density(nd, n_c, degeneracy, ea_shallow, ea_deep, kt)
        return 1.0e3*w/(q*_oracle_drift_electron_mobility(nd)*n)

    def _collapsed_ecr(nd: float) -> float:
        return _oracle_critical_field(b_eff, _oracle_avalanche_zeta(
            _oracle_ionization_dimensionless_group(a_eff, b_eff, eps_r, nd)))

    def _r_collapsed(nd: float) -> float:
        return _design_resistance(nd, _collapsed_ecr(nd))

    def _r_two_carrier(nd: float) -> float:
        return _design_resistance(nd, _oracle_two_carrier_critical_field(a_n, b_n, a_p, b_p, eps_r, nd))

    def _bv_pub(nd: float) -> float:
        z = _oracle_avalanche_zeta(_oracle_ionization_dimensionless_group(a_eff, b_eff, eps_r, nd))
        eta = _oracle_punchthrough_field_ratio(z)
        if not (0.0 <= eta < 1.0):
            return np.inf
        return _oracle_punchthrough_breakdown_voltage(eps_r, nd, _oracle_critical_field(b_eff, z), eta) - bv_target

    def _scan_minimum(fun, centre: float):
        grid = centre*np.logspace(np.log10(0.5), np.log10(1.5), 61)
        vals = np.array([fun(x) for x in grid])
        if not np.any(np.isfinite(vals)):
            raise ValueError('no punch-through design reaches bv_target near the starting doping')
        i = int(np.argmin(np.where(np.isfinite(vals), vals, np.inf)))
        a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        res = minimize_scalar(fun, bounds=(a, b), method='bounded', options=dict(xatol=1.0e-9*grid[i]))
        return (res.x, fun(res.x)) if fun(res.x) <= vals[i] else (grid[i], vals[i])

    lo, hi = 1.0e12, 1.0e19
    while hi > lo and not np.isfinite(_bv_pub(hi)):
        hi *= 0.5
    if hi <= lo or _bv_pub(lo)*_bv_pub(hi) > 0.0:
        raise ValueError('no punch-through design reaches bv_target in the physical doping range')
    nd_pub = brentq(_bv_pub, lo, hi, xtol=1.0e2, rtol=8.9e-16, maxiter=300)
    nd_collapsed, _ = _scan_minimum(_r_collapsed, nd_pub)
    _, r_min = _scan_minimum(_r_two_carrier, nd_collapsed)
    return float(r_min)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.special import exp1\nfrom scipy.integrate import quad\nfrom scipy.optimize import brentq, minimize_scalar",
            "call": "minimum_specific_on_resistance(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 650.0, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
            "gold_call": "_oracle_minimum_specific_on_resistance(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 650.0, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import exp1\nfrom scipy.integrate import quad\nfrom scipy.optimize import brentq, minimize_scalar",
            "call": "minimum_specific_on_resistance(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 1200.0, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
            "gold_call": "_oracle_minimum_specific_on_resistance(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 1200.0, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import exp1\nfrom scipy.integrate import quad\nfrom scipy.optimize import brentq, minimize_scalar",
            "call": "minimum_specific_on_resistance(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 3300.0, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
            "gold_call": "_oracle_minimum_specific_on_resistance(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 3300.0, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import exp1\nfrom scipy.integrate import quad\nfrom scipy.optimize import brentq, minimize_scalar",
            "call": "minimum_specific_on_resistance(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 900.0, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
            "gold_call": "_oracle_minimum_specific_on_resistance(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 900.0, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import exp1\nfrom scipy.integrate import quad\nfrom scipy.optimize import brentq, minimize_scalar\ndef run_model():\n    try:\n        minimum_specific_on_resistance(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, -650.0, 1.7e19, 2, 0.060, 0.120, 0.025852)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_minimum_specific_on_resistance(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, -650.0, 1.7e19, 2, 0.060, 0.120, 0.025852)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
