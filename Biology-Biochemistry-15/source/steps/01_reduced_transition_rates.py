"""
Build the transition rates of the reduced two-state scheme of the polymerase, in which the rapid nucleotide binding/unbinding step is at quasi-equilibrium and the remaining transitions are the open-to-closed change of the polymerase holding a bound nucleotide, its reverse, the polymerization that releases pyrophosphate and reopens the polymerase, and the reverse depolymerization that binds pyrophosphate and closes it. Return rates[m, n, nn, :] = (w_EF, w_FE, w_pol, w_depol) for a copy unit m opposite template unit n when the template unit that follows the pair is nn. The quasi-equilibrium of the binding step enters through the competitive Michaelis-Menten denominator Q_n = 1 + [dATP]/K_An + [dTTP]/K_Tn of a template unit n: the closing rate is k_EF [mP] / (K_mn Q_n) with the denominator of the pair's own template unit, the reopening rate is k_FE and the polymerization rate is k_pol with no denominator, and the depolymerization rate is (k_pol / K_P) [PPi] / Q_nn with the denominator of the template unit that follows the pair, because the reverse step starts from the quasi-equilibrated binding state of the next site. Nucleotide index 0 = A and 1 = T; a pair (m, n) is copy unit m opposite template unit n, so (0, 1) and (1, 0) are correct and (0, 0) and (1, 1) incorrect. kin[m, n] = (K_mn in M, k_EF in 1/s, k_FE in 1/s, k_pol in 1/s) are the dissociation constant of the quasi-equilibrated nucleotide binding, the open-to-closed and closed-to-open conformational rate constants, and the polymerization rate constant of that pair. conc = ([dATP], [dTTP]) in M, ppi = [PPi] in M, kp_const = K_P in M, the pyrophosphorolysis constant relating the depolymerization and polymerization rate constants of every pair. template holds the units of one period of the template in the order in which they are copied; the template is its infinite periodic repetition.

A high-fidelity DNA polymerase binds the incoming nucleotide rapidly and reversibly, then closes around it before catalysing the phosphodiester bond and releasing pyrophosphate. When the binding step is much faster than the conformational and chemical steps, its quasi-equilibrium can be eliminated exactly and the process becomes a Markov jump process between open and closed states whose rates carry the nucleotide concentrations through saturable denominators.

Returns
-------
ndarray of float64, shape (2, 2, 2, 4): rates[m, n, nn] = (w_EF, w_FE, w_pol, w_depol) in 1/s.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reduced_transition_rates(kin: "np.ndarray", conc: "np.ndarray", ppi: float,
                                     kp_const: float) -> "np.ndarray":
    """Build the transition rates of the reduced two-state scheme of the polymerase, in which the rapid nucleotide binding/unbinding step is at quasi-equilibrium and the remaining transitions are the open-to-closed change of the polymerase holding a bound nucleotide, its reverse, the polymerization that releases pyrophosphate and reopens the polymerase, and the reverse depolymerization that binds pyrophosphate and closes it. Return rates[m, n, nn, :] = (w_EF, w_FE, w_pol, w_depol) for a copy unit m opposite template unit n when the template unit that follows the pair is nn. The quasi-equilibrium of the binding step enters through the competitive Michaelis-Menten denominator Q_n = 1 + [dATP]/K_An + [dTTP]/K_Tn of a template unit n: the closing rate is k_EF [mP] / (K_mn Q_n) with the denominator of the pair's own template unit, the reopening rate is k_FE and the polymerization rate is k_pol with no denominator, and the depolymerization rate is (k_pol / K_P) [PPi] / Q_nn with the denominator of the template unit that follows the pair, because the reverse step starts from the quasi-equilibrated binding state of the next site. Nucleotide index 0 = A and 1 = T; a pair (m, n) is copy unit m opposite template unit n, so (0, 1) and (1, 0) are correct and (0, 0) and (1, 1) incorrect. kin[m, n] = (K_mn in M, k_EF in 1/s, k_FE in 1/s, k_pol in 1/s) are the dissociation constant of the quasi-equilibrated nucleotide binding, the open-to-closed and closed-to-open conformational rate constants, and the polymerization rate constant of that pair. conc = ([dATP], [dTTP]) in M, ppi = [PPi] in M, kp_const = K_P in M, the pyrophosphorolysis constant relating the depolymerization and polymerization rate constants of every pair. template holds the units of one period of the template in the order in which they are copied; the template is its infinite periodic repetition.

    Parameters
    ----------
    kin : np.ndarray
        Kinetic table of shape (2, 2, 4), see the description.
    conc : np.ndarray
        Nucleotide concentrations ([dATP], [dTTP]) in M.
    ppi : float
        Pyrophosphate concentration [PPi] in M.
    kp_const : float
        Pyrophosphorolysis constant K_P in M.

    Returns
    -------
    rates : np.ndarray
        Reduced transition rates of shape (2, 2, 2, 4).

    Raises
    ------
    ValueError
        If kin is not (2, 2, 4) finite positive, conc is not two finite positive values, or ppi or kp_const is not finite positive.
    """
    return rates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_kin(kin):
    k = np.asarray(kin, dtype=np.float64)
    if k.shape != (2, 2, 4) or not np.all(np.isfinite(k)) or np.any(k <= 0.0):
        raise ValueError("kin must be a (2, 2, 4) array of finite positive constants")
    return k


def _check_positive(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(x)


def _oracle_reduced_transition_rates(kin: "np.ndarray", conc: "np.ndarray", ppi: float,
                                     kp_const: float) -> "np.ndarray":
    """Eqs. (6)-(12): transition rates of the reduced two-state scheme.

    Returns rates[m, n, nn, :] = (w_EF[m,n], w_FE[m,n], w_pol[m,n], w_depol[m,n,nn]) where nn is the
    template unit that FOLLOWS the pair m:n. Only the depolymerization rate depends on nn, through the
    Michaelis-Menten denominator Q_nn of the next quasi-equilibrated binding step (eq. 9); w_EF carries
    the competitive denominator Q_n of its own template unit (eq. 6); k_depol = k_pol / K_P (eq. 12).
    """
    k = _check_kin(kin)
    c = np.asarray(conc, dtype=np.float64).ravel()
    if c.shape != (2,) or not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        raise ValueError("conc must be two finite positive concentrations (dATP, dTTP)")
    ppi = _check_positive(ppi, "ppi")
    kp_const = _check_positive(kp_const, "kp_const")
    K = k[:, :, 0]
    # Q_n = 1 + sum_m [mP] / K_mn: competitive binding of both nucleotides at template unit n
    Q = 1.0 + np.array([np.sum(c / K[:, n]) for n in range(2)])
    rates = np.zeros((2, 2, 2, 4), dtype=np.float64)
    for m in range(2):
        for n in range(2):
            w_ef = k[m, n, 1] * c[m] / (K[m, n] * Q[n])
            w_fe = k[m, n, 2]
            w_pol = k[m, n, 3]
            for nn in range(2):
                w_dep = (k[m, n, 3] / kp_const) * ppi / Q[nn]
                rates[m, n, nn] = (w_ef, w_fe, w_pol, w_dep)
    return rates

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\n",
            "call": "np.asarray(reduced_transition_rates(kin, conc, ppi, kp_const))",
            "gold_call": "np.asarray(_oracle_reduced_transition_rates(kin, conc, ppi, kp_const))",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([1.0, 1.0])\nppi, kp_const = 1.0e-4, 5.6e-3\n",
            "call": "np.asarray(reduced_transition_rates(kin, conc, ppi, kp_const))",
            "gold_call": "np.asarray(_oracle_reduced_transition_rates(kin, conc, ppi, kp_const))",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[1.0e-2, 100.0, 200.0, 5.0], [5.0e-4, 5000.0, 2.0, 250.0]],\n                [[4.0e-4, 5000.0, 2.0, 300.0], [1.0e-2, 100.0, 200.0, 3.0]]])\ntemplate = np.array([0, 1, 1, 0, 1])\nconc = np.array([1.0e-6, 1.0e-6])\nppi, kp_const = 5.0e-4, 1.0e-3\n",
            "call": "np.asarray(reduced_transition_rates(kin, conc, ppi, kp_const))",
            "gold_call": "np.asarray(_oracle_reduced_transition_rates(kin, conc, ppi, kp_const))",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nppi, kp_const = 1.0e-4, 5.6e-3\ndef run_model():\n    try:\n        reduced_transition_rates(kin, np.array([2.0e-8, -4.0e-9]), ppi, kp_const)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_reduced_transition_rates(kin, np.array([2.0e-8, -4.0e-9]), ppi, kp_const)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
