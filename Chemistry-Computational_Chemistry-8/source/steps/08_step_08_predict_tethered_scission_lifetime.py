"""
Infer both the bending stiffness and the equilibrium bond angle of a Morse chain from its free-strand lifetime and terminal-scission fraction, then predict its tethered lifetime at another force. Orchestrator: yes; In each trial pair and the prediction, use all seven preceding reference steps to build prefactors, propagate the angular statistics, converge the bond-specific dividing surfaces, evaluate the PMFs and sum the bond-resolved rates.

A lifetime alone does not separately identify the strength and preferred angle of bending. The spatial distribution of first-scission events provides an independent observable because orientational correlations change the terminal and interior barriers differently.

Returns
-------
float: predicted mean time to first scission of the tethered strand in picoseconds.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def predict_tethered_scission_lifetime(
    measured_lifetime_ns: float = 18.921926,
    measured_terminal_share: float = 0.99662638,
    measured_force_reduced: float = 0.94,
    predicted_force_reduced: float = 1.04,
    n_bonds: int = 10,
    beta_de: float = 279.0,
    a_le: float = 2.15,
    atom_mass_kg: float = 1.99e-26,
    bond_length_m: float = 1.525e-10,
    temperature_k: float = 293.15,
    n_theta: int = 160,
    n_omega: int = 256,
    n_length: int = 600,
    stiffness_low: float = 1000.0,
    stiffness_high: float = 8000.0,
    angle_low_deg: float = 50.0,
    angle_high_deg: float = 100.0,
) -> float:
    """Return the predicted tethered mean time to first scission, in ps.

    The calibration data belong to a free strand at ``measured_force_reduced``.
    ``measured_terminal_share`` is the probability that the first broken bond is
    either terminal bond. Infer ``s = beta k_phi pi**2`` and the equilibrium
    bond-vector angle simultaneously. For each pair, converge the bond-specific
    rupture thresholds from 2.0 until no threshold changes by ``1e-11`` (at most
    60 sweeps), build the bond-resolved rates, and match the log lifetime and the
    log odds of the combined terminal share. Use bounded nonlinear least squares
    in ``(log(s), angle_deg)`` over the supplied bounds, starting at both interval
    midpoints, with ``xtol = ftol = gtol = 1e-10`` and at most 100 evaluations.
    Reject a fit whose largest residual exceeds ``2e-6``. With the inferred pair,
    repeat the same calculation with atom 0 fixed at
    ``predicted_force_reduced`` and return the lifetime in picoseconds.

    Every PMF uses ``n_length`` Gauss-Legendre nodes on ``[0.5, x_b]`` plus its
    converged barrier top. The kinetic prefactors, bending kernel, intact weights,
    angular messages, stationary points, PMFs and rates must come from the seven
    preceding functions. The defaults reproduce the problem statement.

    Raises
    ------
    ValueError
        If a measured observable or fit bound is invalid, fewer than three bonds
        are requested, the observations cannot be matched inside the bounds, a
        consuming step rejects an argument, or threshold iteration fails.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_predict_tethered_scission_lifetime(
    measured_lifetime_ns: float = 18.921926,
    measured_terminal_share: float = 0.99662638,
    measured_force_reduced: float = 0.94,
    predicted_force_reduced: float = 1.04,
    n_bonds: int = 10,
    beta_de: float = 279.0,
    a_le: float = 2.15,
    atom_mass_kg: float = 1.99e-26,
    bond_length_m: float = 1.525e-10,
    temperature_k: float = 293.15,
    n_theta: int = 160,
    n_omega: int = 256,
    n_length: int = 600,
    stiffness_low: float = 1000.0,
    stiffness_high: float = 8000.0,
    angle_low_deg: float = 50.0,
    angle_high_deg: float = 100.0,
) -> float:
    """Reference two-observable inverse fit using only the reference chain."""
    import numpy as np
    from scipy.optimize import least_squares

    scalars = (
        measured_lifetime_ns, measured_terminal_share, stiffness_low,
        stiffness_high, angle_low_deg, angle_high_deg,
    )
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("measurements and fit bounds must be finite")
    if measured_lifetime_ns <= 0.0 or not 0.0 < measured_terminal_share < 1.0:
        raise ValueError("need a positive lifetime and a terminal share strictly between zero and one")
    if isinstance(n_bonds, bool) or not isinstance(n_bonds, (int, np.integer)) or n_bonds < 3:
        raise ValueError("n_bonds must be an integer of at least 3")
    if not 0.0 < stiffness_low < stiffness_high:
        raise ValueError("need 0 < stiffness_low < stiffness_high")
    if not 0.0 <= angle_low_deg < angle_high_deg <= 180.0:
        raise ValueError("angle bounds must be ordered inside [0, 180]")

    def _rates(stiffness, angle_deg, force, tethered):
        prefactors = _oracle_compute_bond_kinetic_prefactors(
            n_bonds, atom_mass_kg, bond_length_m, temperature_k, tethered
        )
        link_stiffness = np.full(int(n_bonds) - 1, float(stiffness))
        link_angles = np.full(int(n_bonds) - 1, float(angle_deg))
        log_kernels = _oracle_build_bending_kernel(
            n_theta, n_omega, link_stiffness, link_angles
        )
        thresholds = np.full(int(n_bonds), 2.0)
        for _ in range(60):
            log_intact = _oracle_compute_intact_bond_weights(
                thresholds, force, beta_de, a_le, n_theta, n_length
            )
            weights = _oracle_compute_angular_weights(log_intact, log_kernels)
            points = _oracle_locate_pmf_stationary_points(weights, force, beta_de, a_le)
            change = float(np.max(np.abs(points[:, 1] - thresholds)))
            thresholds = points[:, 1].copy()
            if change < 1e-11:
                break
        else:
            raise ValueError("rupture thresholds did not converge within 60 sweeps")

        nodes, _ = np.polynomial.legendre.leggauss(int(n_length))
        table = np.empty((thresholds.size, nodes.size + 1))
        for i, top in enumerate(thresholds):
            half = 0.5 * (top - 0.5)
            x = np.concatenate(([top], half * nodes + 0.5 + half))
            table[i] = _oracle_evaluate_bond_pmf(
                x, weights[i], force, beta_de, a_le
            )[:, 0]
        return _oracle_compute_bond_scission_rates(table, thresholds, prefactors)

    target_log_lifetime = np.log(float(measured_lifetime_ns) * 1.0e-9)
    target_log_odds = np.log(float(measured_terminal_share)) - np.log1p(-float(measured_terminal_share))

    def _residual(parameters):
        stiffness = np.exp(parameters[0])
        rates = _rates(stiffness, parameters[1], measured_force_reduced, False)
        terminal_rate = float(rates[0] + rates[-1])
        interior_rate = float(np.sum(rates[1:-1]))
        log_lifetime = -np.log(float(np.sum(rates)))
        log_odds = np.log(terminal_rate) - np.log(interior_rate)
        return np.array([log_lifetime - target_log_lifetime, log_odds - target_log_odds])

    lower = np.array([np.log(float(stiffness_low)), float(angle_low_deg)])
    upper = np.array([np.log(float(stiffness_high)), float(angle_high_deg)])
    start = 0.5 * (lower + upper)
    fit = least_squares(
        _residual, start, bounds=(lower, upper),
        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=100,
    )
    residual = _residual(fit.x)
    if not fit.success or not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 2e-6:
        raise ValueError("the two observations cannot be matched inside the parameter bounds")

    predicted_rates = _rates(
        np.exp(fit.x[0]), fit.x[1], predicted_force_reduced, True
    )
    return float(1.0e12 / np.sum(predicted_rates))

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    raises = (
        "import numpy as np\n"
        "def _candidate():\n    try:\n        predict_tethered_scission_lifetime({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
        "def _reference():\n    try:\n        _oracle_predict_tethered_scission_lifetime({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "float(np.log(predict_tethered_scission_lifetime(18.495691360353867, 0.9966053877309246, 0.94, 1.04, 10, 279.0, 2.15, 1.99e-26, 1.525e-10, 293.15, 48, 96, 120, 1500.0, 6000.0, 55.0, 95.0)))",
            "gold_call": "float(np.log(_oracle_predict_tethered_scission_lifetime(18.495691360353867, 0.9966053877309246, 0.94, 1.04, 10, 279.0, 2.15, 1.99e-26, 1.525e-10, 293.15, 48, 96, 120, 1500.0, 6000.0, 55.0, 95.0)))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(np.log(predict_tethered_scission_lifetime(386.4778783783167, 0.9887551148538233, 0.90, 1.00, 10, 279.0, 2.15, 1.99e-26, 1.525e-10, 293.15, 40, 80, 100, 1000.0, 5000.0, 50.0, 90.0)))",
            "gold_call": "float(np.log(_oracle_predict_tethered_scission_lifetime(386.4778783783167, 0.9887551148538233, 0.90, 1.00, 10, 279.0, 2.15, 1.99e-26, 1.525e-10, 293.15, 40, 80, 100, 1000.0, 5000.0, 50.0, 90.0)))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(np.log(predict_tethered_scission_lifetime(18.495691360353867, 0.9966053877309246, 0.94, 1.00, 10, 279.0, 2.15, 1.99e-26, 1.525e-10, 293.15, 48, 96, 120, 1500.0, 6000.0, 55.0, 95.0)))",
            "gold_call": "float(np.log(_oracle_predict_tethered_scission_lifetime(18.495691360353867, 0.9966053877309246, 0.94, 1.00, 10, 279.0, 2.15, 1.99e-26, 1.525e-10, 293.15, 48, 96, 120, 1500.0, 6000.0, 55.0, 95.0)))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(np.log(predict_tethered_scission_lifetime(700579.50179393345, 0.99953647124593614, 0.86, 0.99, 5, 279.0, 2.15, 1.99e-26, 1.525e-10, 293.15, 28, 56, 80, 1200.0, 5000.0, 60.0, 105.0)))",
            "gold_call": "float(np.log(_oracle_predict_tethered_scission_lifetime(700579.50179393345, 0.99953647124593614, 0.86, 0.99, 5, 279.0, 2.15, 1.99e-26, 1.525e-10, 293.15, 28, 56, 80, 1200.0, 5000.0, 60.0, 105.0)))",
        },
        {
            "setup": raises.replace("{args}", "18.0, 0.99, 0.94, 1.04, 2, 279.0, 2.15, 1.99e-26, 1.525e-10, 293.15, 16, 32, 60, 1000.0, 8000.0, 50.0, 100.0"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {
            "setup": raises.replace("{args}", "-1.0, 0.99, 0.94, 1.04, 10, 279.0, 2.15, 1.99e-26, 1.525e-10, 293.15, 16, 32, 60, 1000.0, 8000.0, 50.0, 100.0"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {
            "setup": raises.replace("{args}", "18.0, 1.0, 0.94, 1.04, 10, 279.0, 2.15, 1.99e-26, 1.525e-10, 293.15, 16, 32, 60, 1000.0, 8000.0, 50.0, 100.0"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {
            "setup": raises.replace("{args}", "18.0, 0.99, 0.94, 1.04, 10, 279.0, 2.15, 1.99e-26, 1.525e-10, 293.15, 16, 32, 60, 8000.0, 1000.0, 50.0, 100.0"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
    ]
