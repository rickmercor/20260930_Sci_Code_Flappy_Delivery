"""
Predict the mean tension at the final scission of a reversible chain pulled at constant speed, starting from its clamped dwell times.

Dwell times measured at one clamped extension fix the chain model, which then yields the extension-dependent barriers, tension and rupture statistics of a constant-speed pull. Orchestrator: yes

Returns
-------
float: mean tension of the intact chain at its final scission during the constant-speed pull, in pN.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def predict_mean_final_rupture_force(hold_extension_nm: float, intact_dwell_s: float, broken_dwell_s: float, pulling_speed_nm_s: float, kuhn_length_nm: float, temperature_k: float, attempt_frequency_hz: float, n_min: int, n_max: int) -> float:
    """Return the mean tension (pN) of the intact chain at its final scission.

    Convert the clamped end-to-end distance to Kuhn lengths, the dwell times to
    units of the inverse attempt frequency and the pulling speed to the reduced
    loading rate (Kuhn lengths per inverse attempt frequency). Infer the segment
    count within ``[n_min, n_max]`` and the well depth with
    ``infer_segment_count_and_bond_energy``.

    Tabulate the pull on the nodes ``0.0, 0.5, 1.0, 1.5, ...`` Kuhn lengths, stopping
    before the first node at which ``locate_free_energy_extrema`` finds no intact
    state. At each node take the scission and re-formation barriers from
    ``compute_link_free_energy`` at those stationary points and the tension from
    ``compute_intact_chain_tension`` at the intact-state length. Interpolate the
    two barriers and the tension with not-a-knot cubic splines through the
    nodes onto the uniform grid from zero to the last node with spacing 0.001,
    starting the chain intact at zero. Propagate the intact probability with
    ``integrate_intact_probability``, obtain the mean reduced tension at the
    final scission with ``compute_final_scission_statistics``, and multiply it
    by ``k_B * temperature_k / kuhn_length`` with k_B = 1.380649e-23 J/K,
    reporting piconewtons.

    Parameters
    ----------
    hold_extension_nm : float
        Clamped end-to-end distance of the dwell-time measurement (nm).
    intact_dwell_s : float
        Mean intact dwell time at that distance (s).
    broken_dwell_s : float
        Mean broken dwell time at that distance (s).
    pulling_speed_nm_s : float
        Constant pulling speed (nm/s), positive.
    kuhn_length_nm : float
        Kuhn length (nm), positive.
    temperature_k : float
        Temperature (K), positive.
    attempt_frequency_hz : float
        Attempt frequency shared by scission and re-formation (1/s), positive.
    n_min : int
        Smallest segment count considered, at least 10.
    n_max : int
        Largest segment count considered, at least ``n_min``.

    Returns
    -------
    force : float
        Mean tension at the final scission in pN, as a native Python float.

    Raises
    ------
    ValueError
        If the pulling speed, Kuhn length, temperature or attempt frequency is
        not positive, or under the conditions raised by the steps it calls.
    """
    return force

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_predict_mean_final_rupture_force(hold_extension_nm: float, intact_dwell_s: float, broken_dwell_s: float, pulling_speed_nm_s: float, kuhn_length_nm: float, temperature_k: float, attempt_frequency_hz: float, n_min: int, n_max: int) -> float:
    """Reference implementation (end-to-end chain of the earlier oracles)."""
    import numpy as np
    from scipy.interpolate import CubicSpline

    if not (pulling_speed_nm_s > 0.0 and kuhn_length_nm > 0.0 and temperature_k > 0.0 and attempt_frequency_hz > 0.0):
        raise ValueError("speed, Kuhn length, temperature and attempt frequency must be positive")
    y_hold = hold_extension_nm / kuhn_length_nm
    rate = pulling_speed_nm_s / (attempt_frequency_hz * kuhn_length_nm)
    estimate = _oracle_infer_segment_count_and_bond_energy(
        y_hold, attempt_frequency_hz * intact_dwell_s, attempt_frequency_hz * broken_dwell_s, n_min, n_max)
    n_segments, depth = int(round(estimate[0])), float(estimate[1])

    nodes, scission, healing, tension = [], [], [], []
    node = 0
    while True:
        y = 0.5 * node
        points = _oracle_locate_free_energy_extrema(y, n_segments, depth)
        if np.isnan(points[0]):
            break
        energy = _oracle_compute_link_free_energy(points, y, n_segments, depth)
        nodes.append(y)
        scission.append(energy[1] - energy[0])
        healing.append(energy[1] - energy[2])
        tension.append(_oracle_compute_intact_chain_tension(y, points[0], n_segments))
        node += 1

    nodes = np.array(nodes)
    grid = 0.001 * np.arange(int(round(nodes[-1] / 0.001)) + 1)
    scission_f = CubicSpline(nodes, scission)(grid)
    healing_f = CubicSpline(nodes, healing)(grid)
    tension_f = CubicSpline(nodes, tension)(grid)
    intact = _oracle_integrate_intact_probability(grid, scission_f, healing_f, n_segments, rate)
    statistics = _oracle_compute_final_scission_statistics(grid, intact, scission_f, healing_f, tension_f, n_segments, rate)
    force_unit_pn = 1.380649e-23 * temperature_k / (kuhn_length_nm * 1e-9) * 1e12
    return float(statistics[0] * force_unit_pn)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    setup = (
        "def _sig(value, digits):\n"
        "    return float('%.*e' % (digits - 1, float(value)))\n"
    )
    status = (
        "def run_model():\n"
        "    try:\n"
        "        predict_mean_final_rupture_force(11.6, 0.0577, 0.05223, 0.0, 0.4, 300.0, 1e13, 45, 60)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "def run_oracle():\n"
        "    try:\n"
        "        _oracle_predict_mean_final_rupture_force(11.6, 0.0577, 0.05223, 0.0, 0.4, 300.0, 1e13, 45, 60)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": setup,
            "call": "_sig(predict_mean_final_rupture_force(11.60, 0.05770, 0.05223, 1.000, 0.400, 300.0, 1.0e13, 48, 54), 7)",
            "gold_call": "_sig(_oracle_predict_mean_final_rupture_force(11.60, 0.05770, 0.05223, 1.000, 0.400, 300.0, 1.0e13, 48, 54), 7)",
        },
        {
            "setup": setup,
            "call": "_sig(predict_mean_final_rupture_force(9.0, 3.735, 0.7546, 5.0, 0.5, 310.0, 1.0e9, 23, 29), 7)",
            "gold_call": "_sig(_oracle_predict_mean_final_rupture_force(9.0, 3.735, 0.7546, 5.0, 0.5, 310.0, 1.0e9, 23, 29), 7)",
        },
        {
            "setup": setup,
            "call": "_sig(predict_mean_final_rupture_force(8.1, 2585.0, 5994.0, 20.0, 0.3, 290.0, 1.0e12, 33, 39), 7)",
            "gold_call": "_sig(_oracle_predict_mean_final_rupture_force(8.1, 2585.0, 5994.0, 20.0, 0.3, 290.0, 1.0e12, 33, 39), 7)",
        },
        {
            "setup": status,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
