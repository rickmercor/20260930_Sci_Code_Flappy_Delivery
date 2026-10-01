"""
Report where site coherence stops being able to change the measured acceptor population of an inhomogeneous donor-acceptor ensemble by a set amount after the pulse overlap, together with the diagnostics at that point (orchestrator).

The pipeline expands both Drude-Lorentz bath correlation functions into exponentials, propagates the acceptor
population readout of each dimer backwards through the hierarchical equations of motion with recombination and
trapping, averages the readout over the Gaussian distribution of energy gaps, and evaluates the largest possible
coherence-induced change of the averaged acceptor population at every delay. Its maximum over the delay window
decreases as the reorganization energy of the baths grows, and the reorganization energy at which it equals the
threshold is the critical value E_R*.

At E_R* and at the delay of the maximum, the report adds the relative phase of the most favorable donor-acceptor
superposition, the averaged acceptor populations after excitation of the donor alone or of the acceptor alone, the
unpaired benchmark that compares the best preparation with the best coherence-free preparation, and, at the early
maximum, the instantaneous rate at which the coherence-induced change can vary. The integral of that rate from the
early maximum to the window maximum bounds how much the coherence impact can change between the two delays. Three reference values
make the role of each ingredient visible: the window maximum at a smaller reference reorganization energy, the window
maximum at E_R* for a sample without static disorder, and the largest coherence impact at E_R* before the window
opens.

Returns
-------
numpy.ndarray of shape (12,): [E_R*, t_star, phi, P_A_donor, P_A_acceptor, Q_reference, Q_homogeneous, C_early, t_early, Pi, Gamma, V]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coherence_relevance_report(dimer: "np.ndarray", site_sigma_cm: float, cutoff_cm: float, temperature_K: float,
                               n_matsubara: int, depth: int, n_nodes: int, window_ps: "np.ndarray",
                               target: float, reorg_bracket: "np.ndarray", reorg_reference: float) -> "np.ndarray":
    '''Critical reorganization energy of the ensemble coherence impact and the diagnostics at that point.

    Parameters
    ----------
    dimer : np.ndarray
        Shape (4,), [J, mean_gap, gamma_rec, kappa], as in ensemble_readout_operator.
    site_sigma_cm : float
        Standard deviation of each site energy (cm^-1), as in ensemble_readout_operator.
    cutoff_cm : float
        Drude-Lorentz cutoff gamma_c (cm^-1) of both baths.
    temperature_K : float
        Bath temperature (K).
    n_matsubara : int
        Matsubara terms per bath, as in heom_readout_operator.
    depth : int
        Hierarchy truncation, as in heom_readout_operator.
    n_nodes : int
        Gauss-Hermite nodes of the disorder average, as in ensemble_readout_operator.
    window_ps : np.ndarray
        Shape (2,), delay window [t_a, t_b] in ps with 0 < t_a < t_b, as in post_overlap_supremum.
    target : float
        Threshold of the maximum coherence impact, as in critical_reorganization_energy.
    reorg_bracket : np.ndarray
        Shape (2,), bracket of reorganization energies (cm^-1), as in critical_reorganization_energy.
    reorg_reference : float
        Reference reorganization energy (cm^-1, >= 0) at which the window maximum is also reported.

    Returns
    -------
    report : np.ndarray
        Shape (12,), [E_R*, t_star, phi, P_A_donor, P_A_acceptor, Q_reference, Q_homogeneous, C_early, t_early, Pi,
        Gamma, V]: E_R* (cm^-1) from critical_reorganization_energy; t_star (ps) the delay of the window maximum at
        E_R* from post_overlap_supremum; phi and Pi from coherence_diagnostics applied to the ensemble acceptor
        readout at E_R* and t_star; P_A_donor = M_DD and P_A_acceptor = M_AA of that readout, the ensemble-averaged
        acceptor populations at t_star when every molecule starts on the donor or on the acceptor; Q_reference the
        window maximum of the ensemble at reorg_reference; Q_homogeneous the window maximum at E_R* with
        site_sigma_cm = 0 and a single node; C_early and t_early (ps) the maximum of the same coherence impact at E_R*
        over the delays 0 <= t <= t_a before the window, and its delay, from post_overlap_supremum; Gamma (ps^-1) from
        coherence_diagnostics applied to the ensemble acceptor readout at E_R* and t_early; V the integral of the same
        rate Gamma(t) from t_early to t_star, evaluated with the composite Simpson rule on a uniform grid with an even
        number of intervals no wider than 2e-4 ps.

    Raises
    ------
    ValueError
        In the cases listed for critical_reorganization_energy and post_overlap_supremum, or if reorg_reference is
        negative or t_a is not positive.
    '''
    return report

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_coherence_relevance_report(dimer: "np.ndarray", site_sigma_cm: float, cutoff_cm: float,
                                       temperature_K: float, n_matsubara: int, depth: int, n_nodes: int,
                                       window_ps: "np.ndarray", target: float, reorg_bracket: "np.ndarray",
                                       reorg_reference: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    if reorg_reference < 0:
        raise ValueError("reorg_reference must be non-negative")
    t_a = float(np.asarray(window_ps, dtype=float).reshape(2)[0])
    if t_a <= 0:
        raise ValueError("the window must start after zero delay")
    reorg_star = _oracle_critical_reorganization_energy(dimer, site_sigma_cm, cutoff_cm, temperature_K, n_matsubara,
                                                        depth, n_nodes, window_ps, target, reorg_bracket)
    bath_star = np.array([reorg_star, cutoff_cm, temperature_K])
    t_star = _oracle_post_overlap_supremum(dimer, site_sigma_cm, bath_star, n_matsubara, depth, n_nodes,
                                           window_ps)[1]
    rows = _oracle_ensemble_readout_operator(dimer, site_sigma_cm, bath_star, n_matsubara, depth, n_nodes, 1,
                                             np.array([t_star]))
    diag = _oracle_coherence_diagnostics(rows)[0]
    q_reference = _oracle_post_overlap_supremum(dimer, site_sigma_cm, np.array([reorg_reference, cutoff_cm,
                                                temperature_K]), n_matsubara, depth, n_nodes, window_ps)[0]
    q_homogeneous = _oracle_post_overlap_supremum(dimer, 0.0, bath_star, n_matsubara, depth, 1, window_ps)[0]
    early = _oracle_post_overlap_supremum(dimer, site_sigma_cm, bath_star, n_matsubara, depth, n_nodes,
                                          np.array([0.0, t_a]))
    n_int = int(np.ceil((t_star - early[1]) / 2e-4 - 1e-9))
    n_int += n_int % 2
    grid = np.linspace(early[1], t_star, n_int + 1)
    rate = _oracle_coherence_diagnostics(_oracle_ensemble_readout_operator(
        dimer, site_sigma_cm, bath_star, n_matsubara, depth, n_nodes, 1, grid))[:, 3]
    h = (t_star - early[1]) / n_int
    variation = h / 3.0 * (rate[0] + rate[-1] + 4.0 * rate[1:-1:2].sum() + 2.0 * rate[2:-1:2].sum())
    rate_early = rate[0]
    return np.array([reorg_star, t_star, diag[2], rows[0, 0], rows[0, 1], q_reference, q_homogeneous,
                     early[0], early[1], diag[1], rate_early, variation])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: reference ensemble on a shallow hierarchy with few nodes and a short window ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "window = np.array([0.15, 0.3])\n"
                     "bracket = np.array([10.0, 60.0])\n",
            "call": "coherence_relevance_report(dimer.copy(), 70.0, 60.0, 295.0, 1, 2, 5, window.copy(), 0.1, bracket.copy(), 20.0)",
            "gold_call": "_oracle_coherence_relevance_report(dimer, 70.0, 60.0, 295.0, 1, 2, 5, window, 0.1, bracket, 20.0)",
            "tol": 1e-6,
        },
        # --- Boundary: resonant pair without trapping, Drude term only, reference at zero reorganization energy ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([90.0, 0.0, 0.05, 0.0])\n"
                     "window = np.array([0.12, 0.3])\n"
                     "bracket = np.array([5.0, 80.0])\n",
            "call": "coherence_relevance_report(dimer.copy(), 40.0, 90.0, 300.0, 0, 2, 4, window.copy(), 0.15, bracket.copy(), 0.0)",
            "gold_call": "_oracle_coherence_relevance_report(dimer, 40.0, 90.0, 300.0, 0, 2, 4, window, 0.15, bracket, 0.0)",
            "tol": 1e-6,
        },
        # --- Edge: acceptor above the donor on average, cold slow bath and a narrow ensemble ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([110.0, -180.0, 0.0, 0.8])\n"
                     "window = np.array([0.1, 0.25])\n"
                     "bracket = np.array([2.0, 60.0])\n",
            "call": "coherence_relevance_report(dimer.copy(), 20.0, 35.0, 180.0, 1, 2, 4, window.copy(), 0.26, bracket.copy(), 10.0)",
            "gold_call": "_oracle_coherence_relevance_report(dimer, 20.0, 35.0, 180.0, 1, 2, 4, window, 0.26, bracket, 10.0)",
            "tol": 1e-6,
        },
        # --- Invalid: a negative reference reorganization energy must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "def run(fn):\n"
                     "    try:\n"
                     "        fn(dimer, 70.0, 60.0, 295.0, 1, 2, 4, np.array([0.15, 0.3]), 0.1, np.array([10.0, 60.0]), -1.0)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n",
            "call": "run(coherence_relevance_report)",
            "gold_call": "run(_oracle_coherence_relevance_report)",
        },
    ]
