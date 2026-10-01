"""
Resolve the complete metabolic panel.

The final computation composes the reaction-weight, mass-balance, state, energy, physiological and robust-selection operations into the requested scalar.

Returns
-------
One finite float containing the selected robust score.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def resolve_metabolic_panel(data: dict | None = None) -> float:
    """
    data is None for the numerical fixture in the main problem, or a dictionary with
    exactly these keys: S, b, weights, factors, irreversible, standard_mean,
    standard_sd, log_bounds, contrasts, contrast_bounds, R, T, tau, epsilon, excess,
    target, normalization. Their shapes, units and valid domains are the corresponding
    inputs in steps 1-9. Compose all preceding public functions in order and use their
    returned values. Return the selected robust score as one float. Invalid inputs or no
    eligible candidate raise ValueError. The default raw-input constructor is supplied
    in this item; it contains no precomputed answers. Numerical acceptance uses absolute
    tolerance 2e-6. Unresolved numerical solver failures propagate as RuntimeError.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_resolve_metabolic_panel(data: dict | None = None) -> float:
    if data is None:
        data = _benchmark_inputs()
    if not isinstance(data, dict):
        raise ValueError("Input must be a data dictionary")
    required = {
        "S",
        "b",
        "weights",
        "factors",
        "irreversible",
        "standard_mean",
        "standard_sd",
        "log_bounds",
        "contrasts",
        "contrast_bounds",
        "R",
        "T",
        "tau",
        "epsilon",
        "excess",
        "target",
        "normalization",
    }
    if set(data) != required:
        raise ValueError("Input keys do not match the contract")
    weights = _oracle_perturbed_weights(data["weights"], data["factors"])
    balance = _oracle_balance_rows(data["S"], data["b"])
    nets = _oracle_infer_net_panel(balance, weights, data["irreversible"])
    pairs = _oracle_directional_panel(nets, weights)
    energies = _oracle_energy_panel(pairs, data["R"], data["T"])
    directions = _oracle_active_directions(nets, data["tau"])
    ranges = _oracle_physiological_ranges(
        data["S"],
        directions,
        data["standard_mean"],
        data["standard_sd"],
        data["log_bounds"],
        data["contrasts"],
        data["contrast_bounds"],
        data["R"],
        data["T"],
        data["epsilon"],
    )
    compatibility = _oracle_compatibility_panel(
        energies, directions, ranges, data["excess"]
    )
    selected = _oracle_select_robust_candidate(
        nets, compatibility, data["target"], data["normalization"]
    )
    return float(selected[2])


def _benchmark_inputs():
    import numpy as np

    edges = [
        (0, 1),
        (0, 2),
        (2, 1),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
        (4, 6),
        (5, 6),
        (6, 1),
        (5, 2),
        (0, 6),
    ]
    S = np.zeros((7, 14))
    for j, (left, right) in enumerate(edges):
        S[left, j] = -1.0
        S[right, j] = 1.0
    weights = np.array(
        [
            [
                4.8,
                13.1,
                2.4,
                9.7,
                3.9,
                12.6,
                2.1,
                4.3,
                7.2,
                16.5,
                5.4,
                8.1,
                3.6,
                1.1,
            ],
            [
                8.4,
                10.2,
                6.3,
                5.1,
                9.8,
                11.3,
                7.4,
                3.6,
                4.9,
                14.2,
                6.7,
                4.5,
                9.3,
                1.7,
            ],
            [
                12.2,
                7.6,
                3.1,
                13.4,
                6.8,
                8.9,
                4.2,
                7.1,
                2.7,
                18.4,
                3.9,
                6.8,
                5.1,
                2.4,
            ],
            [
                6.1,
                14.8,
                9.2,
                6.3,
                12.1,
                15.7,
                3.6,
                5.8,
                10.4,
                13.7,
                7.8,
                3.2,
                8.6,
                0.9,
            ],
            [
                10.7,
                9.3,
                4.6,
                11.9,
                5.2,
                10.8,
                8.1,
                6.4,
                3.8,
                15.9,
                4.7,
                7.3,
                2.9,
                1.4,
            ],
            [
                7.9,
                12.4,
                7.1,
                8.6,
                7.7,
                13.9,
                5.5,
                4.9,
                6.3,
                17.1,
                5.9,
                5.6,
                6.2,
                1.9,
            ],
        ]
    )
    factors = np.ones((6, 14))
    changes = [
        [(0, 0.22), (8, 1.8)],
        [(3, 0.28), (13, 1.5)],
        [(5, 0.55), (10, 1.4)],
        [(6, 0.18), (9, 0.75)],
        [(1, 0.35), (3, 1.6)],
    ]
    for row, entries in enumerate(changes, 1):
        for column, value in entries:
            factors[row, column] = value
    H = np.zeros((3, 7))
    H[0, 1] = 1.0
    H[0, 2] = -1.0
    H[1, 6] = 1.0
    H[1, 4] = -1.0
    H[2, [0, 3]] = 1.0
    H[2, [1, 4]] = -1.0
    hb = np.tile([[-6.0, 6.0], [-6.0, 6.0], [-12.0, 12.0]], (6, 1, 1))
    hb[0, 0] = [0.18, 0.42]
    hb[3, 1] = [-0.58, -0.22]
    hb[4, 2] = [2.25, 2.9]
    return dict(
        S=S,
        b=np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 2.0, 8.0]),
        weights=weights,
        factors=factors,
        irreversible=np.array([2, 11, 12]),
        standard_mean=np.zeros(14),
        standard_sd=np.full(14, 0.04),
        log_bounds=np.tile([-10.5, -4.5], (6, 7, 1)),
        contrasts=H,
        contrast_bounds=hb,
        R=0.00831446261815324,
        T=298.15,
        tau=1e-07,
        epsilon=0.001,
        excess=0.1,
        target=9,
        normalization=8.0,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "resolve_metabolic_panel()",
            "gold_call": "_oracle_resolve_metabolic_panel()",
            "tol": 2e-06,
        },
        {
            "setup": "import numpy as np\n"
            "d = dict(S=np.array([[-1.0], [1.0]]), b=np.array([-2.0, 2.0]), "
            "weights=np.array([[2.0], [5.0]]), factors=np.array([[1.0], [0.5]]), "
            "irreversible=np.array([0], dtype=int), standard_mean=np.zeros(1), "
            "standard_sd=np.zeros(1), log_bounds=np.tile([-4.0, 0.0], (2, 2, 1)), "
            "contrasts=np.empty((0, 2)), contrast_bounds=np.empty((2, 0, 2)), R=1.0, "
            "T=1.0, tau=1e-07, epsilon=0.01, excess=0.3, target=0, normalization=4.0)",
            "call": "resolve_metabolic_panel(d)",
            "gold_call": "_oracle_resolve_metabolic_panel(d)",
            "tol": 2e-06,
        },
        {
            "setup": "import numpy as np\n"
            "d = dict(S=np.array([[-1.0], [1.0]]), b=np.array([-2.0, 2.0]), "
            "weights=np.array([[2.0], [5.0]]), factors=np.array([[1.0], [0.5]]), "
            "irreversible=np.array([0], dtype=int), standard_mean=np.zeros(1), "
            "standard_sd=np.zeros(1), log_bounds=np.tile([-4.0, 0.0], (2, 2, 1)), "
            "contrasts=np.empty((0, 2)), contrast_bounds=np.empty((2, 0, 2)), R=1.0, "
            "T=1.0, tau=1e-07, epsilon=0.01, excess=0.3, target=0, "
            "normalization=4.0)\n"
            "d['b'][:] = 0.0",
            "call": "resolve_metabolic_panel(d)",
            "gold_call": "_oracle_resolve_metabolic_panel(d)",
            "tol": 2e-06,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "def _error_check(fn, args):\n"
            "    try:\n"
            "        fn(*args)\n"
            "    except ValueError:\n"
            "        return 1.0\n"
            "    return 0.0",
            "call": "_error_check(resolve_metabolic_panel, ({},))",
            "gold_call": "_error_check(_oracle_resolve_metabolic_panel, ({},))",
            "tol": 0.0,
        },
    ]
