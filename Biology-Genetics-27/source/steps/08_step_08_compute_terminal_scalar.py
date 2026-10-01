"""
Compose the seven preceding numerical functions across the source-defined state transitions and return the terminal scalar.

The target trait occupies index 0.



For each current trait state:



1. Call compute_state_scalar and compute_replicate_vector.

2. Compare the first element of the replicate vector with 0.5 before applying any scalar adjustment.

3. If that element is not greater than 0.5, call construct_replicate_state, generate_perturbed_states, construct_state_components, and solve_state_scale.

4. The current state is terminal only when those calls succeed and the solved scale is at least 0.5. Return the adjusted scalar from that terminal state.

5. Otherwise, apply the thresholds [0.5, 0.4, 0.3, 0.2, 0.1] in order through select_state_mask. At the first threshold that removes at least one trait, apply the same mask to the point matrix and every aligned packed replicate, then restart all current-state calculations.

6. Continue with the next unused threshold if the recomputed state remains nonterminal. Raise ValueError if no remaining threshold can change a nonterminal state.



For every new trait state, restart the stochastic calculation with the supplied seed.

Returns
-------
terminal_scalar : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_terminal_scalar(
    rg_point: np.ndarray,
    rg_replicates: np.ndarray,
    seed: int = 123,
    sample_count: int = 200,
    proposal_limit: int = 200,
    tolerance: float = 1e-10,
    max_iterations: int = 80,
) -> float:
    """Return the terminal scalar from the composed numerical analysis.

    Parameters
    ----------
    rg_point : np.ndarray
        Full-sample correlation matrix with the target trait first.
    rg_replicates : np.ndarray
        Aligned packed block-deletion correlation replicates.
    seed : int
        Seed restarted for each current trait state.
    sample_count : int
        Number of retained perturbed states per trait state.
    proposal_limit : int
        Maximum proposals allowed for one retained state.
    tolerance : float
        Positive absolute scalar-search tolerance.
    max_iterations : int
        Positive maximum number of midpoint iterations.

    Returns
    -------
    float
        Terminal scalar from the first source-compliant state.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_terminal_scalar(
    rg_point: np.ndarray,
    rg_replicates: np.ndarray,
    seed: int = 123,
    sample_count: int = 200,
    proposal_limit: int = 200,
    tolerance: float = 1e-10,
    max_iterations: int = 80,
) -> float:
    current_point = np.asarray(
        rg_point,
        dtype=float,
    ).copy()

    current_replicates = np.asarray(
        rg_replicates,
        dtype=float,
    ).copy()

    if (
        current_point.ndim != 2
        or current_point.shape[0]
        != current_point.shape[1]
        or current_point.shape[0] < 2
    ):
        raise ValueError(
            "rg_point must be square with at least two traits"
        )

    expected_width = (
        current_point.shape[0]
        * (current_point.shape[0] - 1)
        // 2
    )

    if (
        current_replicates.ndim != 2
        or current_replicates.shape[0] < 2
        or current_replicates.shape[1]
        != expected_width
    ):
        raise ValueError(
            "rg_replicates has incompatible shape"
        )

    thresholds = np.array(
        [0.5, 0.4, 0.3, 0.2, 0.1],
        dtype=float,
    )

    threshold_index = 0

    def subset_packed_replicates(
        packed_replicates: np.ndarray,
        keep_mask: np.ndarray,
    ) -> np.ndarray:
        n_traits = int(keep_mask.size)

        packed_pairs = [
            (row, column)
            for column in range(n_traits - 1)
            for row in range(
                column + 1,
                n_traits,
            )
        ]

        matrices = np.repeat(
            np.eye(
                n_traits,
                dtype=float,
            )[None, :, :],
            packed_replicates.shape[0],
            axis=0,
        )

        for packed_index, (
            row,
            column,
        ) in enumerate(packed_pairs):
            matrices[
                :,
                row,
                column,
            ] = packed_replicates[
                :,
                packed_index,
            ]

            matrices[
                :,
                column,
                row,
            ] = packed_replicates[
                :,
                packed_index,
            ]

        keep_indices = np.flatnonzero(
            keep_mask > 0.5
        )

        if keep_indices.size < 2:
            raise ValueError(
                "a state transition must retain the target "
                "and at least one non-target trait"
            )

        matrices = matrices[
            :,
            keep_indices,
        ][
            :,
            :,
            keep_indices,
        ]

        retained_traits = int(
            keep_indices.size
        )

        retained_pairs = [
            (row, column)
            for column in range(
                retained_traits - 1
            )
            for row in range(
                column + 1,
                retained_traits,
            )
        ]

        return np.stack(
            [
                matrices[
                    :,
                    row,
                    column,
                ]
                for row, column
                in retained_pairs
            ],
            axis=1,
        )

    while True:
        initial_scalar = _oracle_compute_state_scalar(
            current_point,
            0,
        )

        replicate_vector = _oracle_compute_replicate_vector(
            current_point,
            current_replicates,
            0,
        )

        initial_scale = float(
            replicate_vector[0]
        )

        target_scores = np.asarray(
            replicate_vector[1:],
            dtype=float,
        )

        solved_state = None
        numerical_failure = False

        if initial_scale <= 0.5:
            try:
                replicate_state = (
                    _oracle_construct_replicate_state(
                        current_replicates
                    )
                )

                perturbed_states = (
                    _oracle_generate_perturbed_states(
                        current_point,
                        replicate_state,
                        int(sample_count),
                        int(proposal_limit),
                        int(seed),
                    )
                )

                state_components = (
                    _oracle_construct_state_components(
                        current_point,
                        perturbed_states,
                        0,
                    )
                )

                solved_state = _oracle_solve_state_scale(
                    initial_scalar,
                    initial_scale,
                    state_components,
                    float(tolerance),
                    int(max_iterations),
                )

            except ValueError:
                numerical_failure = True

        nonterminal = (
            initial_scale > 0.5
            or numerical_failure
            or solved_state is None
            or float(solved_state[0]) < 0.5
        )

        if not nonterminal:
            terminal_scalar = float(
                solved_state[1]
            )

            if not np.isfinite(
                terminal_scalar
            ):
                raise ValueError(
                    "terminal scalar must be finite"
                )

            return terminal_scalar

        if current_point.shape[0] < 3:
            raise ValueError(
                "nonterminal state has no non-target pair "
                "available for transition"
            )

        transitioned = False

        while threshold_index < thresholds.size:
            keep_mask = _oracle_select_state_mask(
                current_point,
                0,
                target_scores,
                float(
                    thresholds[
                        threshold_index
                    ]
                ),
            )

            threshold_index += 1

            if (
                int(np.sum(keep_mask))
                < current_point.shape[0]
            ):
                retained = (
                    keep_mask > 0.5
                )

                current_point = current_point[
                    np.ix_(
                        retained,
                        retained,
                    )
                ]

                current_replicates = (
                    subset_packed_replicates(
                        current_replicates,
                        keep_mask,
                    )
                )

                transitioned = True
                break

        if not transitioned:
            raise ValueError(
                "nonterminal state cannot be changed "
                "by the remaining thresholds"
            )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for compute_terminal_scalar."""
    return [
        {
            "setup": """import numpy as np

rg_point = np.array([
    [1.00, 0.30, 0.20],
    [0.30, 1.00, 0.10],
    [0.20, 0.10, 1.00],
], dtype=float)

rg_replicates = np.array([
    [0.29, 0.19, 0.09],
    [0.31, 0.20, 0.11],
    [0.30, 0.21, 0.10],
    [0.28, 0.20, 0.12],
    [0.32, 0.18, 0.08],
    [0.30, 0.22, 0.10],
], dtype=float)

seed = 123
sample_count = 50
proposal_limit = 200
tolerance = 1e-10
max_iterations = 80
""",
            "call": (
                "compute_terminal_scalar("
                "rg_point, rg_replicates, seed, "
                "sample_count, proposal_limit, "
                "tolerance, max_iterations)"
            ),
            "gold_call": (
                "_oracle_compute_terminal_scalar("
                "rg_point, rg_replicates, seed, "
                "sample_count, proposal_limit, "
                "tolerance, max_iterations)"
            ),
        },
        {
            "setup": """import numpy as np

rg_point = np.array([
    [1.0000000000, 0.1700000000, 0.6000000000, 0.2200000000, 0.1600000000],
    [0.1700000000, 1.0000000000, 0.7200000000, 0.1800000000, 0.0800000000],
    [0.6000000000, 0.7200000000, 1.0000000000, 0.1500000000, 0.1000000000],
    [0.2200000000, 0.1800000000, 0.1500000000, 1.0000000000, 0.2500000000],
    [0.1600000000, 0.0800000000, 0.1000000000, 0.2500000000, 1.0000000000],
], dtype=float)

rg_replicates = np.array([
    [0.1653801057, 0.6494734492, 0.2571888598, 0.1644683322, 0.8060784974, 0.2304952515, 0.0344004736, 0.4356793215, 0.3265773590, 0.3034208421],
    [0.1672879433, 0.5791059667, 0.2432566771, 0.1640343348, 0.6811308108, 0.2115426641, 0.0519380860, 0.0245900964, 0.0000964831, 0.2836919628],
    [0.1670666226, 0.5895607522, 0.2333512134, 0.1614766631, 0.7018187176, 0.2010257711, 0.0617102569, 0.0908502596, 0.0534203146, 0.2721661840],
    [0.1715496642, 0.6086931318, 0.2118597549, 0.1587537426, 0.7369085983, 0.1676020853, 0.0910708457, 0.2039872800, 0.1433090936, 0.2380373411],
    [0.1637714331, 0.5833000847, 0.2715484917, 0.1656956621, 0.6907875210, 0.2501979939, 0.0164538415, 0.0541020582, 0.0232001678, 0.3244129754],
    [0.1583691065, 0.5933096389, 0.3130540162, 0.1719352627, 0.7082044063, 0.3064936195, -0.0345575713, 0.1119648544, 0.0694800545, 0.3852178228],
    [0.1710424286, 0.6466052134, 0.2102284988, 0.1584449872, 0.8029831711, 0.1665463101, 0.0923382506, 0.4264171240, 0.3187512497, 0.2361653245],
    [0.1683369970, 0.6306383339, 0.2267835960, 0.1608904997, 0.7728095984, 0.1891453234, 0.0720939331, 0.3256636676, 0.2397119585, 0.2599425244],
    [0.1727666282, 0.6129445578, 0.1994666404, 0.1573902104, 0.7434500350, 0.1524738717, 0.1057898817, 0.2271320628, 0.1607540381, 0.2194992288],
    [0.1776319912, 0.5942100264, 0.1564840658, 0.1522195038, 0.7084103917, 0.0932806071, 0.1581871567, 0.1102150592, 0.0689512945, 0.1582614322],
    [0.1682373991, 0.5966043662, 0.2388530913, 0.1617093726, 0.7138052414, 0.2053215882, 0.0564289924, 0.1307740052, 0.0849165736, 0.2766348908],
    [0.1733851495, 0.6415754702, 0.1988916862, 0.1568362719, 0.7954352227, 0.1501962057, 0.1069953523, 0.3958289959, 0.2941834523, 0.2177034421],
    [0.1617223031, 0.5943708272, 0.2909395616, 0.1689665260, 0.7106849483, 0.2779444766, -0.0075831452, 0.1197574536, 0.0775998403, 0.3534267871],
    [0.1740016896, 0.6121727460, 0.1893330666, 0.1566875262, 0.7426305369, 0.1388469450, 0.1171631014, 0.2240687676, 0.1588877590, 0.2054961561],
    [0.1711828211, 0.6126706284, 0.2107616362, 0.1588156242, 0.7422015537, 0.1667415034, 0.0911014520, 0.2260740807, 0.1601476148, 0.2366191799],
    [0.1764410416, 0.5741806720, 0.1646741629, 0.1536439452, 0.6751741489, 0.1024865265, 0.1502491289, 0.0012292115, -0.0181174910, 0.1693744817],
    [0.1701795152, 0.6347922401, 0.2190148301, 0.1594967049, 0.7815100215, 0.1786821860, 0.0799185989, 0.3495185299, 0.2583129708, 0.2490980862],
    [0.1662501542, 0.5686709431, 0.2458764814, 0.1631812700, 0.6643637184, 0.2167678372, 0.0476229817, -0.0342204174, -0.0464634332, 0.2887129629],
    [0.1645009594, 0.5964633418, 0.2601253793, 0.1650257093, 0.7158196507, 0.2344382788, 0.0304103696, 0.1342164030, 0.0882355453, 0.3087530542],
    [0.1778943473, 0.5208420624, 0.1546886483, 0.1516716438, 0.5793270778, 0.0898601365, 0.1618698187, -0.3136498730, -0.2688470826, 0.1537847767],
    [0.1727038760, 0.6355973180, 0.2023314693, 0.1587400635, 0.7830590113, 0.1569485083, 0.1008222911, 0.3576005735, 0.2651608088, 0.2245922713],
    [0.1660097504, 0.6121266922, 0.2519527117, 0.1644198612, 0.7409470041, 0.2240012928, 0.0399315571, 0.2197123879, 0.1546833937, 0.2969623572],
    [0.1744915825, 0.5777708831, 0.1806941045, 0.1549768504, 0.6809542206, 0.1269394865, 0.1279859530, 0.0186061212, -0.0038354126, 0.1926467242],
    [0.1797964904, 0.5343206543, 0.1486413565, 0.1505194323, 0.6015058960, 0.0820215306, 0.1676583935, -0.2401180235, -0.2091165526, 0.1453791913],
], dtype=float)

seed = 123
sample_count = 200
proposal_limit = 200
tolerance = 1e-10
max_iterations = 80
""",
            "call": (
                "compute_terminal_scalar("
                "rg_point, rg_replicates, seed, "
                "sample_count, proposal_limit, "
                "tolerance, max_iterations)"
            ),
            "gold_call": (
                "_oracle_compute_terminal_scalar("
                "rg_point, rg_replicates, seed, "
                "sample_count, proposal_limit, "
                "tolerance, max_iterations)"
            ),
        },
        {
            "setup": """import numpy as np

rg_point = np.array([
    [1.00, 0.30, 0.20],
    [0.30, 1.00, 0.10],
    [0.20, 0.10, 1.00],
], dtype=float)

rg_replicates = np.array([
    [0.29, 0.19, 0.09],
    [0.31, 0.20, 0.11],
    [0.30, 0.21, 0.10],
    [0.28, 0.20, 0.12],
    [0.32, 0.18, 0.08],
    [0.30, 0.22, 0.10],
], dtype=float)

def run_model():
    first = compute_terminal_scalar(
        rg_point,
        rg_replicates,
        123,
        50,
        200,
        1e-10,
        80,
    )

    second = compute_terminal_scalar(
        rg_point,
        rg_replicates[::-1],
        123,
        50,
        200,
        1e-10,
        80,
    )

    return np.array(
        [
            first,
            second,
            first - second,
        ],
        dtype=float,
    )

def run_gold():
    first = _oracle_compute_terminal_scalar(
        rg_point,
        rg_replicates,
        123,
        50,
        200,
        1e-10,
        80,
    )

    second = _oracle_compute_terminal_scalar(
        rg_point,
        rg_replicates[::-1],
        123,
        50,
        200,
        1e-10,
        80,
    )

    return np.array(
        [
            first,
            second,
            first - second,
        ],
        dtype=float,
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

rg_point = np.eye(
    3,
    dtype=float,
)

rg_replicates = np.zeros(
    (6, 3),
    dtype=float,
)

def run_model():
    try:
        compute_terminal_scalar(
            rg_point,
            rg_replicates,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_terminal_scalar(
            rg_point,
            rg_replicates,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
