"""
Compose every earlier step to obtain one species concentration at the positive steady state of a stoichiometric compatibility class.

Translating only the blocks that are not weakly reversible and deficiency zero, merging them into one generalized network and parametrizing that network turns the steady-state problem into a small system fixed by the conserved totals.

Returns
-------
float: steady-state concentration of the target species.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def run_equilibrium_pipeline(
    source_complexes: "np.ndarray | None" = None,
    product_complexes: "np.ndarray | None" = None,
    rate_constants: "np.ndarray | None" = None,
    conservation_matrix: "np.ndarray | None" = None,
    totals: "np.ndarray | None" = None,
    free_species: "np.ndarray | None" = None,
    target_species: "int | None" = None,
    rate_bracket: tuple = (1e-8, 1e8),
) -> float:
    """Return one species concentration at the positive steady state of a class.

    The mass-action network is decomposed into its finest independent
    subnetworks. Every subnetwork that is not both weakly reversible and of
    deficiency zero is translated through its elementary flux modes, a
    compatible reaction-to-reaction graph and the resulting translation
    complexes, while the other subnetworks keep a zero translation. The
    translated subnetworks are merged back into one network, which must be
    weakly reversible with deficiency zero. Its generalized network is built,
    the phantom rate is fixed when the kinetic deficiency requires it, tree
    constants are evaluated with the reaction rate constants (and the phantom
    rate), the complex-balanced equilibria are parametrized by
    ``free_species``, and the conserved totals are imposed.

    Any argument left as ``None`` takes its value from the EnvZ-OmpR
    benchmark of the problem statement: species ordered X, XD, XT, Xp, Y,
    Yp, XpY, XDYp, XTYp; the fourteen reactions and rate constants in the
    order listed there; conservation rows for total EnvZ and total OmpR
    with totals 3.31 and 4.21; free species XpY and Y (indices 6 and 4);
    and target species XpY (index 6).

    Parameters
    ----------
    source_complexes, product_complexes : np.ndarray
        Integer arrays with shape ``(r, m)``.
    rate_constants : np.ndarray
        Positive mass-action rate constant of each reaction, shape ``(r,)``.
    conservation_matrix : np.ndarray
        Nonnegative conservation weights, shape ``(d, m)``.
    totals : np.ndarray
        Positive conserved totals, shape ``(d,)``.
    free_species : np.ndarray
        Species indices used as free coordinates, shape ``(d,)``.
    target_species : int
        Index of the species whose steady-state concentration is returned.
    rate_bracket : tuple
        Search bracket for the phantom rate.

    Returns
    -------
    float
        Steady-state concentration of ``target_species``.

    Raises
    ------
    ValueError
        If any stage rejects its input, if the merged translated network is
        not weakly reversible with deficiency zero, if ``rate_constants`` does
        not hold ``r`` finite positive values, or if ``target_species`` is not
        a valid species index.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_run_equilibrium_pipeline(
    source_complexes: "np.ndarray | None" = None,
    product_complexes: "np.ndarray | None" = None,
    rate_constants: "np.ndarray | None" = None,
    conservation_matrix: "np.ndarray | None" = None,
    totals: "np.ndarray | None" = None,
    free_species: "np.ndarray | None" = None,
    target_species: "int | None" = None,
    rate_bracket: tuple = (1e-8, 1e8),
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    reactions = [([1], [0]), ([0], [1]), ([0], [2]), ([2], [0]), ([2], [3]),
                 ([3, 4], [6]), ([6], [3, 4]), ([6], [0, 5]), ([1, 5], [7]),
                 ([7], [1, 5]), ([7], [1, 4]), ([2, 5], [8]), ([8], [2, 5]),
                 ([8], [2, 4])]
    bench_source = np.zeros((14, 9), dtype=int)
    bench_product = np.zeros((14, 9), dtype=int)
    for row, (left, right) in enumerate(reactions):
        for species in left:
            bench_source[row, species] += 1
        for species in right:
            bench_product[row, species] += 1
    if source_complexes is None:
        source_complexes = bench_source
    if product_complexes is None:
        product_complexes = bench_product
    if rate_constants is None:
        rate_constants = [1.93, 1.72, 1.90, 0.19, 0.51, 3.39, 2.90,
                          1.45, 0.28, 4.31, 0.35, 0.60, 2.84, 1.66]
    if conservation_matrix is None:
        conservation_matrix = [[1, 1, 1, 1, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1]]
    if totals is None:
        totals = [3.31, 4.21]
    if free_species is None:
        free_species = [6, 4]
    if target_species is None:
        target_species = 6
    source = np.asarray(source_complexes)
    product = np.asarray(product_complexes)
    rates = np.asarray(rate_constants, dtype=float)
    labels = _oracle_decompose_independent_subnetworks(source, product)
    if rates.shape != (source.shape[0],) or not np.all(np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("rate_constants must hold one finite positive value per reaction")
    def is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    if not (is_integer(target_species) and 0 <= int(target_species) < source.shape[1]):
        raise ValueError("target_species must be an integer index of an existing species")

    translation = np.zeros(source.shape, dtype=int)
    for block in np.unique(labels):
        members = np.flatnonzero(labels == block)
        block_source, block_product = source[members], product[members]
        summary = _oracle_compute_network_deficiency(
            block_source, block_product, np.zeros(block_source.shape, dtype=int)
        )
        if summary[3] == 0 and summary[4] == 1:
            continue
        modes = _oracle_compute_elementary_flux_modes(block_source, block_product)
        graph = _oracle_build_reaction_graph(block_source, block_product, modes)
        translation[members] = _oracle_compute_translation_complexes(
            block_source, block_product, graph
        )

    merged = _oracle_compute_network_deficiency(source, product, translation)
    if merged[3] != 0 or merged[4] != 1:
        raise ValueError("the merged translated network is not weakly reversible with deficiency zero")
    stoichiometric, kinetic, edges = _oracle_build_generalized_network(source, product, translation)
    phantom = _oracle_solve_phantom_rate(
        kinetic, edges, rates, _oracle_compute_tree_constants, rate_bracket
    )
    edge_rates = np.array(
        [rates[k] if k >= 0 else phantom[0] for k in edges[:, 2]], dtype=float
    )
    tree = _oracle_compute_tree_constants(stoichiometric.shape[0], edges, edge_rates)
    offset, exponents = _oracle_parametrize_steady_states(
        kinetic, edges, tree, np.asarray(free_species)
    )
    state = _oracle_solve_class_steady_state(
        offset, exponents, np.asarray(conservation_matrix, dtype=float), np.asarray(totals, dtype=float)
    )
    result = float(state[int(target_species)])
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("the steady-state concentration must be finite and positive")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only tests with immutable end-to-end targets."""
    builder = (
        "import numpy as np\n"
        "def _net(m, rx):\n"
        "    S = np.zeros((len(rx), m), dtype=int)\n"
        "    P = np.zeros((len(rx), m), dtype=int)\n"
        "    for r, (a, b) in enumerate(rx):\n"
        "        for i in a: S[r, i] += 1\n"
        "        for i in b: P[r, i] += 1\n"
        "    return S, P\n"
    )
    envz = (
        "S, P = _net(9, [([1], [0]), ([0], [1]), ([0], [2]), ([2], [0]), ([2], [3]),"
        " ([3, 4], [6]), ([6], [3, 4]), ([6], [0, 5]), ([1, 5], [7]), ([7], [1, 5]),"
        " ([7], [1, 4]), ([2, 5], [8]), ([8], [2, 5]), ([8], [2, 4])])\n"
        "W = np.array([[1, 1, 1, 1, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=float)\n"
    )
    dual = (
        "S, P = _net(8, [([0], [1]), ([1, 2], [0, 3]), ([4, 3], [6]), ([6], [4, 3]),"
        " ([6], [4, 2]), ([5, 3], [7]), ([7], [5, 3]), ([7], [5, 2]),"
        " ([4], [5]), ([5], [4])])\n"
        "W = np.array([[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]], dtype=float)\n"
        "k = np.array([1.2, 2.3, 1.7, 0.6, 0.9, 2.6, 0.4, 1.1, 0.8, 1.5])\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "run_equilibrium_pipeline()",
            "gold_call": "_oracle_run_equilibrium_pipeline()",
        },
        {
            "setup": builder + envz + (
                "k = np.array([1.93, 1.72, 1.90, 0.19, 0.51, 3.39, 2.90, 1.45, 0.28, 4.31, 0.35, 0.60, 2.84, 1.66])\n"
                "perm = np.array([5, 0, 8, 2, 12, 1, 9, 3, 6, 10, 4, 13, 7, 11])\n"
            ),
            "call": "run_equilibrium_pipeline(S[perm], P[perm], k[perm], W, np.array([3.31, 4.21]), np.array([6, 4]), 6)",
            "gold_call": "0.4183519652823077",
        },
        {
            "setup": builder + envz + "k = np.array([0.83, 1.37, 2.46, 0.52, 1.18, 3.05, 0.74, 1.61, 2.27, 0.96, 0.43, 1.92, 0.68, 0.37])\n",
            "call": "run_equilibrium_pipeline(S, P, k, W, np.array([1.75, 4.30]), np.array([6, 4]), 6)",
            "gold_call": "0.19045388784938536",
        },
        {
            "setup": builder + envz + "k = np.array([2.2, 0.45, 1.1, 0.9, 0.35, 1.8, 2.4, 0.95, 0.6, 1.3, 0.75, 0.4, 1.6, 2.1])\n",
            "call": "run_equilibrium_pipeline(S, P, k, W, np.array([2.4, 5.6]), np.array([0, 4]), 4)",
            "gold_call": "3.889485877470316",
        },
        {
            "setup": builder + envz + "k = np.array([2.2, 0.45, 1.1, 0.9, 0.35, 1.8, 2.4, 0.95, 0.6, 1.3, 0.75, 0.4, 1.6, 2.1])\n",
            "call": "run_equilibrium_pipeline(S, P, k, W, np.array([2.4, 5.6]), np.array([6, 4]), 3)",
            "gold_call": "0.13494802127021072",
        },
        {
            "setup": builder + "S, P = _net(3, [([0, 1], [2, 1]), ([0], [1]), ([1], [0]), ([2], []), ([], [0])])\nk = np.array([1.3, 0.7, 2.1, 0.9, 1.6])\n",
            "call": "run_equilibrium_pipeline(S, P, k, np.zeros((0, 3)), np.zeros(0), np.array([], dtype=int), 0)",
            "gold_call": "float(np.sqrt(2.1 * 1.6 / (1.3 * 0.7)))",
        },
        {
            "setup": builder + dual,
            "call": "run_equilibrium_pipeline(S, P, k, W, np.array([0.9, 2.4, 1.3]), np.array([1, 3, 5]), 6)",
            "gold_call": "0.3776550185861278",
        },
    ]
