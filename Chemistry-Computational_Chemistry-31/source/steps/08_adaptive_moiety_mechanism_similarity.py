"""
Orchestrate adaptive moiety-rule search and archive re-ranking.

The complete benchmark combines adaptive-radius selection, parsimonious rule-set generation, prefix-feasible mechanism ordering, arrow-environment profile construction, maximum archive similarity, and stable re-ranking. The final scalar is one hundred times the highest maximum Jaccard similarity among the retained feasible mechanisms.

Returns
-------
Percentage equal to 100 times the winning maximum archive similarity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adaptive_moiety_mechanism_similarity(
    changes_by_radius: 'np.ndarray',
    radii: 'np.ndarray',
    rule_matrix: 'np.ndarray',
    initial_counts: 'np.ndarray',
    exempt_moieties: 'np.ndarray',
    rule_arrow_matrix: 'np.ndarray',
    reference_profiles: 'np.ndarray',
    max_steps: int,
    top_k: int,
    candidate_limit: int,
) -> float:
    """Run the seven preceding stages and return the winning similarity percent.

    The routine must call the preceding public functions. It selects the first
    informative radius, generates the requested parsimonious feasible
    mechanisms, constructs their arrow-environment profiles, scores and
    re-ranks them, and returns one hundred times the first row's similarity.

    Returns
    -------
    float
        Percentage equal to ``100`` times the winning maximum archive
        similarity.

    Raises
    ------
    ValueError
        If the supplied instance yields no feasible mechanism or an earlier
        public contract is violated.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_adaptive_moiety_mechanism_similarity(
    changes_by_radius: 'np.ndarray',
    radii: 'np.ndarray',
    rule_matrix: 'np.ndarray',
    initial_counts: 'np.ndarray',
    exempt_moieties: 'np.ndarray',
    rule_arrow_matrix: 'np.ndarray',
    reference_profiles: 'np.ndarray',
    max_steps: int,
    top_k: int,
    candidate_limit: int,
) -> float:
    resolved = _oracle_select_informative_radius(changes_by_radius, radii)
    target_change = resolved[1:]
    mechanisms = _oracle_generate_parsimonious_mechanisms(
        rule_matrix,
        target_change,
        initial_counts,
        exempt_moieties,
        max_steps,
        top_k,
        candidate_limit,
    )
    if mechanisms.shape[0] == 0:
        raise ValueError("the supplied instance yields no feasible mechanism")

    profiles = _oracle_build_arrow_environment_profiles(
        mechanisms,
        rule_arrow_matrix,
    )
    similarity = _oracle_maximum_archive_similarity(
        profiles,
        reference_profiles,
    )
    ranked = _oracle_rerank_mechanisms(
        mechanisms,
        similarity,
    )

    return float(100.0 * ranked[0, 0])


def _append_benchmark_rule(
    columns,
    n_moieties,
    consumed,
    produced,
    catalyst_delta=0,
):
    column = np.zeros(n_moieties, dtype=int)

    for index, amount in consumed.items():
        column[index] -= amount
    for index, amount in produced.items():
        column[index] += amount

    column[-1] += catalyst_delta
    columns.append(column)


def _benchmark_mechanism_fixture(
    permutation_seed,
    arrow_seed,
):
    n_layers = 5
    layer_width = 4
    source_index = 0
    layer_start = 1
    target_index = layer_start + n_layers * layer_width
    cycle_start = target_index + 1
    catalyst_index = cycle_start + 6
    n_moieties = catalyst_index + 1
    columns = []

    for offset in range(5):
        _append_benchmark_rule(
            columns,
            n_moieties,
            {cycle_start + offset: 1},
            {cycle_start + offset + 1: 1},
        )

    _append_benchmark_rule(
        columns,
        n_moieties,
        {cycle_start + 5: 1, source_index: 1},
        {cycle_start: 1, target_index: 1},
    )

    for node in range(layer_width):
        _append_benchmark_rule(
            columns,
            n_moieties,
            {source_index: 1},
            {layer_start + node: 1},
            -1,
        )

    for layer in range(n_layers - 1):
        left = layer_start + layer * layer_width
        right = left + layer_width

        for a in range(layer_width):
            for b in range(layer_width):
                _append_benchmark_rule(
                    columns,
                    n_moieties,
                    {left + a: 1},
                    {right + b: 1},
                )

    final_layer = layer_start + (n_layers - 1) * layer_width

    for node in range(layer_width):
        _append_benchmark_rule(
            columns,
            n_moieties,
            {final_layer + node: 1},
            {target_index: 1},
            1,
        )

    rule_matrix = np.stack(columns, axis=1)
    permutation_rng = np.random.default_rng(permutation_seed)
    rule_permutation = np.concatenate(
        (
            np.arange(6),
            permutation_rng.permutation(
                np.arange(6, rule_matrix.shape[1])
            ),
        )
    )
    rule_matrix = rule_matrix[:, rule_permutation]

    target = np.zeros(n_moieties, dtype=int)
    target[source_index], target[target_index] = -1, 1

    changes_by_radius = np.stack(
        (
            np.zeros_like(target),
            np.zeros_like(target),
            target,
            2 * target,
        )
    )
    radii = np.array([1, 2, 3, 4], dtype=int)

    initial_counts = np.zeros(n_moieties, dtype=int)
    initial_counts[source_index] = 1

    exempt_moieties = np.zeros(n_moieties, dtype=bool)
    exempt_moieties[catalyst_index] = True

    arrow_rng = np.random.default_rng(arrow_seed)
    rule_arrow_matrix = np.zeros(
        (rule_matrix.shape[1], 83),
        dtype=int,
    )

    for rule in range(rule_matrix.shape[1]):
        width = 4 + (3 * rule) % 5
        features = arrow_rng.choice(
            83,
            size=width,
            replace=False,
        )
        rule_arrow_matrix[rule, features] = 1

    reference_profiles = (
        arrow_rng.random((13, 83)) < 0.11
    ).astype(int)

    for reference in range(reference_profiles.shape[0]):
        reference_profiles[
            reference,
            (11 * reference + 5) % 83,
        ] = 1

    return (
        changes_by_radius,
        radii,
        rule_matrix,
        initial_counts,
        exempt_moieties,
        rule_arrow_matrix,
        reference_profiles,
        6,
        10,
        18,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ndef copy_args(values):\n return tuple(value.copy() if isinstance(value, np.ndarray) else value for value in values)\nargs=_benchmark_mechanism_fixture(26091331,26091343)",
            "call": "adaptive_moiety_mechanism_similarity(*copy_args(args))",
            "gold_call": "_oracle_adaptive_moiety_mechanism_similarity(*copy_args(args))",
        },
        {
            "setup": "import numpy as np\ndef copy_args(values):\n return tuple(value.copy() if isinstance(value, np.ndarray) else value for value in values)\nT=np.array([[-1,0,-1,0],[1,-1,0,0],[0,0,1,-1],[0,1,0,1]])\nt=np.array([-1,0,0,1])\nx=np.stack([np.zeros(4,dtype=int),t])\nr=np.array([1,2])\nc=np.array([1,0,0,0])\ne=np.zeros(4,dtype=bool)\na=np.array([[1,0,1,0,0],[0,1,1,0,0],[1,0,0,1,0],[0,1,0,1,1]])\nrefs=np.array([[1,1,1,0,0],[0,0,1,1,1]])",
            "call": "adaptive_moiety_mechanism_similarity(*copy_args((x,r,T,c,e,a,refs,2,2,4)))",
            "gold_call": "_oracle_adaptive_moiety_mechanism_similarity(*copy_args((x,r,T,c,e,a,refs,2,2,4)))",
        },
        {
            "setup": "import numpy as np\ndef copy_args(values):\n return tuple(value.copy() if isinstance(value, np.ndarray) else value for value in values)\nT=np.array([[-1,-1,1,0],[1,0,0,-1],[0,1,-1,0],[0,0,1,-1],[0,0,0,1]])\nt=np.array([-1,0,0,0,1])\nx=t.reshape(1,-1)\nr=np.array([1])\nc=np.array([1,0,0,0,0])\ne=np.zeros(5,dtype=bool)\na=np.eye(4,dtype=int)\nrefs=np.array([[1,1,1,1],[1,0,0,0]])",
            "call": "adaptive_moiety_mechanism_similarity(*copy_args((x,r,T,c,e,a,refs,4,1,2)))",
            "gold_call": "_oracle_adaptive_moiety_mechanism_similarity(*copy_args((x,r,T,c,e,a,refs,4,1,2)))",
        },
        {
            "setup": "import numpy as np\ndef copy_args(values):\n return tuple(value.copy() if isinstance(value, np.ndarray) else value for value in values)\nx=np.zeros((2,3),dtype=int)\nr=np.array([1,2])\nT=np.eye(3,dtype=int)\nc=np.zeros(3,dtype=int)\ne=np.zeros(3,dtype=bool)\na=np.eye(3,dtype=int)\nrefs=np.eye(3,dtype=int)\ndef check(fn):\n try: fn(*copy_args((x,r,T,c,e,a,refs,3,1,2)))\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(adaptive_moiety_mechanism_similarity)",
            "gold_call": "check(_oracle_adaptive_moiety_mechanism_similarity)",
        },
    ]
