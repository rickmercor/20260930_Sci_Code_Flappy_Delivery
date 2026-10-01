"""
Generate the first feasible mechanisms from cut-ranked rule sets.

The decomposed source workflow first generates balanced rule multisets with minRules and then submits them to to OrderRules. A balanced multiset that cannot be arranged without prematurely consuming a required intermediate is rejected. Candidate ranks must be preserved when infeasible multisets are skipped so that later similarity re-ranking remains traceable to the original parsimony order.

Returns
-------
Integer table containing original candidate rank, step count, and the padded one-based ordered rule sequence.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generate_parsimonious_mechanisms(
    rule_matrix: 'np.ndarray',
    target_change: 'np.ndarray',
    initial_counts: 'np.ndarray',
    exempt_moieties: 'np.ndarray',
    max_steps: int,
    top_k: int,
    candidate_limit: int,
) -> 'np.ndarray':
    """Filter cut-ranked balanced multisets through chemical ordering.

    Obtain up to ``candidate_limit`` rule-count vectors from
    ``rank_parsimonious_rule_sets``. Test them in that order with
    ``order_rule_multiset``, skip every multiset without a feasible ordering,
    and retain at most ``top_k`` mechanisms.

    Returns
    -------
    np.ndarray
        Integer array with ``max_steps + 2`` columns. Each row stores the
        one-based rank of its balanced candidate, its step count, and its
        one-based ordered rule labels padded on the right with zeros.

    Raises
    ------
    ValueError
        If scalar search bounds are not positive integers, an array violates
        an upstream step contract, or an ordered result cannot fit the
        declared maximum length.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_generate_parsimonious_mechanisms(
    rule_matrix: 'np.ndarray',
    target_change: 'np.ndarray',
    initial_counts: 'np.ndarray',
    exempt_moieties: 'np.ndarray',
    max_steps: int,
    top_k: int,
    candidate_limit: int,
) -> 'np.ndarray':
    import numpy as np

    for value, name in (
        (max_steps, "max_steps"),
        (top_k, "top_k"),
        (candidate_limit, "candidate_limit"),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")

    ranked = _oracle_rank_parsimonious_rule_sets(
        rule_matrix,
        target_change,
        int(max_steps),
        int(candidate_limit),
    )

    rows = []

    for candidate_rank, row in enumerate(ranked, start=1):
        n_steps = int(row[0])
        sequence = _oracle_order_rule_multiset(
            row[1:],
            rule_matrix,
            initial_counts,
            exempt_moieties,
        )

        if sequence.size == 0:
            continue
        if sequence.size != n_steps or n_steps > int(max_steps):
            raise ValueError(
                "ordered mechanism length is inconsistent"
            )

        padded = np.zeros(int(max_steps), dtype=int)
        padded[:n_steps] = sequence

        rows.append(
            np.concatenate(
                (
                    np.array(
                        [candidate_rank, n_steps],
                        dtype=int,
                    ),
                    padded,
                )
            )
        )

        if len(rows) == int(top_k):
            break

    if not rows:
        return np.empty(
            (0, int(max_steps) + 2),
            dtype=int,
        )

    return np.vstack(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nT=np.array([[-0,-0,-1,-1,0,0],[0,0,1,0,0,1],[-1,0,1,0,0,0],[1,-1,0,1,-1,0],[0,1,-1,0,1,-1],[0,0,0,0,0,1]])\nt=np.array([-1,1,0,0,0,0])\nc=np.array([1,0,0,0,0,0])\ne=np.zeros(6,dtype=bool)",
            "call": "generate_parsimonious_mechanisms(T.copy(),t.copy(),c.copy(),e.copy(),3,1,4)",
            "gold_call": "_oracle_generate_parsimonious_mechanisms(T.copy(),t.copy(),c.copy(),e.copy(),3,1,4)",
        },
        {
            "setup": "import numpy as np\nT=np.array([[-1,0,-1,0],[1,-1,0,0],[0,0,1,-1],[0,1,0,1]])\nt=np.array([-1,0,0,1])\nc=np.array([1,0,0,0])\ne=np.zeros(4,dtype=bool)",
            "call": "generate_parsimonious_mechanisms(T.copy(),t.copy(),c.copy(),e.copy(),2,2,4)",
            "gold_call": "_oracle_generate_parsimonious_mechanisms(T.copy(),t.copy(),c.copy(),e.copy(),2,2,4)",
        },
        {
            "setup": "import numpy as np\nT=np.array([[-1,0],[1,-1],[0,1],[-1,1]])\nt=np.array([-1,0,1,0])\nc=np.array([1,0,0,0])\ne=np.array([False,False,False,True])",
            "call": "generate_parsimonious_mechanisms(T.copy(),t.copy(),c.copy(),e.copy(),2,1,3)",
            "gold_call": "_oracle_generate_parsimonious_mechanisms(T.copy(),t.copy(),c.copy(),e.copy(),2,1,3)",
        },
        {
            "setup": "import numpy as np\nT=np.array([[0,-1,-1,0],[-1,1,0,0],[1,-1,0,0],[0,1,0,1],[0,0,1,-1]])\nt=np.array([-1,0,0,1,0])\nc=np.array([1,0,0,0,0])\ne=np.zeros(5,dtype=bool)",
            "call": "generate_parsimonious_mechanisms(T.copy(),t.copy(),c.copy(),e.copy(),2,1,2)",
            "gold_call": "_oracle_generate_parsimonious_mechanisms(T.copy(),t.copy(),c.copy(),e.copy(),2,1,2)",
        },
        {
            "setup": "import numpy as np\nT=np.eye(2,dtype=int)\nt=np.array([1,0])\nc=np.zeros(2,dtype=int)\ne=np.zeros(2,dtype=bool)\ndef check(fn):\n try: fn(T.copy(),t.copy(),c.copy(),e.copy(),2,0,3)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(generate_parsimonious_mechanisms)",
            "gold_call": "check(_oracle_generate_parsimonious_mechanisms)",
        },
    ]
