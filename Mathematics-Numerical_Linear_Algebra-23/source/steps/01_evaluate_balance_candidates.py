"""
Evaluate the Balance Method 2 candidates formed by the supplied harmonic Ritz roots, preserving complex-conjugate pairs as indivisible candidates and identifying the candidate closest to the initial derivative quantity.

For harmonic Ritz roots $\\theta_i$, define the derivative quantity



$$

\\varphi'(0) = \\sum_i \\frac{1}{\\theta_i}.

$$



For a valid real-coefficient residual polynomial, every nonreal harmonic Ritz root must occur together with its complex conjugate, so $\\varphi'(0)$ is real up to numerical roundoff.



Treat each real root $\\theta$ as one candidate with reciprocal contribution



$$

\\xi = \\frac{1}{\\theta}.

$$



Treat each complex-conjugate pair $(\\theta, \\overline{\\theta})$ as one indivisible candidate with reciprocal contribution



$$

\\xi = \\frac{1}{\\theta} + \\frac{1}{\\overline{\\theta}}.

$$



For every candidate, compute



$$

d = \\left|\\varphi'(0) - \\xi\\right|.

$$



Numerical conventions, with $\\tau = 10^{-12}$: a root $\\theta$ is treated as real when $|\\operatorname{Im}\\theta| \\le \\tau \\max(1, |\\theta|)$, and its real part is then used. A root with $|\\theta| \\le \\tau$ is invalid. Conjugate partners are matched by scanning the input in index order: for a nonreal root at index $i$ that is not yet paired, its partner is the smallest unpaired index $j > i$ with $|\\theta_j - \\overline{\\theta_i}| \\le \\tau \\max(1, |\\theta_i|, |\\theta_j|)$. Identical conjugate pairs occurring more than once therefore form separate candidates with distinct partner indices. Repeated real roots form separate candidates, one per occurrence.



Select the candidate having the smallest value of $d$. If two candidates have exactly the same difference, select the candidate with the smallest primary source index.



All source indices in the returned table are zero-based NumPy indices.



Return one row per candidate, ordered by primary source index. Each row must contain



$$

[\\text{primary\\_index},\\ \\text{partner\\_index},\\ \\text{multiplicity},\\ \\xi,\\ d,\\ \\varphi'(0),\\ \\text{selected\\_flag}].

$$



For a real root, $\\text{partner\\_index} = -1$ and $\\text{multiplicity} = 1$. For a complex-conjugate pair, $\\text{multiplicity} = 2$ and $\\text{partner\\_index}$ is the zero-based index of the conjugate partner.



$\\text{selected\\_flag} = 1$ only for the selected candidate; all other rows use $\\text{selected\\_flag} = 0$.



The returned values of $\\xi$, $d$, and $\\varphi'(0)$ must be real-valued.



Invalid input (empty or non-one-dimensional array, non-finite entries, a root equal to zero, or a nonreal root without a conjugate partner) must raise ValueError.

Returns
-------
A real NumPy array of shape (n_candidates, 7), ordered by primary source index.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_balance_candidates(
    harmonic_ritz_roots: np.ndarray,
) -> np.ndarray:
    """
    Evaluate Balance Method 2 candidates from harmonic Ritz roots.

    Parameters
    ----------
    harmonic_ritz_roots : np.ndarray
        One-dimensional array of nonzero harmonic Ritz roots. Every
        nonreal root must have a complex-conjugate partner.

    Returns
    -------
    np.ndarray
        Candidate table with one row per real root or conjugate pair
        and columns
        [primary_index, partner_index, multiplicity, contribution,
         difference, phi_prime_0, selected_flag].

    Raises
    ------
    ValueError
        If the array is empty or not one-dimensional, contains non-finite
        or zero entries, or a nonreal root has no conjugate partner.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_balance_candidates(
    harmonic_ritz_roots,
):
    roots = np.asarray(harmonic_ritz_roots, dtype=complex)

    if roots.ndim != 1 or roots.size == 0:
        raise ValueError(
            "harmonic_ritz_roots must be a non-empty one-dimensional array."
        )

    if not np.all(
        np.isfinite(roots.real) & np.isfinite(roots.imag)
    ):
        raise ValueError(
            "harmonic_ritz_roots must contain only finite values."
        )

    tol = 1.0e-12

    if np.any(np.abs(roots) <= tol):
        raise ValueError(
            "harmonic Ritz roots must be nonzero."
        )

    reciprocal_sum = np.sum(1.0 / roots)

    scale = max(
        1.0,
        abs(reciprocal_sum),
    )

    if abs(reciprocal_sum.imag) > tol * scale:
        raise ValueError(
            "The reciprocal sum must be real for a conjugate-symmetric root set."
        )

    phi_prime_0 = float(reciprocal_sum.real)

    used = np.zeros(
        roots.size,
        dtype=bool,
    )

    candidates = []

    for i, root in enumerate(roots):
        if used[i]:
            continue

        root_scale = max(
            1.0,
            abs(root),
        )

        if abs(root.imag) <= tol * root_scale:
            contribution = float(
                (1.0 / root).real
            )

            candidates.append(
                (
                    i,
                    -1,
                    1,
                    contribution,
                )
            )

            used[i] = True
            continue

        partner = None
        target = np.conjugate(root)

        for j in range(i + 1, roots.size):
            if used[j]:
                continue

            pair_scale = max(
                1.0,
                abs(target),
                abs(roots[j]),
            )

            if abs(roots[j] - target) <= tol * pair_scale:
                partner = j
                break

        if partner is None:
            raise ValueError(
                "Every nonreal harmonic Ritz root must have a complex-conjugate partner."
            )

        pair_contribution = (
            1.0 / root
            +
            1.0 / roots[partner]
        )

        pair_scale = max(
            1.0,
            abs(pair_contribution),
        )

        if abs(pair_contribution.imag) > tol * pair_scale:
            raise ValueError(
                "A conjugate-pair reciprocal contribution must be real."
            )

        contribution = float(
            pair_contribution.real
        )

        candidates.append(
            (
                i,
                partner,
                2,
                contribution,
            )
        )

        used[i] = True
        used[partner] = True

    table = np.zeros(
        (len(candidates), 7),
        dtype=float,
    )

    for row, (
        primary,
        partner,
        multiplicity,
        contribution,
    ) in enumerate(candidates):
        difference = abs(
            phi_prime_0
            -
            contribution
        )

        table[row, 0] = primary
        table[row, 1] = partner
        table[row, 2] = multiplicity
        table[row, 3] = contribution
        table[row, 4] = difference
        table[row, 5] = phi_prime_0

    selected_row = min(
        range(table.shape[0]),
        key=lambda k: (
            table[k, 4],
            table[k, 0],
        ),
    )

    table[selected_row, 6] = 1.0

    return table

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    -5.0,
    -1.3,
    0.9,
    2.0 + 1.5j,
    2.0 - 1.5j
], dtype=complex)
""",
            "call": """evaluate_balance_candidates(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_evaluate_balance_candidates(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    -6.0,
    -2.0,
    0.8,
    3.5
], dtype=complex)
""",
            "call": """evaluate_balance_candidates(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_evaluate_balance_candidates(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    1.25,
    2.0 + 1.0j,
    -3.0 + 1.0e-14j,
    2.0 - 1.0j,
    -0.75
], dtype=complex)
""",
            "call": """evaluate_balance_candidates(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_evaluate_balance_candidates(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    2.0,
    -2.0,
    1.0 + 1.0j,
    1.0 - 1.0j,
    1.0 + 1.0j,
    1.0 - 1.0j
], dtype=complex)
""",
            "call": """evaluate_balance_candidates(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_evaluate_balance_candidates(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    0.5,
    0.5,
    -3.0,
    -6.0
], dtype=complex)
""",
            "call": """evaluate_balance_candidates(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_evaluate_balance_candidates(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np


def _run_invalid(fn, *args):
    try:
        fn(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

import numpy as np

harmonic_ritz_roots = np.array([
    -5.0,
    -1.3,
    2.0 + 1.5j
], dtype=complex)
""",
            "call": """_run_invalid(
    evaluate_balance_candidates, harmonic_ritz_roots
)""",
            "gold_call": """_run_invalid(
    _oracle_evaluate_balance_candidates, harmonic_ritz_roots
)""",
        },
    ]
