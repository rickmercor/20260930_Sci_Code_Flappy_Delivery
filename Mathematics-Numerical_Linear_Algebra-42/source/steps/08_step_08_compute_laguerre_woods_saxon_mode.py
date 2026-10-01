"""
Run the complete half-line collocation pipeline and report one mode magnitude.



This final stage composes the simultaneous GLR root/derivative state, the direct

first/second collocation block, physical scaling, stable Woods--Saxon sampling,

generalized-pencil assembly, homogeneous finite-spectrum extraction, and

one-based mode selection.  The physical task uses the classical Laguerre case

`$alpha = 0$` while retaining the full-grid degree convention.

Returns
-------
one finite float: the mode_index-th smallest magnitude among finite eigenvalues
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_laguerre_woods_saxon_mode(
    n: int,
    beta: float,
    radius: float,
    diffuseness: float,
    mode_index: int,
) -> float:
    """Return one magnitude-ordered generalized eigenvalue.

    ``n`` and ``mode_index`` must be non-boolean integers with ``n >= 1`` and
    ``1 <= mode_index <= n``.  ``beta``, ``radius`` and ``diffuseness`` must be
    finite positive scalars.  Every validation rule of the earlier stages also
    applies.  A ``ValueError`` is raised if fewer than ``mode_index`` finite
    eigenvalues are available or the result is nonfinite.

    Parameters
    ----------
    n : int
        Number of positive roots and reduced pencil dimension.
    beta : float
        Positive coordinate scale.
    radius : float
        Positive potential radius.
    diffuseness : float
        Positive potential surface thickness.
    mode_index : int
        One-based index after sorting finite eigenvalue magnitudes.

    Returns
    -------
    float
        The selected finite eigenvalue magnitude.
    """
    return mode_magnitude  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_laguerre_woods_saxon_mode(
    n: int,
    beta: float,
    radius: float,
    diffuseness: float,
    mode_index: int,
) -> float:
    """Reference end-to-end composition of the seven earlier oracles."""
    if isinstance(mode_index, (bool, np.bool_)) or not isinstance(
        mode_index, (int, np.integer)
    ):
        raise ValueError("mode_index must be an integer")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    mode_index = int(mode_index)
    if n < 1 or mode_index < 1 or mode_index > n:
        raise ValueError("mode_index must lie between one and n")

    alpha = 0.0
    nodes, derivatives = _oracle_compute_glr_laguerre_state(n)  # noqa: F821
    collocation = _oracle_build_truncated_collocation_state(  # noqa: F821
        n, alpha, nodes, derivatives
    )
    physical_nodes, reduced_d2 = _oracle_scale_and_reduce_second_order(  # noqa: F821
        nodes, collocation[1], beta
    )
    potential = _oracle_sample_woods_saxon_profile(  # noqa: F821
        physical_nodes, radius, diffuseness
    )
    a_matrix, q_matrix = _oracle_assemble_generalized_pencil(  # noqa: F821
        reduced_d2, potential
    )
    magnitudes = _oracle_compute_finite_spectral_magnitudes(  # noqa: F821
        a_matrix, q_matrix
    )
    return _oracle_select_one_based_mode(magnitudes, mode_index)  # noqa: F821

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return four whole-pipeline cases and one invalid-index case."""
    return [
        {
            "setup": """n = 160
beta = 10.0
radius = 5.08685476
diffuseness = 0.929852862
mode_index = 25
""",
            "call": "compute_laguerre_woods_saxon_mode(n, beta, radius, diffuseness, mode_index)",
            "gold_call": "_oracle_compute_laguerre_woods_saxon_mode(n, beta, radius, diffuseness, mode_index)",
        },
        {
            "setup": """n = 64
beta = 10.0
radius = 5.08685476
diffuseness = 0.929852862
mode_index = 5
""",
            "call": "compute_laguerre_woods_saxon_mode(n, beta, radius, diffuseness, mode_index)",
            "gold_call": "_oracle_compute_laguerre_woods_saxon_mode(n, beta, radius, diffuseness, mode_index)",
        },
        {
            "setup": """n = 96
beta = 8.0
radius = 5.08685476
diffuseness = 0.929852862
mode_index = 10
""",
            "call": "compute_laguerre_woods_saxon_mode(n, beta, radius, diffuseness, mode_index)",
            "gold_call": "_oracle_compute_laguerre_woods_saxon_mode(n, beta, radius, diffuseness, mode_index)",
        },
        {
            "setup": """n = 128
beta = 12.0
radius = 4.5
diffuseness = 0.8
mode_index = 20
""",
            "call": "compute_laguerre_woods_saxon_mode(n, beta, radius, diffuseness, mode_index)",
            "gold_call": "_oracle_compute_laguerre_woods_saxon_mode(n, beta, radius, diffuseness, mode_index)",
        },
        {
            "setup": """n = 12
beta = 10.0
radius = 5.08685476
diffuseness = 0.929852862
mode_index = 13
def run_model():
    try:
        compute_laguerre_woods_saxon_mode(n, beta, radius, diffuseness, mode_index)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_laguerre_woods_saxon_mode(n, beta, radius, diffuseness, mode_index)
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
