"""
Implement `build_cell_matrix` which adds radioactive decay and engineered removal
to the transmutation matrix to form the complete single-cell depletion matrix.

Radioactive decay changes the isotopic inventory throughout the fuel loop,
including regions outside the neutron flux. Decay branches connect parent and
daughter nuclides, while chemical processing can remove selected species from
the salt. These processes act alongside neutron-induced reactions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_cell_matrix(T: "np.ndarray",
                      decay_constants: "np.ndarray", branching_ratios: "np.ndarray",
                      removal_rates: "np.ndarray") -> "np.ndarray":
    '''Adds decay and removal terms to the transmutation matrix.

    Parameters
    ----------
    T : np.ndarray
        Shape (m, m) transmutation matrix from step 1.
    decay_constants : np.ndarray
        Shape (m,) decay constants [s⁻¹].
    branching_ratios : np.ndarray
        Shape (m, m) decay branching ratios. branching_ratios[j, i] = BR for j → i.
    removal_rates : np.ndarray
        Shape (m,) engineered removal rates [s⁻¹] (0 if not removed).

    Returns
    -------
    A : np.ndarray
        Shape (m, m) complete depletion matrix.

    Raises
    ------
    ValueError
        If any decay constant is negative.
    '''
    return A

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_cell_matrix(T: "np.ndarray", decay_constants: "np.ndarray", branching_ratios: "np.ndarray", removal_rates: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    if np.any(np.asarray(decay_constants) < 0):
        raise ValueError("decay constants must be non-negative")
    m = T.shape[0]
    A = T.copy()

    for i in range(m):
        # Diagonal: decay loss and engineered removal
        A[i, i] -= decay_constants[i] + removal_rates[i]

        for j in range(m):
            if j == i:
                continue
            # Decay production: b_{j->i} * lambda_j
            A[i, j] += branching_ratios[j, i] * decay_constants[j]

    return A

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the retained cases with independent mutable inputs."""
    from textwrap import dedent

    return [
        {
            "setup": dedent(
                """\
                import copy

                import numpy as np

                flux = 6.0e14
                sigma_gamma = np.array([98.71, 80.03, 2.778e6, 8.302])
                sigma_f = np.array([585.1, 0.0, 0.0, 0.0])
                fission_yields = np.zeros((4, 4))
                fission_yields[0, 1] = 0.0628
                fission_yields[0, 2] = 0.0016
                transmutation_args = (
                    flux, sigma_gamma, sigma_f, fission_yields
                )
                T_model = build_transmutation_matrix(
                    *copy.deepcopy(transmutation_args)
                )
                T_gold = _oracle_build_transmutation_matrix(
                    *copy.deepcopy(transmutation_args)
                )
                decay_constants = np.array(
                    [3.12e-17, 2.93e-5, 2.11e-5, 9.55e-15]
                )
                branching_ratios = np.zeros((4, 4))
                branching_ratios[1, 2] = 1.0
                branching_ratios[2, 3] = 1.0
                removal_rates = np.zeros(4)
                """
            ),
            "call": dedent(
                """\
                build_cell_matrix(
                    *copy.deepcopy(
                        (
                            T_model,
                            decay_constants,
                            branching_ratios,
                            removal_rates,
                        )
                    )
                )
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_build_cell_matrix(
                    *copy.deepcopy(
                        (
                            T_gold,
                            decay_constants,
                            branching_ratios,
                            removal_rates,
                        )
                    )
                )
                """
            ),
        },
        {
            "setup": dedent(
                """\
                import copy

                import numpy as np
                T = np.zeros((4, 4))
                decay_constants = np.array(
                    [3.12e-17, 2.93e-5, 2.11e-5, 9.55e-15]
                )
                branching_ratios = np.zeros((4, 4))
                branching_ratios[1, 2] = 1.0
                branching_ratios[2, 3] = 1.0
                removal_rates = np.array([0.0, 0.0, 6.9e-4, 0.0])
                """
            ),
            "call": dedent(
                """\
                build_cell_matrix(
                    *copy.deepcopy(
                        (
                            T,
                            decay_constants,
                            branching_ratios,
                            removal_rates,
                        )
                    )
                )
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_build_cell_matrix(
                    *copy.deepcopy(
                        (
                            T,
                            decay_constants,
                            branching_ratios,
                            removal_rates,
                        )
                    )
                )
                """
            ),
        },
        {
            "setup": dedent(
                """\
                import copy

                import numpy as np
                T = np.zeros((4, 4))
                decay_constants = np.array(
                    [3.12e-17, 2.93e-5, 2.11e-5, 9.55e-15]
                )
                branching_ratios = np.zeros((4, 4))
                branching_ratios[1, 2] = 1.0
                branching_ratios[2, 3] = 1.0
                removal_rates = np.zeros(4)
                """
            ),
            "call": dedent(
                """\
                build_cell_matrix(
                    *copy.deepcopy(
                        (
                            T,
                            decay_constants,
                            branching_ratios,
                            removal_rates,
                        )
                    )
                )
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_build_cell_matrix(
                    *copy.deepcopy(
                        (
                            T,
                            decay_constants,
                            branching_ratios,
                            removal_rates,
                        )
                    )
                )
                """
            ),
        },
    ]
