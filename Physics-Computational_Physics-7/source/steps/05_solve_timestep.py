"""
Implement `solve_timestep` which advances the isotopic state vector by one time step with a continuously acting external source.

Irradiation, radioactive decay and circulation can act on very different
timescales within a depletion calculation. External feeds also add material
throughout the evolution. A finite interval must account for both the changing
inventory and the continuing supply of new nuclides.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_timestep(R: "np.ndarray", dt: float, N: "np.ndarray",
                   S: "np.ndarray") -> "np.ndarray":
    '''Evolves the linear system dN/dt = R N + S over the supplied time step.

    Parameters
    ----------
    R : np.ndarray
        Shape (n, n) system matrix, constant during the step.
        Singular and nearly singular matrices are valid inputs.
    dt : float
        Time step [s].
    N : np.ndarray
        Shape (n,) current state vector of isotopic number densities.
    S : np.ndarray
        Shape (n,) external source vector [atoms·barn⁻¹·cm⁻¹·s⁻¹],
        constant throughout the step. May be all zeros.

    Returns
    -------
    N_new : np.ndarray
        Shape (n,) state vector at time t + dt, including the continuously
        acting source.

    Raises
    ------
    ValueError
        If dt is not positive.
    '''
    return N_new

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _cram16_action(A, v):
    """Ordinary PFD action; Pusa (2012), Eq. (5) and Table 2."""
    THETA = np.array([
     -1.0843917078696988026e1 + 1.9277446167181652284e1j,
     -5.2649713434426468895 + 1.6220221473167927305e1j,
      5.9481522689511774808 + 3.5874573620183222829j,
      3.5091036084149180974 + 8.4361989858843750826j,
      6.4161776990994341923 + 1.1941223933701386874j,
      1.4193758971856659786 + 1.0925363484496722585e1j,
      4.9931747377179963991 + 5.9968817136039422260j,
     -1.4139284624888862114 + 1.3497725698892745389e1j,
    ])
    ALPHA = np.array([
     -5.0901521865224915650e-7 - 2.4220017652852287970e-5j,
      2.1151742182466030907e-4 + 4.3892969647380673918e-3j,
      1.1339775178483930527e2 + 1.0194721704215856450e2j,
      1.5059585270023467528e1 - 5.7514052776421819979j,
     -6.4500878025539646595e1 - 2.2459440762652096056e2j,
     -1.4793007113557999718 + 1.7686588323782937906j,
     -6.2518392463207918892e1 - 1.1190391094283228480e1j,
      4.1023136835410021273e-2 - 1.5743466173455468191e-1j,
    ])
    ALPHA0 = 2.1248537104952237488e-16
    identity = np.eye(A.shape[0])
    terms = [np.real(alpha * np.linalg.solve(A - theta * identity, v))
             for theta, alpha in zip(THETA, ALPHA)]
    return ALPHA0 * v + 2.0 * np.sum(terms, axis=0)

def _oracle_solve_timestep(R: "np.ndarray", dt: float, N: "np.ndarray", S: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if not np.any(S != 0.0):
        return _cram16_action(R * dt, N)

    # A constant final coordinate integrates the source without cancellation.
    size = R.shape[0]
    augmented = np.zeros((size + 1, size + 1))
    augmented[:size, :size] = R
    augmented[:size, size] = S
    return _cram16_action(augmented * dt, np.r_[N, 1.0])[:size]

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
                from scipy.linalg import expm
                R = np.array([[-0.1, 0.0], [0.1, -0.01]])
                dt = 10.0
                N = np.array([1.0, 0.0])
                S = np.array([0.0, 0.0])
                """
            ),
            "call": dedent(
                """\
                solve_timestep(
                    *copy.deepcopy(
                        (
                            R,
                            dt,
                            N,
                            S,
                        )
                    )
                )
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_solve_timestep(
                    *copy.deepcopy(
                        (
                            R,
                            dt,
                            N,
                            S,
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
                from scipy.linalg import expm
                R = np.array([[-0.1, 0.0], [0.1, -0.01]])
                dt = 10.0
                N = np.array([1.0, 0.0])
                S = np.array([0.05, 0.0])  # constant feed of species 0
                """
            ),
            "call": dedent(
                """\
                solve_timestep(
                    *copy.deepcopy(
                        (
                            R,
                            dt,
                            N,
                            S,
                        )
                    )
                )
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_solve_timestep(
                    *copy.deepcopy(
                        (
                            R,
                            dt,
                            N,
                            S,
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
                from scipy.linalg import expm
                sigma_gamma = np.array([98.71, 80.03, 2.778e6, 8.302])
                sigma_f = np.array([585.1, 0.0, 0.0, 0.0])
                fission_yields = np.zeros((4, 4))
                fission_yields[0, 1] = 0.0628
                fission_yields[0, 2] = 0.0016
                decay_constants = np.array(
                    [3.12e-17, 2.93e-5, 2.11e-5, 9.55e-15]
                )
                branching_ratios = np.zeros((4, 4))
                branching_ratios[1, 2] = 1.0
                branching_ratios[2, 3] = 1.0
                transmutation_args = (
                    6.0e14, sigma_gamma, sigma_f, fission_yields
                )
                T_model = build_transmutation_matrix(
                    *copy.deepcopy(transmutation_args)
                )
                T_gold = _oracle_build_transmutation_matrix(
                    *copy.deepcopy(transmutation_args)
                )
                R_model = build_cell_matrix(
                    *copy.deepcopy(
                        (
                            T_model,
                            decay_constants,
                            branching_ratios,
                            np.zeros(4),
                        )
                    )
                )
                R_gold = _oracle_build_cell_matrix(
                    *copy.deepcopy(
                        (
                            T_gold,
                            decay_constants,
                            branching_ratios,
                            np.zeros(4),
                        )
                    )
                )
                dt = 3600.0
                N = np.array([8.5e-4, 0.0, 0.0, 0.0])
                S = np.zeros(4)
                """
            ),
            "call": dedent(
                """\
                solve_timestep(
                    *copy.deepcopy(
                        (
                            R_model,
                            dt,
                            N,
                            S,
                        )
                    )
                )
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_solve_timestep(
                    *copy.deepcopy(
                        (
                            R_gold,
                            dt,
                            N,
                            S,
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
                from scipy.linalg import expm
                R = np.array([[0.0]])
                dt = 1.0
                N = np.array([1.0])
                S = np.array([2.0])
                """
            ),
            "call": dedent(
                """\
                solve_timestep(
                    *copy.deepcopy(
                        (
                            R,
                            dt,
                            N,
                            S,
                        )
                    )
                )
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_solve_timestep(
                    *copy.deepcopy(
                        (
                            R,
                            dt,
                            N,
                            S,
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
                from scipy.linalg import expm
                R = np.array([[-1e-20]])
                dt = 1.0
                N = np.array([1.0])
                S = np.array([2.0])
                """
            ),
            "call": dedent(
                """\
                solve_timestep(
                    *copy.deepcopy(
                        (
                            R,
                            dt,
                            N,
                            S,
                        )
                    )
                )
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_solve_timestep(
                    *copy.deepcopy(
                        (
                            R,
                            dt,
                            N,
                            S,
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
                from scipy.linalg import expm
                R = np.array([[0.0, 1.0], [0.0, 0.0]])
                dt = 1.0
                N = np.array([1.0, 1.0])
                S = np.array([0.0, 2.0])
                """
            ),
            "call": dedent(
                """\
                solve_timestep(
                    *copy.deepcopy(
                        (
                            R,
                            dt,
                            N,
                            S,
                        )
                    )
                )
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_solve_timestep(
                    *copy.deepcopy(
                        (
                            R,
                            dt,
                            N,
                            S,
                        )
                    )
                )
                """
            ),
        },
    ]
