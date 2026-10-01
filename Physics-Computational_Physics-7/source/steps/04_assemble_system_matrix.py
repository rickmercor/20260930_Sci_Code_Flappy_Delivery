"""
Implement `assemble_system_matrix` which builds the full coupled system matrix R.

The regions of a circulating-fuel loop have different local reaction conditions,
but their isotopic histories are linked by material exchange. Changes in one
region can affect later inventories elsewhere in the loop. A coupled description
is needed to account for these interactions across the full system.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_system_matrix(cell_matrices: list, outflow_diags: list,
                           inflow_blocks: list) -> "np.ndarray":
    '''Assembles the full coupled system matrix R.

    Parameters
    ----------
    cell_matrices : list
        List of n (m, m) per-cell depletion matrices from build_cell_matrix.
    outflow_diags : list
        List of n (m, m) outflow diagonal matrices from build_flow_coupling.
    inflow_blocks : list of lists
        n × n list from build_flow_coupling. inflow_blocks[k][l] is an (m, m)
        flow matrix from cell k to cell l, placed at block row l, block column k.

    Returns
    -------
    R : np.ndarray
        Shape (m*n, m*n) system matrix.

    Raises
    ------
    ValueError
        If cell_matrices is empty.
    '''
    return R

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_system_matrix(cell_matrices: list, outflow_diags: list, inflow_blocks: list) -> "np.ndarray":
    """Reference implementation."""
    if len(cell_matrices) == 0:
        raise ValueError("cell_matrices must be non-empty")
    n = len(cell_matrices)
    m = cell_matrices[0].shape[0]
    size = m * n
    R = np.zeros((size, size))

    for k in range(n):
        # Diagonal block: cell matrix + outflow
        row_start = k * m
        row_end = (k + 1) * m
        R[row_start:row_end, row_start:row_end] = cell_matrices[k] + outflow_diags[k]

        # Add all inflows, including flow returning to the same cell.
        for j in range(n):
            col_start = j * m
            col_end = (j + 1) * m
            # inflow_blocks[j][k] = flow FROM cell j TO cell k
            R[row_start:row_end, col_start:col_end] += inflow_blocks[j][k]

    return R

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
                m = 2
                # Cell matrices (simple decay-only)
                A1 = np.array([[-0.1, 0.0], [0.05, -0.2]])
                A2 = np.array([[-0.05, 0.0], [0.0, -0.1]])
                cell_matrices = [A1, A2]
                # Outflow
                O1 = -0.5 * np.eye(m)
                O2 = -0.5 * np.eye(m)
                outflow_diags = [O1, O2]
                # Inflow: cell 0 -> cell 1 and cell 1 -> cell 0
                F01 = np.zeros((m, m))  # from 0 to 0
                F10 = np.zeros((m, m))  # from 1 to 0
                inflow_01 = 0.5 * np.eye(m)  # from 0 to 1
                inflow_10 = 0.5 * np.eye(m)  # from 1 to 0
                inflow_blocks = [[F01, inflow_01], [inflow_10, F10]]
                """
            ),
            "call": dedent(
                """\
                assemble_system_matrix(
                    *copy.deepcopy(
                        (
                            cell_matrices,
                            outflow_diags,
                            inflow_blocks,
                        )
                    )
                )
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_assemble_system_matrix(
                    *copy.deepcopy(
                        (
                            cell_matrices,
                            outflow_diags,
                            inflow_blocks,
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
                m = 4
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
                volumes = np.array([1.3e6, 6.2e5, 8.1e4])
                Q = 7.5e4
                flow_fractions = np.array([
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ])


                def build_inputs(transmutation, cell, coupling):
                    T_core = transmutation(
                        *copy.deepcopy(
                            (6.0e14, sigma_gamma, sigma_f, fission_yields)
                        )
                    )
                    T_zero = transmutation(
                        *copy.deepcopy(
                            (0.0, sigma_gamma, sigma_f, fission_yields)
                        )
                    )
                    cell_matrices = [
                        cell(
                            *copy.deepcopy(
                                (
                                    T_core,
                                    decay_constants,
                                    branching_ratios,
                                    np.zeros(4),
                                )
                            )
                        ),
                        cell(
                            *copy.deepcopy(
                                (
                                    T_zero,
                                    decay_constants,
                                    branching_ratios,
                                    np.zeros(4),
                                )
                            )
                        ),
                        cell(
                            *copy.deepcopy(
                                (
                                    T_zero,
                                    decay_constants,
                                    branching_ratios,
                                    np.array([0.0, 0.0, 6.9e-4, 0.0]),
                                )
                            )
                        ),
                    ]
                    outflow_diags, inflow_blocks = coupling(
                        *copy.deepcopy((3, m, volumes, Q, flow_fractions))
                    )
                    return cell_matrices, outflow_diags, inflow_blocks


                model_inputs = build_inputs(
                    build_transmutation_matrix,
                    build_cell_matrix,
                    build_flow_coupling,
                )
                gold_inputs = build_inputs(
                    _oracle_build_transmutation_matrix,
                    _oracle_build_cell_matrix,
                    _oracle_build_flow_coupling,
                )
                """
            ),
            "call": dedent(
                """\
                assemble_system_matrix(*copy.deepcopy(model_inputs))
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_assemble_system_matrix(*copy.deepcopy(gold_inputs))
                """
            ),
        },
        {
            "setup": dedent(
                """\
                import copy

                import numpy as np
                m = 2
                A1 = np.array([[-1.0, 0.5], [0.3, -0.8]])
                cell_matrices = [A1]
                outflow_diags = [np.zeros((m, m))]
                inflow_blocks = [[np.zeros((m, m))]]
                """
            ),
            "call": dedent(
                """\
                assemble_system_matrix(
                    *copy.deepcopy(
                        (
                            cell_matrices,
                            outflow_diags,
                            inflow_blocks,
                        )
                    )
                )
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_assemble_system_matrix(
                    *copy.deepcopy(
                        (
                            cell_matrices,
                            outflow_diags,
                            inflow_blocks,
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
                cell_matrices = [np.zeros((2, 2))]
                outflow_diags = [-0.5 * np.eye(2)]
                inflow_blocks = [[0.5 * np.eye(2)]]
                """
            ),
            "call": dedent(
                """\
                assemble_system_matrix(
                    *copy.deepcopy(
                        (
                            cell_matrices,
                            outflow_diags,
                            inflow_blocks,
                        )
                    )
                )
                """
            ),
            "gold_call": dedent(
                """\
                _oracle_assemble_system_matrix(
                    *copy.deepcopy(
                        (
                            cell_matrices,
                            outflow_diags,
                            inflow_blocks,
                        )
                    )
                )
                """
            ),
        },
    ]
