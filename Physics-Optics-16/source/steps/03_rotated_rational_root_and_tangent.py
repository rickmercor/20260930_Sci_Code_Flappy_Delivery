"""
Implement the rotated rational matrix-square-root iteration and its exact directional tangent using the algorithm, branch convention, stopping rule, and restrictions stated in the main problem.

Returns

-------

Return the tuple:

(G, dG, update_count, final_residual, rho_2, d_rho_2)

in exactly that order.

- G: complex128 NumPy array of shape (N, N), containing the branch-selected matrix square root.

- dG: complex128 NumPy array of shape (N, N), containing its directional derivative through the executed rational iterations.

- update_count: Python integer counting completed updates, excluding initialization.

- final_residual: Python float equal to the converged Frobenius residual ||I - Z @ Y||_F.

- rho_2: Python float equal to ||I - Z_2 @ Y_2||_F after exactly two completed updates, recorded before the third correction.

- d_rho_2: Python float equal to the directional derivative of rho_2 with respect to the supplied tangent direction.

A successful return satisfies:

- 0 <= update_count <= 50

- final_residual < 1e-13

If convergence has not occurred after 50 completed updates, raise RuntimeError instead of returning an unconverged result. A singular linear solve may raise LinAlgError.

The final_residual is the converged iteration residual ||I - Z @ Y||_F, while rho_2 and d_rho_2 are the residual norm and its directional derivative after exactly two completed updates. Do not return a derivative of final_residual or of update_count.

Apply the matrix ordering and differentiated-solve rule specified in the main problem directly to the supplied product and tangent.



The two-update residual diagnostics are recorded before performing the third correction. Continue from that same state until convergence, then apply the prescribed branch rotation. Do not substitute a different square-root method or alter the nonnormal inputs.

Returns
-------
Return (G, G_tangent, update_count, residual, rho2, drho2), in that order. G and G_tangent are complex128 arrays of shape (N,N), containing the selected root and its directional derivative through the executed rational updates. update_count is the integer number of completed updates at the first residual below 10^-13. residual is the final Frobenius norm ||I-ZY||_F. rho2 is the Frobenius norm ||I-Z_2Y_2||_F after exactly two completed updates; drho2 is its directional derivative induced by the supplied tangent. The two-update diagnostics are recorded before later continuation and do not replace the converged root outputs. A successful return has residual below 10^-13 and update_count at most50. Raise RuntimeError if convergence has not occurred after50 completed updates; a singular required solve may raise LinAlgError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rotated_root(
    product: np.ndarray,
    tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float]:
    """Evaluate the rotated rational root and its directional derivative.

    Parameters
    ----------
    product : np.ndarray
        Complex square matrix M. It may be nonnormal.
    tangent : np.ndarray
        Complex directional derivative dM with the same shape as product.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int, float, float, float]
        G, dG, completed update count, converged residual,
        residual after exactly two updates, and the directional derivative
        of that two-update residual.

    Raises
    ------
    RuntimeError
        If the rational iteration does not converge within 50 updates.
    np.linalg.LinAlgError
        If a required correction solve is singular.
    """
    return (None, None, None, None, None, None)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rotated_root(
    product: np.ndarray,
    tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float]:
    import numpy as np
    from scipy.linalg import solve

    def constant(value):
        array = np.asarray(value, dtype=np.complex128)
        return array, np.zeros_like(array)

    def add(left, right):
        return left[0] + right[0], left[1] + right[1]

    def multiply(left, right):
        return (
            left[0] @ right[0],
            left[1] @ right[0] + left[0] @ right[1],
        )

    def scale(factor, pair):
        return factor * pair[0], factor * pair[1]

    def solve_pair(left, right):
        value = solve(left[0], right[0])
        derivative = solve(
            left[0],
            right[1] - left[1] @ value,
        )
        return value, derivative

    identity = constant(
        np.eye(product.shape[0], dtype=np.complex128)
    )

    phi = -0.25 * np.pi
    root_pair = scale(
        np.exp(1j * phi),
        (product, tangent),
    )
    inverse_pair = identity

    rho_after_two = None
    rho_tangent_after_two = None

    for updates in range(51):
        iterate = multiply(inverse_pair, root_pair)

        residual = float(
            np.linalg.norm(identity[0] - iterate[0], ord="fro")
        )

        if updates == 2:
            residual_matrix = identity[0] - iterate[0]
            residual_tangent = -iterate[1]
            rho_after_two = float(
                np.linalg.norm(residual_matrix, ord="fro")
            )

            if rho_after_two == 0.0:
                raise RuntimeError(
                    "two-update residual norm is zero"
                )

            rho_tangent_after_two = float(
                np.real(
                    np.vdot(
                        residual_matrix,
                        residual_tangent,
                    )
                )
                / rho_after_two
            )

        if residual < 1e-13:
            if (
                rho_after_two is None
                or rho_tangent_after_two is None
            ):
                raise RuntimeError(
                    "iteration converged before two-update "
                    "diagnostics were available"
                )

            root, root_tangent = scale(
                np.exp(-0.5j * phi),
                root_pair,
            )

            return (
                root,
                root_tangent,
                updates,
                residual,
                rho_after_two,
                rho_tangent_after_two,
            )

        if updates == 50:
            break

        squared = multiply(iterate, iterate)
        cubed = multiply(squared, iterate)

        numerator = add(
            add(
                scale(7, identity),
                scale(35, iterate),
            ),
            add(
                scale(21, squared),
                cubed,
            ),
        )

        denominator = add(
            add(
                identity,
                scale(21, iterate),
            ),
            add(
                scale(35, squared),
                scale(7, cubed),
            ),
        )

        correction = solve_pair(denominator, numerator)
        root_pair = multiply(root_pair, correction)
        inverse_pair = multiply(correction, inverse_pair)

    raise RuntimeError(
        "rational square root did not converge in 50 updates"
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'direction = np.array([\n'
               '    [0.1, 0.02],\n'
               '    [0.02, -0.05],\n'
               '], dtype=np.complex128)\n'
               'def _checked_root_result(result):\n'
               '    assert 0.0 <= result[3] < 1e-13, "converged residual must be below 1e-13"\n'
               '    return result\n',
      'call': '_checked_root_result(rotated_root(np.diag([4.0, -9.0]).astype(np.complex128), '
              'direction))',
      'gold_call': '_checked_root_result(_oracle_rotated_root(np.diag([4.0, '
                   '-9.0]).astype(np.complex128), direction))'},
     {'setup': 'import numpy as np\n'
               'product = np.array([\n'
               '    [0.7, 0.1j],\n'
               '    [-0.1j, 0.9],\n'
               '], dtype=np.complex128)\n'
               'direction = np.array([\n'
               '    [0.2, 0.04],\n'
               '    [0.04, -0.1],\n'
               '], dtype=np.complex128)\n'
               'def _checked_root_result(result):\n'
               '    assert 0.0 <= result[3] < 1e-13, "converged residual must be below 1e-13"\n'
               '    return result\n',
      'call': '_checked_root_result(rotated_root(product, direction))',
      'gold_call': '_checked_root_result(_oracle_rotated_root(product, direction))'},
     {'setup': 'import numpy as np\n'
               'direction = np.array([\n'
               '    [0.1, 0.02],\n'
               '    [0.02, -0.05],\n'
               '], dtype=np.complex128)\n'
               'def _checked_root_result(result):\n'
               '    assert 0.0 <= result[3] < 1e-13, "converged residual must be below 1e-13"\n'
               '    return result\n',
      'call': '_checked_root_result(rotated_root(np.eye(2, dtype=np.complex128), direction))',
      'gold_call': '_checked_root_result(_oracle_rotated_root(np.eye(2, dtype=np.complex128), '
                   'direction))'},
     {'setup': 'import numpy as np\n'
               'product = np.array([\n'
               '    [0.8, 0.35 + 0.2j, 0.1],\n'
               '    [0.0, 1.1, -0.25j],\n'
               '    [0.0, 0.0, 0.6],\n'
               '], dtype=np.complex128)\n'
               'direction = np.array([\n'
               '    [0.12, -0.04 + 0.03j, 0.02j],\n'
               '    [0.07, -0.08, 0.05],\n'
               '    [0.01j, -0.02, 0.09],\n'
               '], dtype=np.complex128)\n'
               'def _checked_root_result(result):\n'
               '    assert 0.0 <= result[3] < 1e-13, "converged residual must be below 1e-13"\n'
               '    return result\n',
      'call': '_checked_root_result(rotated_root(product, direction))',
      'gold_call': '_checked_root_result(_oracle_rotated_root(product, direction))'}]
