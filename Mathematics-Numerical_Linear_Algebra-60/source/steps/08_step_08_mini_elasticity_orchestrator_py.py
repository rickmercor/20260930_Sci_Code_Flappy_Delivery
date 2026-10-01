"""
End-to-end deterministic computation of the requested Mini mixed finite element eigenvalue on the benchmark-specific mesh and material parameters.

The paper's eigenproblem uses the Mini mixed finite-element spaces together with the mixed elasticity bilinear structure and displacement $L^2$ mass normalization. The orchestrator combines those components into the deterministic numerical instance specified by the benchmark.

Returns
-------
float, the second-smallest positive discrete eigenvalue
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def solve_mini_elasticity(
    nx: int = 48,
    ny: int = 48,
    refine: int = 1,
    mu: float = 1.0,
    lam: float = 7777.0,
) -> float:
    """
    Compute the benchmark Mini mixed finite-element eigenvalue.

    Parameters
    ----------
    nx : int
        Initial number of Cartesian cells in x.
    ny : int
        Initial number of Cartesian cells in y.
    refine : int
        Number of uniform refinements.
    mu : float
        Shear modulus.
    lam : float
        Lame parameter.

    Returns
    -------
    result : float
        Second-smallest positive discrete eigenvalue.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_mini_elasticity(
    nx: int = 48,
    ny: int = 48,
    refine: int = 1,
    mu: float = 1.0,
    lam: float = 7777.0,
) -> float:
    """Reference end-to-end implementation."""
    import scipy.sparse as sp

    # ---------------------------------------------------------------
    # Step 01: deterministic mesh construction.
    # ---------------------------------------------------------------
    vertices, triangles = _oracle_build_mesh(
        nx=nx,
        ny=ny,
        refine=refine,
    )

    # ---------------------------------------------------------------
    # Step 04: assemble the mixed Mini finite-element system.
    # ---------------------------------------------------------------
    system = _oracle_assemble_mixed_system(
        vertices,
        triangles,
        mu=mu,
        lam=lam,
    )

    A = system["A"]
    B = system["B"]
    C = system["C"]
    M = system["M"]

    # ---------------------------------------------------------------
    # Apply homogeneous Dirichlet boundary conditions on the
    # displacement space before computing the spectrum.
    # ---------------------------------------------------------------
    tol = 1e-12

    on_boundary = (
        (np.abs(vertices[:, 0]) < tol)
        | (np.abs(vertices[:, 0] - 1.0) < tol)
        | (np.abs(vertices[:, 1]) < tol)
        | (np.abs(vertices[:, 1] - 1.0) < tol)
    )

    boundary_scalar_dofs = np.where(on_boundary)[0]

    scalar_disp = int(
        system["space"]["scalar_displacement_count"][0]
    )

    fixed_dofs = np.concatenate(
        [
            boundary_scalar_dofs,
            scalar_disp + boundary_scalar_dofs,
        ]
    )

    free_mask = np.ones(
        A.shape[0],
        dtype=bool,
    )

    free_mask[fixed_dofs] = False
    free_dofs = np.where(free_mask)[0]

    A = A[
        free_dofs
    ][:, free_dofs].tocsr()

    M = M[
        free_dofs
    ][:, free_dofs].tocsr()

    B = B[
        :,
        free_dofs,
    ].tocsr()

    # ---------------------------------------------------------------
    # Construct the continuous P1 pressure mass matrix used for the
    # zero-mean pressure constraint.
    # ---------------------------------------------------------------
    n_q = system["B"].shape[0]

    rows = []
    cols = []
    data = []

    for tri in triangles:
        p = vertices[tri]

        J = np.column_stack(
            (
                p[1] - p[0],
                p[2] - p[0],
            )
        )

        detJ = abs(np.linalg.det(J))

        if detJ <= 1e-14:
            raise ValueError(
                "degenerate triangle"
            )

        area = 0.5 * detJ

        local = (area / 12.0) * np.array(
            [
                [2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0],
            ]
        )

        for i, gi in enumerate(tri):
            for j, gj in enumerate(tri):
                rows.append(int(gi))
                cols.append(int(gj))
                data.append(float(local[i, j]))

    Q = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_q, n_q),
    ).tocsr()

    # ---------------------------------------------------------------
    # Step 05: pressure condensation.
    #
    # The returned Z, Br and Cr are passed directly into Step 06.
    # No separately reconstructed mean_vector is attached here.
    # ---------------------------------------------------------------
    condensed_pressure = _oracle_pressure_condense(
        B,
        C,
        Q,
    )

    # ---------------------------------------------------------------
    # Step 06: solve for the positive discrete spectrum.
    #
    # Step 06 consumes the reduced pressure operators returned by
    # Step 05.
    # ---------------------------------------------------------------
    vals = _oracle_solve_positive_spectrum(
        A,
        B,
        C,
        M,
        Q,
        k=4,
        condensed_pressure=condensed_pressure,
    )

    # ---------------------------------------------------------------
    # Step 07: extract the second positive eigenvalue.
    # ---------------------------------------------------------------
    return _oracle_extract_second_eigenvalue(vals)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np""",
            "call": "solve_mini_elasticity(2, 2, 1, 1.0, 10.0)",
            "gold_call": "_oracle_solve_mini_elasticity(2, 2, 1, 1.0, 10.0)",
        },
        {
            "setup": """import numpy as np""",
            "call": "solve_mini_elasticity(1, 1, 0, 1.0, 1.0)",
            "gold_call": "_oracle_solve_mini_elasticity(1, 1, 0, 1.0, 1.0)",
        },
        {
            "setup": """import numpy as np""",
            "call": "solve_mini_elasticity(2, 1, 1, 2.0, 100.0)",
            "gold_call": "_oracle_solve_mini_elasticity(2, 1, 1, 2.0, 100.0)",
        },
    ]
