"""
Assemble the global Mini mixed finite-element matrices from the mesh and local element operators.

The paper's mixed finite-element problem couples the displacement and pressure spaces through the elasticity, divergence, and pressure forms, producing the discrete saddle-point eigenproblem.

Returns
-------
dict, containing the assembled sparse matrices A, B, C, M
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_mixed_system(
    vertices: np.ndarray,
    triangles: np.ndarray,
    mu: float,
    lam: float,
) -> dict:
    """
    Assemble the global Mini mixed finite-element matrices.

    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertices.
    triangles : np.ndarray
        Triangle connectivity.
    mu : float
        Shear modulus.
    lam : float
        Lame parameter.

    Returns
    -------
    system : dict
        Global sparse mixed finite-element system containing the following
        required keys:

        - ``A`` : sparse matrix
            Global displacement stiffness matrix for the two-component
            Mini displacement space.
        - ``B`` : sparse matrix
            Global displacement-pressure divergence coupling matrix.
        - ``C`` : sparse matrix
            Global pressure matrix associated with the compressibility term.
        - ``M`` : sparse matrix
            Global displacement mass matrix.
        - ``space`` : dict
            The deterministic degree-of-freedom and element-connectivity maps
            returned by ``build_mini_space``.
    """
    return system

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_assemble_mixed_system(
    vertices: np.ndarray,
    triangles: np.ndarray,
    mu: float,
    lam: float,
) -> dict:
    """Reference implementation."""
    import scipy.sparse as sp

    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (N,2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (T,3)")
    if len(triangles) == 0:
        raise ValueError("triangles must be nonempty")
    if not np.isscalar(mu) or float(mu) <= 0:
        raise ValueError("mu must be positive")
    if not np.isscalar(lam) or float(lam) <= 0:
        raise ValueError("lam must be positive")

    space = _oracle_build_mini_space(vertices, triangles)

    n_v = int(space["vertex_count"][0])
    n_t = int(space["triangle_count"][0])
    n_s = int(space["scalar_displacement_count"][0])
    n_d = int(space["displacement_count"][0])
    n_q = n_v

    A_rows, A_cols, A_data = [], [], []
    M_rows, M_cols, M_data = [], [], []
    B_rows, B_cols, B_data = [], [], []
    C_rows, C_cols, C_data = [], [], []

    for e, tri in enumerate(triangles):
        loc = _oracle_local_matrices(vertices[tri], mu, lam)

        ax = space["element_x"][e]
        ay = space["element_y"][e]

        adofs = np.concatenate([ax, ay])

        Al = loc["A_local"]
        Ml = loc["M_local"]
        Bl = loc["B_local"]
        Cl = loc["C_local"]

        for i, gi in enumerate(adofs):
            for j, gj in enumerate(adofs):
                if Al[i, j] != 0.0:
                    A_rows.append(int(gi))
                    A_cols.append(int(gj))
                    A_data.append(float(Al[i, j]))
                if Ml[i, j] != 0.0:
                    M_rows.append(int(gi))
                    M_cols.append(int(gj))
                    M_data.append(float(Ml[i, j]))

        pdofs = triangles[e]

        for i, gi in enumerate(pdofs):
            for j, gj in enumerate(adofs):
                if Bl[i, j] != 0.0:
                    B_rows.append(int(gi))
                    B_cols.append(int(gj))
                    B_data.append(float(Bl[i, j]))

            for j, gj in enumerate(pdofs):
                if Cl[i, j] != 0.0:
                    C_rows.append(int(gi))
                    C_cols.append(int(gj))
                    C_data.append(float(Cl[i, j]))

    A = sp.coo_matrix(
        (A_data, (A_rows, A_cols)),
        shape=(n_d, n_d),
    ).tocsr()

    M = sp.coo_matrix(
        (M_data, (M_rows, M_cols)),
        shape=(n_d, n_d),
    ).tocsr()

    B = sp.coo_matrix(
        (B_data, (B_rows, B_cols)),
        shape=(n_q, n_d),
    ).tocsr()

    C = sp.coo_matrix(
        (C_data, (C_rows, C_cols)),
        shape=(n_q, n_q),
    ).tocsr()

    return {
        "A": A,
        "B": B,
        "C": C,
        "M": M,
        "space": space,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np

v = np.array([
    [0., 0.],
    [1., 0.],
    [1., 1.],
    [0., 1.]
])

t = np.array([
    [0, 1, 2],
    [0, 2, 3]
], dtype=int)

def pack_system(system):
    space = system["space"]
    return np.concatenate([
        np.asarray(system["A"].toarray(), dtype=float).ravel(),
        np.asarray(system["B"].toarray(), dtype=float).ravel(),
        np.asarray(system["C"].toarray(), dtype=float).ravel(),
        np.asarray(system["M"].toarray(), dtype=float).ravel(),
        np.asarray(space["vertex_count"], dtype=float).ravel(),
        np.asarray(space["triangle_count"], dtype=float).ravel(),
        np.asarray(space["scalar_displacement_count"], dtype=float).ravel(),
        np.asarray(space["displacement_count"], dtype=float).ravel(),
        np.asarray(space["pressure_count"], dtype=float).ravel(),
        np.asarray(space["element_scalar"], dtype=float).ravel(),
        np.asarray(space["element_x"], dtype=float).ravel(),
        np.asarray(space["element_y"], dtype=float).ravel(),
    ])
""",
            "call": "pack_system(assemble_mixed_system(v, t, 1.0, 7777.0))",
            "gold_call": "pack_system(_oracle_assemble_mixed_system(v, t, 1.0, 7777.0))",
        },
        {
            "setup": """import numpy as np

v = np.array([
    [0., 0.],
    [1., 0.],
    [0., 1.]
])

t = np.array([
    [0, 1, 2]
], dtype=int)

def pack_system(system):
    space = system["space"]
    return np.concatenate([
        np.asarray(system["A"].toarray(), dtype=float).ravel(),
        np.asarray(system["B"].toarray(), dtype=float).ravel(),
        np.asarray(system["C"].toarray(), dtype=float).ravel(),
        np.asarray(system["M"].toarray(), dtype=float).ravel(),
        np.asarray(space["vertex_count"], dtype=float).ravel(),
        np.asarray(space["triangle_count"], dtype=float).ravel(),
        np.asarray(space["scalar_displacement_count"], dtype=float).ravel(),
        np.asarray(space["displacement_count"], dtype=float).ravel(),
        np.asarray(space["pressure_count"], dtype=float).ravel(),
        np.asarray(space["element_scalar"], dtype=float).ravel(),
        np.asarray(space["element_x"], dtype=float).ravel(),
        np.asarray(space["element_y"], dtype=float).ravel(),
    ])
""",
            "call": "pack_system(assemble_mixed_system(v, t, 1.0, 1.0))",
            "gold_call": "pack_system(_oracle_assemble_mixed_system(v, t, 1.0, 1.0))",
        },
        {
            "setup": """import numpy as np

v = np.array([
    [0., 0.],
    [1., 0.],
    [1e-6, 2e-6]
])

t = np.array([
    [0, 1, 2]
], dtype=int)

def pack_system(system):
    space = system["space"]
    return np.concatenate([
        np.asarray(system["A"].toarray(), dtype=float).ravel(),
        np.asarray(system["B"].toarray(), dtype=float).ravel(),
        np.asarray(system["C"].toarray(), dtype=float).ravel(),
        np.asarray(system["M"].toarray(), dtype=float).ravel(),
        np.asarray(space["vertex_count"], dtype=float).ravel(),
        np.asarray(space["triangle_count"], dtype=float).ravel(),
        np.asarray(space["scalar_displacement_count"], dtype=float).ravel(),
        np.asarray(space["displacement_count"], dtype=float).ravel(),
        np.asarray(space["pressure_count"], dtype=float).ravel(),
        np.asarray(space["element_scalar"], dtype=float).ravel(),
        np.asarray(space["element_x"], dtype=float).ravel(),
        np.asarray(space["element_y"], dtype=float).ravel(),
    ])
""",
            "call": "pack_system(assemble_mixed_system(v, t, 2.0, 100.0))",
            "gold_call": "pack_system(_oracle_assemble_mixed_system(v, t, 2.0, 100.0))",
        },
    ]
