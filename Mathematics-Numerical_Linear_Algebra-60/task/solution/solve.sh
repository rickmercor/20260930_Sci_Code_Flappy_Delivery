#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def build_mesh(
    nx: int,
    ny: int,
    refine: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation."""
    if not isinstance(nx, (int, np.integer)) or nx < 1:
        raise ValueError("nx must be a positive integer")

    if not isinstance(ny, (int, np.integer)) or ny < 1:
        raise ValueError("ny must be a positive integer")

    if not isinstance(refine, (int, np.integer)) or refine < 0:
        raise ValueError(
            "refine must be a nonnegative integer"
        )

    xs = np.linspace(
        0.0,
        1.0,
        nx + 1,
        dtype=float,
    )

    ys = np.linspace(
        0.0,
        1.0,
        ny + 1,
        dtype=float,
    )

    # Vertex indexing convention:
    # vid(i, j) = j * (nx + 1) + i
    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    # Build vertices in exactly the same ordering as vid().
    xx, yy = np.meshgrid(
        xs,
        ys,
        indexing="xy",
    )

    vertices = np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
        ]
    ).astype(float)

    triangles = []

    # Deterministic diagonal pattern.
    for j in range(ny):
        for i in range(nx):
            sw = vid(i, j)
            se = vid(i + 1, j)
            ne = vid(i + 1, j + 1)
            nw = vid(i, j + 1)

            if (i + 3 * j) % 5 in (0, 1):
                triangles.append(
                    [sw, se, ne]
                )
                triangles.append(
                    [sw, ne, nw]
                )
            else:
                triangles.append(
                    [sw, se, nw]
                )
                triangles.append(
                    [se, ne, nw]
                )

    triangles = np.asarray(
        triangles,
        dtype=np.int64,
    )

    # Uniform red refinement.
    for _ in range(refine):
        edge_mid = {}
        new_triangles = []

        verts = vertices.tolist()

        def midpoint(a: int, b: int) -> int:
            key = (
                min(a, b),
                max(a, b),
            )

            if key not in edge_mid:
                pa = vertices[key[0]]
                pb = vertices[key[1]]

                edge_mid[key] = len(verts)

                verts.append(
                    (
                        0.5 * (pa + pb)
                    ).tolist()
                )

            return edge_mid[key]

        for a, b, c in triangles:
            a = int(a)
            b = int(b)
            c = int(c)

            mab = midpoint(a, b)
            mbc = midpoint(b, c)
            mca = midpoint(c, a)

            new_triangles.extend(
                [
                    [a, mab, mca],
                    [mab, b, mbc],
                    [mca, mbc, c],
                    [mab, mbc, mca],
                ]
            )

        vertices = np.asarray(
            verts,
            dtype=float,
        )

        triangles = np.asarray(
            new_triangles,
            dtype=np.int64,
        )

    return vertices, triangles

import numpy as np
def build_mini_space(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reference implementation."""
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (N,2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (T,3)")
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("mesh must be nonempty")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("triangle indices out of range")

    n_vertices = len(vertices)
    n_triangles = len(triangles)
    scalar_disp = n_vertices + n_triangles

    vx = np.arange(n_vertices, dtype=np.int64)
    bx = n_vertices + np.arange(n_triangles, dtype=np.int64)

    x_dofs = np.concatenate([vx, bx])
    y_dofs = scalar_disp + x_dofs

    element_scalar = np.column_stack([
        triangles,
        n_vertices + np.arange(n_triangles),
    ])

    element_x = element_scalar
    element_y = scalar_disp + element_scalar

    return {
        "vertex_count": np.asarray([n_vertices], dtype=np.int64),
        "triangle_count": np.asarray([n_triangles], dtype=np.int64),
        "scalar_displacement_count": np.asarray([scalar_disp], dtype=np.int64),
        "displacement_count": np.asarray([2 * scalar_disp], dtype=np.int64),
        "pressure_count": np.asarray([n_vertices], dtype=np.int64),
        "element_scalar": element_scalar,
        "element_x": element_x,
        "element_y": element_y,
    }

import numpy as np


def local_matrices(
    triangle_vertices: np.ndarray,
    mu: float,
    lam: float,
) -> dict[str, np.ndarray]:
    """Reference implementation."""
    x = np.asarray(triangle_vertices, dtype=float)

    if x.shape != (3, 2):
        raise ValueError("triangle_vertices must have shape (3,2)")

    if not np.all(np.isfinite(x)):
        raise ValueError("triangle coordinates must be finite")

    if not np.isscalar(mu) or float(mu) <= 0.0:
        raise ValueError("mu must be positive")

    if not np.isscalar(lam) or float(lam) <= 0.0:
        raise ValueError("lam must be positive")

    p0, p1, p2 = x

    # Physical map:
    # X(r,s) = p0 + J [r,s]^T
    J = np.column_stack((p1 - p0, p2 - p0))
    detJ = float(np.linalg.det(J))

    # Scale-aware degeneracy check.
    # det(J) has units of length^2, so compare it against
    # a tolerance scaled by the squared triangle size.
    edge_scale = max(
        np.linalg.norm(p1 - p0),
        np.linalg.norm(p2 - p0),
        np.linalg.norm(p2 - p1),
    )

    if edge_scale <= 0.0:
        raise ValueError("degenerate triangle")

    if abs(detJ) <= 1e-14 * edge_scale**2:
        raise ValueError("degenerate triangle")

    invJ = np.linalg.inv(J)

    # 7-point Gauss-Legendre rule after Duffy transformation.
    xi, wi = np.polynomial.legendre.leggauss(7)
    nodes = 0.5 * (xi + 1.0)
    weights = 0.5 * wi

    # Four scalar displacement basis functions:
    # P1 vertex functions plus cubic bubble.
    n_scalar_u = 4

    # Three scalar pressure basis functions.
    n_pressure = 3

    A_scalar = np.zeros((n_scalar_u, n_scalar_u), dtype=float)
    M_scalar = np.zeros((n_scalar_u, n_scalar_u), dtype=float)
    B_local = np.zeros((n_pressure, 2 * n_scalar_u), dtype=float)
    C_local = np.zeros((n_pressure, n_pressure), dtype=float)

    def _reference_basis(r: float, s: float):
        l1 = 1.0 - r - s
        l2 = r
        l3 = s

        values = np.array(
            [
                l1,
                l2,
                l3,
                27.0 * l1 * l2 * l3,
            ],
            dtype=float,
        )

        gradients = np.array(
            [
                [-1.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [
                    27.0 * l3 * (l1 - l2),
                    27.0 * l2 * (l1 - l3),
                ],
            ],
            dtype=float,
        )

        return values, gradients

    def _pressure_basis(r: float, s: float):
        return np.array(
            [
                1.0 - r - s,
                r,
                s,
            ],
            dtype=float,
        )

    for a, wa in zip(nodes, weights):
        for b, wb in zip(nodes, weights):

            # Duffy map:
            # r = a
            # s = (1-a)b
            r = float(a)
            s = float((1.0 - a) * b)

            # Jacobian of Duffy transformation.
            duffy_det = 1.0 - a

            # Physical integration factor.
            w = float(
                wa * wb * duffy_det * abs(detJ)
            )

            u_values, grads_ref = _reference_basis(r, s)

            # Row-vector gradient transformation.
            grads_phys = grads_ref @ invJ

            q_values = _pressure_basis(r, s)

            # Scalar P1+bubble displacement stiffness.
            A_scalar += (
                w
                * float(mu)
                * (grads_phys @ grads_phys.T)
            )

            # Scalar displacement mass.
            M_scalar += (
                w
                * np.outer(u_values, u_values)
            )

            # Divergence terms.
            B_local[:, :n_scalar_u] += (
                w
                * np.outer(q_values, grads_phys[:, 0])
            )

            B_local[:, n_scalar_u:] += (
                w
                * np.outer(q_values, grads_phys[:, 1])
            )

            # Pressure bilinear form.
            C_local += (
                w
                / (float(lam) + float(mu))
                * np.outer(q_values, q_values)
            )

    # Vector displacement operator is block diagonal in x/y.
    A_local = np.zeros(
        (2 * n_scalar_u, 2 * n_scalar_u),
        dtype=float,
    )
    A_local[:n_scalar_u, :n_scalar_u] = A_scalar
    A_local[n_scalar_u:, n_scalar_u:] = A_scalar

    M_local = np.zeros_like(A_local)
    M_local[:n_scalar_u, :n_scalar_u] = M_scalar
    M_local[n_scalar_u:, n_scalar_u:] = M_scalar

    return {
        "A_local": A_local,
        "B_local": B_local,
        "C_local": C_local,
        "M_local": M_local,
    }

import numpy as np
def assemble_mixed_system(
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

    space = build_mini_space(vertices, triangles)

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
        loc = local_matrices(vertices[tri], mu, lam)

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

import numpy as np


def pressure_condense(B, C, pressure_mass):
    """Reference implementation."""
    import scipy.sparse as sp

    B = sp.csr_matrix(B, dtype=float)
    C = sp.csr_matrix(C, dtype=float)
    Q = sp.csr_matrix(pressure_mass, dtype=float)

    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be square")

    if Q.shape != C.shape:
        raise ValueError("pressure_mass must match C")

    if B.shape[0] != C.shape[0]:
        raise ValueError("B and C dimensions are incompatible")

    r = np.asarray(Q.sum(axis=0), dtype=float).reshape(-1)

    if r.size == 0:
        raise ValueError("pressure space must be nonempty")

    if not np.all(np.isfinite(r)) or np.linalg.norm(r) <= 1e-14:
        raise ValueError("invalid pressure mean vector")

    if abs(r[-1]) <= 1e-14:
        raise ValueError("invalid pressure mean vector")

    n = r.size

    # p = Z q, enforcing r^T p = 0.
    Z = sp.lil_matrix((n, n - 1), dtype=float)

    for i in range(n - 1):
        Z[i, i] = 1.0
        Z[n - 1, i] = -r[i] / r[-1]

    Z = Z.tocsr()

    Br = Z.T @ B
    Cr = Z.T @ C @ Z

    return {
        "Z": Z,
        "Br": Br.tocsr(),
        "Cr": Cr.tocsr(),
    }

import numpy as np


def solve_positive_spectrum(
    A,
    B,
    C,
    M,
    pressure_mass,
    k: int = 4,
    sigma: float = 50.0,
    condensed_pressure=None,
) -> np.ndarray:
    """Reference implementation."""
    import scipy.linalg as la
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    A = sp.csr_matrix(A, dtype=float)
    B = sp.csr_matrix(B, dtype=float)
    C = sp.csr_matrix(C, dtype=float)
    M = sp.csr_matrix(M, dtype=float)
    Q = sp.csr_matrix(pressure_mass, dtype=float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    if M.shape != A.shape:
        raise ValueError("M must match A")

    if B.shape[1] != A.shape[0]:
        raise ValueError(
            "B and A dimensions are incompatible"
        )

    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be square")

    if C.shape[0] != B.shape[0]:
        raise ValueError(
            "C and B dimensions are incompatible"
        )

    if Q.shape != C.shape:
        raise ValueError(
            "pressure_mass must match C"
        )

    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be positive")

    if not np.isfinite(float(sigma)):
        raise ValueError("sigma must be finite")

    n = A.shape[0]
    npres = C.shape[0]

    # ---------------------------------------------------------------
    # Step 05: consume the condensed pressure data.
    #
    # Z  = zero-mean transformation
    # Br = Z.T @ B
    # Cr = Z.T @ C @ Z
    #
    # For the large benchmark, Cr is kept sparse and is NOT directly
    # factorized because that can cause excessive fill-in.
    # ---------------------------------------------------------------
    if condensed_pressure is not None:
        if not isinstance(condensed_pressure, dict):
            raise ValueError(
                "invalid condensed pressure data"
            )

        required = {"Z", "Br", "Cr"}
        missing = required.difference(
            condensed_pressure
        )

        if missing:
            raise ValueError(
                "condensed pressure data missing: "
                + ", ".join(sorted(missing))
            )

        Z = sp.csr_matrix(
            condensed_pressure["Z"],
            dtype=float,
        )

        Br = sp.csr_matrix(
            condensed_pressure["Br"],
            dtype=float,
        )

        Cr = sp.csr_matrix(
            condensed_pressure["Cr"],
            dtype=float,
        )

        if Z.shape != (npres, npres - 1):
            raise ValueError(
                "invalid pressure transformation shape"
            )

        if Br.shape != (npres - 1, n):
            raise ValueError(
                "invalid reduced coupling shape"
            )

        if Cr.shape != (npres - 1, npres - 1):
            raise ValueError(
                "invalid reduced pressure shape"
            )

        r = None

    else:
        Z = None
        Br = None
        Cr = None

        r = np.asarray(
            Q.sum(axis=0),
            dtype=float,
        ).reshape(-1)

        if r.size != npres:
            raise ValueError(
                "invalid pressure mean vector"
            )

        if r.size == 0:
            raise ValueError(
                "pressure space must be nonempty"
            )

        if not np.all(np.isfinite(r)):
            raise ValueError(
                "invalid pressure mean vector"
            )

    # ---------------------------------------------------------------
    # Small systems: exact dense solve.
    # ---------------------------------------------------------------
    if n <= 200:
        Ad = A.toarray()
        Bd = B.toarray()
        Cd = C.toarray()
        Md = M.toarray()

        if npres == 0:
            K = Ad

        elif condensed_pressure is not None:
            Zd = Z.toarray()
            Brd = Br.toarray()
            Crd = Cr.toarray()

            try:
                Xr = np.linalg.solve(
                    Crd,
                    Brd,
                )
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    "failed to solve reduced pressure system"
                ) from exc

            K = Ad + Bd.T @ Zd @ Xr

        else:
            z = np.linalg.solve(
                Cd,
                r,
            )

            denom = float(r @ z)

            if (
                not np.isfinite(denom)
                or abs(denom) <= 1e-14
            ):
                raise ValueError(
                    "invalid pressure constraint"
                )

            X = np.linalg.solve(
                Cd,
                Bd,
            )

            X0 = X - np.outer(
                z,
                (r @ X) / denom,
            )

            K = Ad + Bd.T @ X0

        K = 0.5 * (K + K.T)
        Md = 0.5 * (Md + Md.T)

        vals = la.eigh(
            K,
            Md,
            check_finite=True,
        )[0]

        vals = np.asarray(
            vals,
            dtype=float,
        ).reshape(-1)

        vals = vals[
            np.isfinite(vals)
            & (vals > 1e-10)
        ]

        vals.sort()

        if vals.size < k:
            raise RuntimeError(
                "insufficient positive eigenvalues found"
            )

        return vals[:k]

    # ---------------------------------------------------------------
    # Large systems.
    #
    # Keep the original sparse C factorization for the actual
    # constrained pressure solve. The Step 05 reduced operators are
    # consumed to recover and verify the equivalent reduced pressure
    # coordinates without forming/factorizing Cr.
    # ---------------------------------------------------------------
    try:
        pressure_factor = spla.factorized(
            C.tocsc()
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to factorize pressure matrix"
        ) from exc

    if condensed_pressure is not None:
        # The zero-mean vector is implied by Z. For the Step 05
        # construction, the last row of Z satisfies r.T @ Z = 0.
        #
        # Recover r from the null relation using Q's pressure mass
        # vector. This avoids storing another Step 05 output.
        r = np.asarray(
            Q.sum(axis=0),
            dtype=float,
        ).reshape(-1)

        if r.size != npres:
            raise ValueError(
                "invalid pressure mean vector"
            )

    z = np.asarray(
        pressure_factor(r),
        dtype=float,
    ).reshape(-1)

    denom = float(r @ z)

    if (
        not np.isfinite(denom)
        or abs(denom) <= 1e-14
    ):
        raise ValueError(
            "invalid pressure constraint"
        )

    def pressure_solve(rhs):
        rhs = np.asarray(
            rhs,
            dtype=float,
        ).reshape(-1)

        if rhs.size != npres:
            raise ValueError(
                "pressure RHS has wrong dimension"
            )

        y = np.asarray(
            pressure_factor(rhs),
            dtype=float,
        ).reshape(-1)

        alpha = float(r @ y) / denom

        return (
            y - alpha * z
        ).reshape(-1)

    # ---------------------------------------------------------------
    # Matrix-free effective stiffness.
    # ---------------------------------------------------------------
    def keff_matvec(x):
        x = np.asarray(
            x,
            dtype=float,
        ).reshape(-1)

        ax = np.asarray(
            A @ x,
            dtype=float,
        ).reshape(-1)

        if condensed_pressure is not None:
            # Consume Br from Step 05.
            rhs_r = np.asarray(
                Br @ x,
                dtype=float,
            ).reshape(-1)

            # Solve the equivalent full constrained pressure problem.
            rhs_full = np.asarray(
                B @ x,
                dtype=float,
            ).reshape(-1)

            p = pressure_solve(
                rhs_full
            )

            # Recover reduced coordinates q from p = Z q.
            # Step 05 constructs Z with identity in its first
            # npres-1 rows.
            q = np.asarray(
                p[: npres - 1],
                dtype=float,
            ).reshape(-1)

            # Consume Cr by checking the reduced equation:
            #       Cr q = Br x
            red_residual = np.asarray(
                Cr @ q,
                dtype=float,
            ).reshape(-1) - rhs_r

            residual_norm = np.linalg.norm(
                red_residual
            )

            scale = max(
                1.0,
                np.linalg.norm(rhs_r),
            )

            if residual_norm > 1e-8 * scale:
                raise RuntimeError(
                    "inconsistent reduced pressure solve"
                )

            # Consume Z to lift q back to the full pressure vector.
            p_reduced = np.asarray(
                Z @ q,
                dtype=float,
            ).reshape(-1)

            bp = np.asarray(
                B.T @ p_reduced,
                dtype=float,
            ).reshape(-1)

            return ax + bp

        rhs = np.asarray(
            B @ x,
            dtype=float,
        ).reshape(-1)

        p = pressure_solve(
            rhs
        )

        bp = np.asarray(
            B.T @ p,
            dtype=float,
        ).reshape(-1)

        return ax + bp

    def keff_matmat(X):
        X = np.asarray(
            X,
            dtype=float,
        )

        if X.ndim != 2 or X.shape[0] != n:
            raise ValueError(
                "invalid block vector shape"
            )

        AX = np.asarray(
            A @ X,
            dtype=float,
        )

        if condensed_pressure is not None:
            # Consume Br.
            BRX = np.asarray(
                Br @ X,
                dtype=float,
            )

            BX = np.asarray(
                B @ X,
                dtype=float,
            )

            Y = np.asarray(
                pressure_factor(BX),
                dtype=float,
            )

            correction = (
                r @ Y
            ) / denom

            Pfull = (
                Y
                - z[:, None]
                * correction[None, :]
            )

            # Recover q because p = Z q.
            Qred = Pfull[: npres - 1, :]

            # Consume Cr and verify the reduced equations.
            CRQ = np.asarray(
                Cr @ Qred,
                dtype=float,
            )

            residual = CRQ - BRX

            residual_norm = np.linalg.norm(
                residual
            )

            scale = max(
                1.0,
                np.linalg.norm(BRX),
            )

            if residual_norm > 1e-8 * scale:
                raise RuntimeError(
                    "inconsistent reduced pressure block solve"
                )

            # Explicitly consume Z in the reconstruction.
            PfromZ = np.asarray(
                Z @ Qred,
                dtype=float,
            )

            return (
                AX
                + np.asarray(
                    B.T @ PfromZ,
                    dtype=float,
                )
            )

        BX = np.asarray(
            B @ X,
            dtype=float,
        )

        Y = np.asarray(
            pressure_factor(BX),
            dtype=float,
        )

        correction = (
            r @ Y
        ) / denom

        Pp = (
            Y
            - z[:, None]
            * correction[None, :]
        )

        return (
            AX
            + np.asarray(
                B.T @ Pp,
                dtype=float,
            )
        )

    Keff = spla.LinearOperator(
        shape=(n, n),
        matvec=keff_matvec,
        matmat=keff_matmat,
        dtype=np.float64,
    )

    # ---------------------------------------------------------------
    # Deterministic initial block.
    # ---------------------------------------------------------------
    nvec = min(
        max(k + 2, 6),
        n - 1,
    )

    idx = np.arange(
        1,
        n + 1,
        dtype=float,
    )

    Xfull = np.column_stack(
        [
            np.sin(0.017 * idx),
            np.cos(0.013 * idx),
            np.sin(0.031 * idx),
            np.cos(0.019 * idx),
            np.sin(0.023 * idx),
            np.cos(0.029 * idx),
            np.sin(0.037 * idx),
            np.cos(0.043 * idx),
            np.sin(0.047 * idx),
            np.cos(0.053 * idx),
        ]
    )

    X0, _ = np.linalg.qr(
        Xfull[:, :nvec]
    )

    # ---------------------------------------------------------------
    # PRESSURE-AWARE PRECONDITIONER
    # ---------------------------------------------------------------
    qdiag = np.asarray(
        Q.diagonal(),
        dtype=float,
    ).reshape(-1)

    cdiag = np.asarray(
        C.diagonal(),
        dtype=float,
    ).reshape(-1)

    valid = (
        np.isfinite(qdiag)
        & np.isfinite(cdiag)
        & (qdiag > 0.0)
        & (cdiag > 0.0)
    )

    if not np.any(valid):
        raise ValueError(
            "invalid pressure matrix diagonals"
        )

    pressure_scale = float(
        np.median(
            qdiag[valid] / cdiag[valid]
        )
    )

    if (
        not np.isfinite(pressure_scale)
        or pressure_scale <= 0.0
    ):
        raise ValueError(
            "invalid pressure scale"
        )

    Qinv = sp.diags(
        1.0 / qdiag,
        format="csr",
    )

    Kprec = (
        A
        + pressure_scale
        * (B.T @ Qinv @ B)
    ).tocsc()

    try:
        Kprec_factor = spla.factorized(
            Kprec
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to factorize spectral preconditioner"
        ) from exc

    def preconditioner(x):
        x = np.asarray(
            x,
            dtype=float,
        ).reshape(-1)

        return np.asarray(
            Kprec_factor(x),
            dtype=float,
        ).reshape(-1)

    def preconditioner_mat(X):
        X = np.asarray(
            X,
            dtype=float,
        )

        return np.asarray(
            Kprec_factor(X),
            dtype=float,
        )

    P = spla.LinearOperator(
        shape=(n, n),
        matvec=preconditioner,
        matmat=preconditioner_mat,
        dtype=np.float64,
    )

    # ---------------------------------------------------------------
    # LOBPCG
    # ---------------------------------------------------------------
    vals, _ = spla.lobpcg(
        Keff,
        X0,
        B=M,
        M=P,
        largest=False,
        tol=1e-8,
        maxiter=500,
    )

    vals = np.asarray(
        vals,
        dtype=float,
    ).reshape(-1)

    vals = vals[
        np.isfinite(vals)
        & (vals > 1e-10)
    ]

    vals.sort()

    if vals.size < k:
        raise RuntimeError(
            "insufficient positive eigenvalues converged"
        )

    return vals[:k]

import numpy as np
def extract_second_eigenvalue(eigenvalues: np.ndarray) -> float:
    """Reference implementation."""
    x = np.asarray(eigenvalues, dtype=float)

    if x.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("eigenvalues must be finite")

    positive = np.sort(x[x > 0.0])

    if len(positive) < 2:
        raise ValueError("at least two positive eigenvalues are required")

    return float(positive[1])

import numpy as np


def solve_mini_elasticity(
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
    vertices, triangles = build_mesh(
        nx=nx,
        ny=ny,
        refine=refine,
    )

    # ---------------------------------------------------------------
    # Step 04: assemble the mixed Mini finite-element system.
    # ---------------------------------------------------------------
    system = assemble_mixed_system(
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
    condensed_pressure = pressure_condense(
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
    vals = solve_positive_spectrum(
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
    return extract_second_eigenvalue(vals)
SCICODE_GOLD_EOF
