"""
Compute the lowest positive eigenvalues of the condensed Mini mixed finite-element problem.

The paper orders the discrete eigenvalues increasingly as positive values and normalizes the corresponding displacement eigenfunctions in the L2L^2 mass inner product. The condensed formulation is spectrally equivalent to the mixed problem because the pressure equation determines the pressure variable for a given displacement.

Returns
-------
np.ndarray of shape (k,), containing increasing positive eigenvalues
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    """
    Compute the k smallest positive eigenvalues of the constrained
    mixed displacement-pressure eigenproblem.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Global displacement stiffness matrix.
    B : scipy.sparse.spmatrix
        Global divergence coupling matrix.
    C : scipy.sparse.spmatrix
        Global pressure bilinear-form matrix from the mixed formulation.
    M : scipy.sparse.spmatrix
        Global displacement L2 mass matrix.
    pressure_mass : scipy.sparse.spmatrix
        Continuous P1 pressure mass matrix.
    k : int
        Number of smallest positive eigenvalues to return.
    sigma : float
        Spectral shift used by the eigensolver.
    condensed_pressure : dict[str, scipy.sparse.spmatrix] or None
        Optional zero-mean pressure reduction produced by the pressure
        condensation step. When provided, it contains exactly the entries

            Z  : zero-mean pressure transformation matrix, built by
                 eliminating the last pressure degree of freedom, so its
                 leading n_q - 1 rows are the identity,
            Br : reduced divergence coupling, Br = Z.T @ B,
            Cr : reduced pressure matrix, Cr = Z.T @ C @ Z.

        The matrices define the pressure operator on the zero-mean pressure
        subspace and may be used to solve the constrained eigenproblem
        without reconstructing the reduction.

    Returns
    -------
    eigenvalues : np.ndarray
        Increasing array containing the k smallest positive discrete
        eigenvalues.

    Raises
    ------
    ValueError
        If the matrix dimensions are incompatible, if k is not positive,
        or if sigma is not finite.
    """
    return eigenvalues

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_positive_spectrum(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
import scipy.sparse as sp

A = sp.eye(2, format='csr')
M = sp.eye(2, format='csr')
B = sp.csr_matrix([[1., -1.]])
C = sp.csr_matrix([[1.]])
Q = sp.csr_matrix([[1.]])""",
            "call": (
                "solve_positive_spectrum("
                "A, B, C, M, Q, k=1)"
            ),
            "gold_call": (
                "_oracle_solve_positive_spectrum("
                "A, B, C, M, Q, k=1)"
            ),
        },
        {
            "setup": """import numpy as np
import scipy.sparse as sp

A = sp.diags(
    [2., 3., 5.],
    format='csr'
)
M = sp.eye(3, format='csr')
B = sp.csr_matrix([[1., 0., -1.]])
C = sp.csr_matrix([[2.]])
Q = sp.csr_matrix([[1.]])""",
            "call": (
                "solve_positive_spectrum("
                "A, B, C, M, Q, k=1)"
            ),
            "gold_call": (
                "_oracle_solve_positive_spectrum("
                "A, B, C, M, Q, k=1)"
            ),
        },
        {
            "setup": """import numpy as np
import scipy.sparse as sp

A = sp.eye(4, format='csr')
M = sp.eye(4, format='csr')
B = sp.csr_matrix([
    [1., -1., 0., 0.],
    [0., 0., 1., -1.]
])
C = 2.0 * sp.eye(2, format='csr')
Q = sp.eye(2, format='csr')""",
            "call": (
                "solve_positive_spectrum("
                "A, B, C, M, Q, k=2)"
            ),
            "gold_call": (
                "_oracle_solve_positive_spectrum("
                "A, B, C, M, Q, k=2)"
            ),
        },
    ]
