"""
For each second-axis wavenumber, solve (I+bj*X_column)@Wj[:,column]=U[:,column], where X_column=M_tilde-I-diag((kx**2+ky[column]**2)/k0**2) and M_tilde is the positive index-squared multiplication matrix from Step 2. Return the complex auxiliary field and one reusable factor per column. Reuse a supplied cache only when the pole, grid, and material are unchanged; the right-hand side may change. Use a new cache when any operator input changes.

Removing exp(i*k0*z) from the forward Helmholtz field gives X_column=M_tilde-I-D_column, with D_column=diag((kx**2+ky[column]**2)/k0**2). Thus I+bj*X_column=(1-bj)*I-bj*D_column+bj*M_tilde. In a homogeneous medium, this agrees with the dispersion sqrt(n**2-k_perp**2/k0**2). Because the material does not vary along the second transverse axis, the auxiliary equation separates into column solves. Factorizations depend on the operator, not the right-hand side, and can be reused across propagation increments with unchanged operator inputs. Equivalent solves are valid; explicit inversion is unnecessary.

Returns
-------
return np.zeros_like(U, dtype=complex), []
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def wjsolver(kx, ky, mx, B, C, bj, k0, U, lu_factors=None):
    """Solve one auxiliary Fourier-space resolvent for every column.

    Solve (I + bj*X_column) @ Wj[:, column] = U[:, column], where
    X_column = M_tilde - I - diag((kx**2 + ky[column]**2) / k0**2).
    M_tilde is the positive index-squared matrix from Step 2.
    kx and ky have shape (ny,) in centered order; U has shape (ny, ny).
    k0 is positive, mx is an integer, B and C are real, and bj is complex.
    Return (Wj, lu_factors), with one reusable factor per column.
    A supplied cache must match the pole, grid, and material, but U may
    differ from the right-hand side used to construct the cache.
    """
    return np.zeros_like(U, dtype=complex), []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu


def _oracle_wjsolver(kx, ky, mx, B, C, bj, k0, U, lu_factors=None):
    """Solve the forward resolvent using reusable column factorizations.

    The operator is (1 - bj)*I - bj*D_column + bj*M_tilde, where
    D_column = diag((kx**2 + ky[column]**2) / k0**2). Cached factors
    are valid only for unchanged pole, grid, and material parameters.
    """
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    U = np.asarray(U, dtype=complex)

    if k0 <= 0:
        raise ValueError("k0 must be positive.")
    if kx.ndim != 1 or ky.ndim != 1:
        raise ValueError("kx and ky must be one-dimensional.")
    if kx.shape != ky.shape:
        raise ValueError("kx and ky must have the same shape.")

    ny = len(kx)
    if U.shape != (ny, ny):
        raise ValueError("U must have shape (ny, ny).")
    if not isinstance(mx, (int, np.integer)):
        raise ValueError("mx must be an integer.")

    if lu_factors is None:
        modulation_matrix = _oracle_build_modulation_matrix(
            ny=ny, mx=mx, B=B, C=C
        )
        identity = eye(ny, format="csc", dtype=complex)
        lu_factors = []

        for column_index in range(ny):
            transverse_diagonal = (
                kx**2 + ky[column_index]**2
            ) / k0**2
            system_matrix = (
                (1.0 - bj) * identity
                - bj * diags(transverse_diagonal, format="csc")
                + bj * modulation_matrix
            )
            lu_factors.append(splu(system_matrix.tocsc()))

    if len(lu_factors) != ny:
        raise ValueError(
            "lu_factors must contain one factorization per ky index."
        )

    auxiliary_field = np.empty((ny, ny), dtype=complex)
    for column_index in range(ny):
        auxiliary_field[:, column_index] = lu_factors[column_index].solve(
            U[:, column_index]
        )

    return auxiliary_field, lu_factors

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np


def centered_wavenumbers(ny, dx=0.5, dy=0.5):
    kx = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(ny, d=dx)
    )
    ky = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(ny, d=dy)
    )
    return kx, ky


ny = 8
kx, ky = centered_wavenumbers(ny)
k0 = 2.0 * np.pi / (3e-6)

rng = np.random.default_rng(1)
U = (
    rng.random((ny, ny))
    + 1j * rng.random((ny, ny))
)


def run_model():
    Wj, _ = wjsolver(
        kx=kx,
        ky=ky,
        mx=1,
        B=0.1,
        C=1.4,
        bj=0.2 + 0.1j,
        k0=k0,
        U=U
    )
    return Wj


def run_gold():
    Wj, _ = _oracle_wjsolver(
        kx=kx,
        ky=ky,
        mx=1,
        B=0.1,
        C=1.4,
        bj=0.2 + 0.1j,
        k0=k0,
        U=U
    )
    return Wj
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np


def centered_wavenumbers(ny, dx=0.5, dy=0.5):
    kx = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(ny, d=dx)
    )
    ky = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(ny, d=dy)
    )
    return kx, ky


ny = 8
kx, ky = centered_wavenumbers(ny)
k0 = 2.0 * np.pi / (3e-6)

# For mx = ny/2, the +mx and -mx shifts
# coincide modulo ny.
U = np.eye(ny, dtype=complex)


def run_model():
    Wj, _ = wjsolver(
        kx=kx,
        ky=ky,
        mx=ny // 2,
        B=0.1,
        C=1.4,
        bj=0.2,
        k0=k0,
        U=U
    )
    return Wj


def run_gold():
    Wj, _ = _oracle_wjsolver(
        kx=kx,
        ky=ky,
        mx=ny // 2,
        B=0.1,
        C=1.4,
        bj=0.2,
        k0=k0,
        U=U
    )
    return Wj
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np


def centered_wavenumbers(ny, dx=0.5, dy=0.5):
    kx = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(ny, d=dx)
    )
    ky = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(ny, d=dy)
    )
    return kx, ky


ny = 8
kx, ky = centered_wavenumbers(ny)
k0 = 2.0 * np.pi / (3e-6)

# For mx = ny, all periodic shifts wrap to the
# diagonal. The modulation matrix must accumulate
# all coincident contributions.
U = np.ones(
    (ny, ny),
    dtype=complex
)


def run_model():
    Wj, _ = wjsolver(
        kx=kx,
        ky=ky,
        mx=ny,
        B=0.1,
        C=1.4,
        bj=0.2,
        k0=k0,
        U=U
    )
    return Wj


def run_gold():
    Wj, _ = _oracle_wjsolver(
        kx=kx,
        ky=ky,
        mx=ny,
        B=0.1,
        C=1.4,
        bj=0.2,
        k0=k0,
        U=U
    )
    return Wj
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np

size = 8
kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(size, d=0.7))
ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(size, d=1.1))
pole = 0.2 + 0.1j
identity = np.eye(size)
index_matrix = 1.4 * identity + 0.05 * (
    np.roll(identity, 1, axis=1) + np.roll(identity, -1, axis=1)
)
material = index_matrix @ index_matrix
transverse = (kx[:, None]**2 + ky[None, :]**2) / 4.0**2
rng = np.random.default_rng(37)
first_rhs = rng.normal(size=(size, size))
first_rhs = first_rhs + 1j * rng.normal(size=(size, size))
second_rhs = (0.8 + 0.3j) * np.roll(first_rhs, 1, axis=0)


def check_residuals():
    cache = None
    errors = []
    for rhs in (first_rhs, second_rhs):
        field, cache = wjsolver(
            kx, ky, 1, 0.1, 1.4, pole, 4.0,
            rhs.copy(), lu_factors=cache,
        )
        residual = field + pole * (
            material @ field - field - transverse * field
        ) - rhs
        errors.append(np.linalg.norm(residual) / np.linalg.norm(rhs))
    return float(np.max(errors))
""",
            "call": "check_residuals()",
            "gold_call": "0.0",
        },
    ]
