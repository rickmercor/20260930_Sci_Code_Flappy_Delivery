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

def build_finite_source(
    L: int,
    R: int,
    charge: int = 1,
) -> np.ndarray:
    """Construct the finite Dirac source in plaquette order (01,02,12)."""
    if (
        isinstance(L, (bool, np.bool_))
        or not isinstance(L, (int, np.integer))
        or L < 4
        or L % 2 != 0
    ):
        raise ValueError("L must be an even integer >= 4")

    if (
        isinstance(R, (bool, np.bool_))
        or not isinstance(R, (int, np.integer))
        or not 1 <= R < L // 2
    ):
        raise ValueError("R must be an integer satisfying 1 <= R < L/2")

    if (
        isinstance(charge, (bool, np.bool_))
        or not isinstance(charge, (int, np.integer))
        or charge == 0
    ):
        raise ValueError("charge must be a nonzero integer")

    sigma = np.zeros((3, L, L, L), dtype=float)
    sigma[0, 0, 0, 1:R + 1] = -charge

    return sigma

import numpy as np

def lattice_curl(B: np.ndarray) -> np.ndarray:
    """Compute the periodic noncompact curl in order (01,02,12)."""
    B = np.asarray(B, dtype=float)

    if (
        B.ndim != 4
        or B.shape[0] != 3
        or min(B.shape[1:]) < 2
        or not np.all(np.isfinite(B))
    ):
        raise ValueError(
            "B must be finite with shape (3,Lx,Ly,Lz), "
            "with each spatial side at least 2"
        )

    components = []

    for u, v in ((0, 1), (0, 2), (1, 2)):
        forward_u_Bv = np.roll(B[v], -1, axis=u)
        forward_v_Bu = np.roll(B[u], -1, axis=v)

        component = (
            B[u]
            + forward_u_Bv
            - forward_v_Bu
            - B[v]
        )

        components.append(component)

    return np.stack(components, axis=0)

import numpy as np

def lattice_equations(
    state: np.ndarray,
    sigma: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
) -> np.ndarray:
    """Return five residual channels and six local curvature channels."""
    q = np.asarray(state, dtype=float)
    s = np.asarray(sigma, dtype=float)

    if (
        q.ndim != 4
        or q.shape[0] != 5
        or min(q.shape[1:]) < 2
    ):
        raise ValueError(
            "state must have shape (5,Lx,Ly,Lz), "
            "with each spatial side at least 2"
        )

    if s.shape != (3,) + q.shape[1:]:
        raise ValueError("sigma and state have incompatible shapes")

    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(s)):
        raise ValueError("state and sigma must contain finite values")

    if (
        not np.isfinite(mB)
        or not np.isfinite(mchi)
        or mB <= 0.0
        or mchi <= 0.0
    ):
        raise ValueError("masses must be positive and finite")

    B = q[:3]
    chi = q[3] + 1j * q[4]

    F = lattice_curl(B) - 2.0 * np.pi * s

    X = np.zeros_like(B)

    for k, (u, v) in enumerate(((0, 1), (0, 2), (1, 2))):
        X[u] += F[k] - np.roll(F[k], 1, axis=v)
        X[v] -= F[k] - np.roll(F[k], 1, axis=u)

    dX = np.empty_like(B)
    magnitude_squared = np.abs(chi) ** 2

    Y = (
        0.5
        * mchi ** 2
        * chi
        * (magnitude_squared - 1.0)
    )

    for u in range(3):
        link = np.exp(1j * B[u])

        forward_transport = (
            link * np.roll(chi, -1, axis=u)
        )

        backward_transport = np.roll(
            np.conjugate(link) * chi,
            1,
            axis=u,
        )

        overlap = np.conjugate(chi) * forward_transport

        X[u] += mB ** 2 * overlap.imag
        dX[u] = 4.0 + mB ** 2 * overlap.real

        Y += (
            2.0 * chi
            - forward_transport
            - backward_transport
        )

    base = 6.0 + 0.5 * mchi ** 2 * (
        magnitude_squared - 1.0
    )

    H11 = base + mchi ** 2 * q[3] ** 2
    H12 = mchi ** 2 * q[3] * q[4]
    H22 = base + mchi ** 2 * q[4] ** 2

    return np.concatenate(
        (
            X,
            np.stack((Y.real, Y.imag), axis=0),
            dX,
            np.stack((H11, H12, H22), axis=0),
        ),
        axis=0,
    )

import numpy as np

def newton_sweep(
    state: np.ndarray,
    sigma: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
    damping: float = 0.8,
) -> np.ndarray:
    """Perform one specified checkerboard block-Newton sweep."""
    q = np.asarray(state, dtype=float)
    s = np.asarray(sigma, dtype=float)

    if (
        q.ndim != 4
        or q.shape[0] != 5
        or min(q.shape[1:]) < 2
    ):
        raise ValueError(
            "state must have shape (5,Lx,Ly,Lz), "
            "with each spatial side at least 2"
        )

    if s.shape != (3,) + q.shape[1:]:
        raise ValueError("sigma and state have incompatible shapes")

    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(s)):
        raise ValueError("state and sigma must contain finite values")

    if (
        not np.isfinite(mB)
        or not np.isfinite(mchi)
        or mB <= 0.0
        or mchi <= 0.0
    ):
        raise ValueError("masses must be positive and finite")

    if any(length % 2 != 0 for length in q.shape[1:]):
        raise ValueError("checkerboard updating requires even side lengths")

    if not np.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0,1]")

    q = q.copy()
    parity = np.indices(q.shape[1:]).sum(axis=0) % 2

    for selected_parity in (0, 1):
        mask = parity == selected_parity

        for direction in range(3):
            values = lattice_equations(
                q, s, mB, mchi
            )

            residual = values[direction]
            curvature = values[5 + direction]

            if np.any(curvature[mask] <= 0.0):
                raise ValueError("nonpositive gauge curvature")

            q[direction][mask] -= (
                damping
                * residual[mask]
                / curvature[mask]
            )

        values = lattice_equations(
            q, s, mB, mchi
        )

        Y1 = values[3]
        Y2 = values[4]
        H11 = values[8]
        H12 = values[9]
        H22 = values[10]

        determinant = H11 * H22 - H12 ** 2

        if (
            np.any(H11[mask] <= 0.0)
            or np.any(determinant[mask] <= 0.0)
        ):
            raise ValueError(
                "scalar Newton block must be positive definite"
            )

        correction_real = (
            H22[mask] * Y1[mask]
            - H12[mask] * Y2[mask]
        ) / determinant[mask]

        correction_imag = (
            H11[mask] * Y2[mask]
            - H12[mask] * Y1[mask]
        ) / determinant[mask]

        q[3][mask] -= damping * correction_real
        q[4][mask] -= damping * correction_imag

    return q

import numpy as np

def solve_finite_tube(
    sigma: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
    tolerance: float = 1.0e-9,
    max_sweeps: int = 20000,
    damping: float = 0.8,
) -> np.ndarray:
    """Return the converged finite-source DGL state."""
    s = np.asarray(sigma, dtype=float)

    if s.ndim != 4 or s.shape[0] != 3:
        raise ValueError(
            "sigma must have shape (3,Lx,Ly,Lz)"
        )

    if not np.all(np.isfinite(s)):
        raise ValueError("sigma must contain finite values")

    if any(
        length < 4 or length % 2 != 0
        for length in s.shape[1:]
    ):
        raise ValueError(
            "spatial side lengths must be even and at least 4"
        )

    if (
        not np.isfinite(mB)
        or not np.isfinite(mchi)
        or mB <= 0.0
        or mchi <= 0.0
    ):
        raise ValueError("masses must be positive and finite")

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    if (
        isinstance(max_sweeps, (bool, np.bool_))
        or not isinstance(max_sweeps, (int, np.integer))
        or max_sweeps < 1
    ):
        raise ValueError("max_sweeps must be a positive integer")

    if not np.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0,1]")

    state = np.zeros((5,) + s.shape[1:], dtype=float)
    state[3] = 1.0

    for iteration in range(max_sweeps + 1):
        values = lattice_equations(
            state,
            s,
            mB,
            mchi,
        )

        max_residual = float(
            np.max(np.abs(values[:5]))
        )

        if max_residual < tolerance:
            return state

        if iteration < max_sweeps:
            state = newton_sweep(
                state,
                s,
                mB,
                mchi,
                damping,
            )

    raise ValueError(
        "solver did not converge within max_sweeps"
    )

import numpy as np

def periodic_green(L: int) -> np.ndarray:
    """Construct the zero-mean periodic massless lattice Green function."""
    if (
        isinstance(L, (bool, np.bool_))
        or not isinstance(L, (int, np.integer))
        or L < 2
    ):
        raise ValueError("L must be an integer >= 2")

    momenta = np.arange(L, dtype=float)

    one_dimensional_eigenvalues = (
        4.0 * np.sin(np.pi * momenta / L) ** 2
    )

    eigenvalues = (
        one_dimensional_eigenvalues[:, None, None]
        + one_dimensional_eigenvalues[None, :, None]
        + one_dimensional_eigenvalues[None, None, :]
    )

    fourier_kernel = np.zeros_like(eigenvalues)

    np.divide(
        1.0,
        eigenvalues,
        out=fourier_kernel,
        where=eigenvalues > 0.0,
    )

    fourier_kernel[0, 0, 0] = 0.0

    return np.fft.ifftn(fourier_kernel).real

import numpy as np

def hodge_action(
    state: np.ndarray,
    sigma: np.ndarray,
    G: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
) -> np.ndarray:
    """Return the full action, its Hodge split, and reconstruction errors."""
    q = np.asarray(state, dtype=float)
    s = np.asarray(sigma, dtype=float)
    G = np.asarray(G, dtype=float)

    if (
        q.ndim != 4
        or q.shape[0] != 5
        or min(q.shape[1:]) < 2
    ):
        raise ValueError(
            "state must have shape (5,Lx,Ly,Lz), "
            "with each spatial side at least 2"
        )

    if s.shape != (3,) + q.shape[1:]:
        raise ValueError("sigma and state have incompatible shapes")

    if G.shape != q.shape[1:]:
        raise ValueError("G and state have incompatible shapes")

    if (
        not np.all(np.isfinite(q))
        or not np.all(np.isfinite(s))
        or not np.all(np.isfinite(G))
    ):
        raise ValueError("all arrays must contain finite values")

    if (
        not np.isfinite(mB)
        or not np.isfinite(mchi)
        or mB <= 0.0
        or mchi <= 0.0
    ):
        raise ValueError("masses must be positive and finite")

    B = q[:3]
    chi = q[3] + 1j * q[4]

    F = lattice_curl(B) - 2.0 * np.pi * s

    charge_density = -(
        np.roll(s[2], -1, axis=0) - s[2]
        - np.roll(s[1], -1, axis=1) + s[1]
        + np.roll(s[0], -1, axis=2) - s[0]
    )

    green_fourier = np.fft.fftn(G)

    psi = np.fft.ifftn(
        green_fourier * np.fft.fftn(charge_density)
    ).real

    C = np.stack(
        (
            -(psi - np.roll(psi, 1, axis=2)),
            psi - np.roll(psi, 1, axis=1),
            -(psi - np.roll(psi, 1, axis=0)),
        ),
        axis=0,
    )

    Fcoul = 2.0 * np.pi * C

    divergence = np.zeros_like(B)

    for k, (u, v) in enumerate(((0, 1), (0, 2), (1, 2))):
        divergence[u] += (
            F[k] - np.roll(F[k], 1, axis=v)
        )
        divergence[v] -= (
            F[k] - np.roll(F[k], 1, axis=u)
        )

    Breg = np.stack(
        [
            np.fft.ifftn(
                green_fourier * np.fft.fftn(component)
            ).real
            for component in divergence
        ],
        axis=0,
    )

    correction = (
        2.0
        * np.pi
        * np.mean(s, axis=(1, 2, 3))
    )[:, None, None, None]

    Fsole = lattice_curl(Breg) - correction

    kinetic_sum = 0.0

    for direction in range(3):
        covariant_difference = (
            chi
            - np.exp(1j * B[direction])
            * np.roll(chi, -1, axis=direction)
        )
        kinetic_sum += np.sum(
            np.abs(covariant_difference) ** 2
        )

    matter_action = (
        0.5 * mB ** 2 * kinetic_sum
        + (mB ** 2 * mchi ** 2 / 8.0)
        * np.sum((np.abs(chi) ** 2 - 1.0) ** 2)
    )

    S = 0.5 * np.sum(F ** 2) + matter_action
    Scoul = 0.5 * np.sum(Fcoul ** 2)
    Ssole = 0.5 * np.sum(Fsole ** 2) + matter_action

    field_error = np.max(
        np.abs(F - Fsole - Fcoul)
    )

    energy_error = S - Scoul - Ssole

    return np.array(
        [
            S,
            Scoul,
            Ssole,
            field_error,
            energy_error,
        ],
        dtype=float,
    )

import numpy as np

def run_length_curvature(
    L: int = 16,
    R: int = 5,
    halfspan: int = 2,
    mB: float = 0.5,
    mchi: float = 0.5,
    tolerance: float = 1.0e-9,
) -> float:
    """Compute the finite-box solenoidal-potential curvature."""
    if (
        isinstance(L, (bool, np.bool_))
        or not isinstance(L, (int, np.integer))
        or L < 4
        or L % 2 != 0
    ):
        raise ValueError("L must be an even integer >= 4")

    if (
        isinstance(R, (bool, np.bool_))
        or not isinstance(R, (int, np.integer))
    ):
        raise ValueError("R must be an integer")

    if (
        isinstance(halfspan, (bool, np.bool_))
        or not isinstance(halfspan, (int, np.integer))
        or halfspan < 1
    ):
        raise ValueError("halfspan must be a positive integer")

    if (
        not np.isfinite(mB)
        or not np.isfinite(mchi)
        or mB <= 0.0
        or mchi <= 0.0
    ):
        raise ValueError("masses must be positive and finite")

    if mB != mchi:
        raise ValueError(
            "this benchmark requires equal masses mB=mchi"
        )

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    separations = (
        R - halfspan,
        R,
        R + halfspan,
    )

    if any(
        not 1 <= separation < L // 2
        for separation in separations
    ):
        raise ValueError(
            "all separations must satisfy 1 <= separation < L/2"
        )

    sources = [
        build_finite_source(L, separation, 1)
        for separation in separations
    ]

    G = periodic_green(L)
    potentials = []

    for sigma in sources:
        state = solve_finite_tube(
            sigma,
            mB=mB,
            mchi=mchi,
            tolerance=tolerance,
            max_sweeps=20000,
            damping=0.8,
        )

        direct_curl = lattice_curl(state[:3])
        if (
            direct_curl.shape != sigma.shape
            or not np.all(np.isfinite(direct_curl))
        ):
            raise ValueError("direct curl consistency check failed")

        direct_equations = lattice_equations(
            state,
            sigma,
            mB,
            mchi,
        )
        direct_residual = float(
            np.max(np.abs(direct_equations[:5]))
        )
        if direct_residual >= tolerance:
            raise ValueError("stationary residual check failed")

        post_sweep_state = newton_sweep(
            state,
            sigma,
            mB,
            mchi,
            0.8,
        )
        post_sweep_equations = lattice_equations(
            post_sweep_state,
            sigma,
            mB,
            mchi,
        )
        post_sweep_residual = float(
            np.max(np.abs(post_sweep_equations[:5]))
        )
        if (
            post_sweep_state.shape != state.shape
            or not np.all(np.isfinite(post_sweep_state))
            or post_sweep_residual >= tolerance
        ):
            raise ValueError("post-sweep consistency check failed")

        actions = hodge_action(
            state,
            sigma,
            G,
            mB=mB,
            mchi=mchi,
        )

        potentials.append(float(actions[2]))

    curvature = (
        100.0
        * (
            potentials[2]
            - 2.0 * potentials[1]
            + potentials[0]
        )
        / (halfspan * np.pi * mB ** 2)
    )

    if not np.isfinite(curvature):
        raise ValueError("computed curvature is not finite")

    return float(curvature)
SCICODE_GOLD_EOF
