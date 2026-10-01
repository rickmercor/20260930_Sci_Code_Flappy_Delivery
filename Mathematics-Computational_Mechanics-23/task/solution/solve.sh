#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def assemble_discrete_hamiltonian_operator(
    M_phi: np.ndarray,
    M_d: np.ndarray,
    C_N: np.ndarray,
    C_M: np.ndarray,
) -> np.ndarray:
    import numpy as np

    M_phi = np.asarray(M_phi, dtype=float)
    M_d = np.asarray(M_d, dtype=float)
    C_N = np.asarray(C_N, dtype=float)
    C_M = np.asarray(C_M, dtype=float)

    if M_phi.shape != (3, 3):
        raise ValueError("M_phi must have shape (3, 3).")
    if M_d.shape != (9, 9):
        raise ValueError("M_d must have shape (9, 9).")
    if C_N.shape != (3, 3):
        raise ValueError("C_N must have shape (3, 3).")
    if C_M.shape != (3, 3):
        raise ValueError("C_M must have shape (3, 3).")

    blocks = [M_phi, M_d, C_N, C_M]

    if not all(np.all(np.isfinite(A)) for A in blocks):
        raise ValueError("All operator blocks must be finite.")

    if not all(
        np.allclose(A, A.T, rtol=1e-12, atol=1e-12)
        for A in blocks
    ):
        raise ValueError("All operator blocks must be symmetric.")

    K = np.zeros((18, 18), dtype=float)

    K[0:3, 0:3] = M_phi
    K[3:12, 3:12] = M_d
    K[12:15, 12:15] = C_N
    K[15:18, 15:18] = C_M

    return K

def evaluate_endpoint_energy_increment(
    v_phi_n: np.ndarray,
    v_d_n: np.ndarray,
    N_n: np.ndarray,
    M_n: np.ndarray,
    v_phi_np1: np.ndarray,
    v_d_np1: np.ndarray,
    N_np1: np.ndarray,
    M_np1: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    import numpy as np

    v_phi_n = np.asarray(v_phi_n, dtype=float)
    v_d_n = np.asarray(v_d_n, dtype=float)
    N_n = np.asarray(N_n, dtype=float)
    M_n = np.asarray(M_n, dtype=float)

    v_phi_np1 = np.asarray(v_phi_np1, dtype=float)
    v_d_np1 = np.asarray(v_d_np1, dtype=float)
    N_np1 = np.asarray(N_np1, dtype=float)
    M_np1 = np.asarray(M_np1, dtype=float)

    K = np.asarray(K, dtype=float)

    vectors = [
        (v_phi_n, (3,)),
        (v_d_n, (9,)),
        (N_n, (3,)),
        (M_n, (3,)),
        (v_phi_np1, (3,)),
        (v_d_np1, (9,)),
        (N_np1, (3,)),
        (M_np1, (3,)),
    ]

    if any(v.shape != shape for v, shape in vectors):
        raise ValueError("Endpoint state vector has invalid shape.")

    if K.shape != (18, 18):
        raise ValueError("K must have shape (18, 18).")

    if not all(np.all(np.isfinite(v)) for v, _ in vectors):
        raise ValueError("Endpoint states must be finite.")

    if not np.all(np.isfinite(K)):
        raise ValueError("K must be finite.")

    z_n = np.concatenate([
        v_phi_n,
        v_d_n,
        N_n,
        M_n,
    ])

    z_np1 = np.concatenate([
        v_phi_np1,
        v_d_np1,
        N_np1,
        M_np1,
    ])

    H_n = 0.5 * float(z_n @ K @ z_n)
    H_np1 = 0.5 * float(z_np1 @ K @ z_np1)
    delta_H = H_np1 - H_n

    result = np.array(
        [H_n, H_np1, delta_H],
        dtype=float,
    )

    if not np.all(np.isfinite(result)):
        raise ValueError("Endpoint energies must be finite.")

    return result

def build_director_kinematic_map(
    directors: np.ndarray,
) -> np.ndarray:
    import numpy as np

    directors = np.asarray(
        directors,
        dtype=float,
    )

    if directors.shape != (3, 3):
        raise ValueError(
            "directors must have shape (3, 3)."
        )

    if not np.all(np.isfinite(directors)):
        raise ValueError(
            "directors must contain only finite values."
        )

    gram = directors @ directors.T

    if not np.allclose(
        gram,
        np.eye(3),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError(
            "directors must form an orthonormal triad."
        )

    if not np.isclose(
        np.linalg.det(directors),
        1.0,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError(
            "directors must form a proper right-handed triad."
        )

    def skew(a):
        x, y, z = a

        return np.array([
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ], dtype=float)

    T = -0.5 * np.vstack([
        skew(directors[0]),
        skew(directors[1]),
        skew(directors[2]),
    ])

    return T

def assemble_mechanical_boundary_effort(
    T: np.ndarray,
    n_boundary: np.ndarray,
    m_boundary: np.ndarray,
) -> np.ndarray:
    import numpy as np

    T = np.asarray(T, dtype=float)
    n_boundary = np.asarray(n_boundary, dtype=float)
    m_boundary = np.asarray(m_boundary, dtype=float)

    if T.shape != (9, 3):
        raise ValueError("T must have shape (9, 3).")

    if n_boundary.shape != (3,):
        raise ValueError("n_boundary must have shape (3,).")

    if m_boundary.shape != (3,):
        raise ValueError("m_boundary must have shape (3,).")

    if not (
        np.all(np.isfinite(T))
        and np.all(np.isfinite(n_boundary))
        and np.all(np.isfinite(m_boundary))
    ):
        raise ValueError("Boundary-effort inputs must be finite.")

    director_effort = T @ m_boundary

    boundary_effort = np.concatenate([
        n_boundary,
        director_effort,
    ])

    return boundary_effort

def evaluate_midpoint_boundary_port(
    T: np.ndarray,
    boundary_effort: np.ndarray,
    v_d_boundary: np.ndarray,
    v_phi_boundary: np.ndarray,
) -> np.ndarray:
    import numpy as np

    T = np.asarray(T, dtype=float)
    boundary_effort = np.asarray(boundary_effort, dtype=float)
    v_d_boundary = np.asarray(v_d_boundary, dtype=float)
    v_phi_boundary = np.asarray(v_phi_boundary, dtype=float)

    if T.shape != (9, 3):
        raise ValueError("T must have shape (9, 3).")

    if boundary_effort.shape != (12,):
        raise ValueError("boundary_effort must have shape (12,).")

    if v_d_boundary.shape != (9,):
        raise ValueError("v_d_boundary must have shape (9,).")

    if v_phi_boundary.shape != (3,):
        raise ValueError("v_phi_boundary must have shape (3,).")

    if not (
        np.all(np.isfinite(T))
        and np.all(np.isfinite(boundary_effort))
        and np.all(np.isfinite(v_d_boundary))
        and np.all(np.isfinite(v_phi_boundary))
    ):
        raise ValueError("Boundary-port inputs must be finite.")

    omega = T.T @ v_d_boundary

    translational_effort = boundary_effort[:3]
    director_effort = boundary_effort[3:]

    p_trans = float(v_phi_boundary @ translational_effort)
    p_rot = float(v_d_boundary @ director_effort)
    p_total = p_trans + p_rot

    return np.array(
        [
            omega[0],
            omega[1],
            omega[2],
            p_trans,
            p_rot,
            p_total,
        ],
        dtype=float,
    )

def compute_discrete_power_balance_defect(
    energy_summary: np.ndarray,
    port_summary: np.ndarray,
    h: float,
) -> np.ndarray:
    import numpy as np

    energy_summary = np.asarray(energy_summary, dtype=float)
    port_summary = np.asarray(port_summary, dtype=float)

    if energy_summary.shape != (3,):
        raise ValueError("energy_summary must have shape (3,).")

    if port_summary.shape != (6,):
        raise ValueError("port_summary must have shape (6,).")

    if not np.all(np.isfinite(energy_summary)):
        raise ValueError("energy_summary must contain finite values.")

    if not np.all(np.isfinite(port_summary)):
        raise ValueError("port_summary must contain finite values.")

    h = float(h)

    if not np.isfinite(h):
        raise ValueError("h must be finite.")

    if h < 0.0:
        raise ValueError("h must be nonnegative.")

    delta_H = float(energy_summary[2])
    total_power = float(port_summary[5])

    boundary_work = h * total_power
    defect = delta_H - boundary_work

    return np.array(
        [boundary_work, defect],
        dtype=float,
    )

def compute_energy_balance_defect(
    M_phi: np.ndarray,
    M_d: np.ndarray,
    C_N: np.ndarray,
    C_M: np.ndarray,
    v_phi_n: np.ndarray,
    v_d_n: np.ndarray,
    N_n: np.ndarray,
    M_n: np.ndarray,
    v_phi_np1: np.ndarray,
    v_d_np1: np.ndarray,
    N_np1: np.ndarray,
    M_np1: np.ndarray,
    directors: np.ndarray,
    v_d_boundary: np.ndarray,
    v_phi_boundary: np.ndarray,
    n_boundary: np.ndarray,
    m_boundary: np.ndarray,
    h: float,
) -> float:
    import numpy as np

    K = assemble_discrete_hamiltonian_operator(
        M_phi,
        M_d,
        C_N,
        C_M,
    )

    energy_summary = evaluate_endpoint_energy_increment(
        v_phi_n,
        v_d_n,
        N_n,
        M_n,
        v_phi_np1,
        v_d_np1,
        N_np1,
        M_np1,
        K,
    )

    T = build_director_kinematic_map(
        directors,
    )

    boundary_effort = assemble_mechanical_boundary_effort(
        T,
        n_boundary,
        m_boundary,
    )

    port_summary = evaluate_midpoint_boundary_port(
        T,
        boundary_effort,
        v_d_boundary,
        v_phi_boundary,
    )

    balance_summary = compute_discrete_power_balance_defect(
        energy_summary,
        port_summary,
        h,
    )

    defect = float(balance_summary[1])

    if not np.isfinite(defect):
        raise ValueError("The energy-balance defect must be finite.")

    return defect
SCICODE_GOLD_EOF
