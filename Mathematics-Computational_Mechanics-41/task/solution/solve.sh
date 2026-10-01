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


def tet10_shape_data(
    xi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    xi = np.asarray(
        xi,
        dtype=float,
    )

    if xi.shape != (3,):
        raise ValueError(
            "xi must have shape (3,)"
        )

    x, y, z = xi

    L = np.array(
        [
            1.0 - x - y - z,
            x,
            y,
            z,
        ],
        dtype=float,
    )

    if (
        np.min(L) < -1e-14
        or np.max(L) > 1.0 + 1e-14
    ):
        raise ValueError(
            "point lies outside the reference tetrahedron"
        )

    dL = np.array(
        [
            [-1.0, -1.0, -1.0],
            [ 1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0,  1.0],
        ],
        dtype=float,
    )

    N = np.empty(
        10,
        dtype=float,
    )

    dN = np.empty(
        (10, 3),
        dtype=float,
    )

    for a in range(4):

        N[a] = (
            L[a]
            * (
                2.0 * L[a]
                - 1.0
            )
        )

        dN[a] = (
            (
                4.0 * L[a]
                - 1.0
            )
            * dL[a]
        )

    edges = (
        (0, 1),
        (1, 2),
        (2, 0),
        (0, 3),
        (1, 3),
        (2, 3),
    )

    for k, (a, b) in enumerate(
        edges,
        start=4,
    ):

        N[k] = (
            4.0
            * L[a]
            * L[b]
        )

        dN[k] = (
            4.0
            * (
                dL[a] * L[b]
                + L[a] * dL[b]
            )
        )

    return N, dN

import numpy as np


def tet10_kinematics(
    coords: np.ndarray,
    disp: np.ndarray,
    xi: np.ndarray,
):

    coords = np.asarray(
        coords,
        dtype=float,
    )

    disp = np.asarray(
        disp,
        dtype=float,
    )

    if (
        coords.shape != (10, 3)
        or disp.shape != (10, 3)
    ):
        raise ValueError(
            "coords and disp must have shape (10,3)"
        )

    N, dN_nat = (
        tet10_shape_data(
            xi
        )
    )

    J = (
        coords.T
        @ dN_nat
    )

    detJ = float(
        np.linalg.det(J)
    )

    if (
        not np.isfinite(detJ)
        or detJ <= 0.0
    ):
        raise ValueError(
            "nonpositive reference Jacobian"
        )

    gradN = (
        dN_nat
        @ np.linalg.inv(J)
    )

    current = (
        coords
        + disp
    )

    F = (
        current.T
        @ gradN
    )

    E_tensor = (
        0.5
        * (
            F.T @ F
            - np.eye(3)
        )
    )

    E_green = np.array(
        [
            E_tensor[0, 0],
            E_tensor[1, 1],
            E_tensor[2, 2],
            2.0 * E_tensor[0, 1],
            2.0 * E_tensor[0, 2],
            2.0 * E_tensor[1, 2],
        ],
        dtype=float,
    )

    B = np.zeros(
        (6, 30),
        dtype=float,
    )

    x1 = F[:, 0]
    x2 = F[:, 1]
    x3 = F[:, 2]

    for i in range(10):

        g1, g2, g3 = (
            gradN[i]
        )

        block = np.vstack(
            (
                g1 * x1,
                g2 * x2,
                g3 * x3,
                g1 * x2
                + g2 * x1,
                g1 * x3
                + g3 * x1,
                g2 * x3
                + g3 * x2,
            )
        )

        B[
            :,
            3 * i : 3 * i + 3,
        ] = block

    return (
        detJ,
        gradN,
        F,
        E_green,
        B,
    )

import numpy as np


def neo_hooke_green_response(
    E_green: np.ndarray,
    young: float,
    poisson: float,
):

    E_green = np.asarray(
        E_green,
        dtype=float,
    )

    if E_green.shape != (6,):
        raise ValueError(
            "E_green must have shape (6,)"
        )

    young = float(young)
    poisson = float(poisson)

    if (
        not np.isfinite(young)
        or young <= 0.0
        or not np.isfinite(poisson)
        or not (-1.0 < poisson < 0.5)
    ):
        raise ValueError(
            "invalid elastic constants"
        )

    mu = (
        young
        / (
            2.0
            * (
                1.0 + poisson
            )
        )
    )

    lame = (
        young
        * poisson
        / (
            (
                1.0 + poisson
            )
            * (
                1.0
                - 2.0 * poisson
            )
        )
    )

    E = np.array(
        [
            [
                E_green[0],
                0.5 * E_green[3],
                0.5 * E_green[4],
            ],
            [
                0.5 * E_green[3],
                E_green[1],
                0.5 * E_green[5],
            ],
            [
                0.5 * E_green[4],
                0.5 * E_green[5],
                E_green[2],
            ],
        ],
        dtype=float,
    )

    C = (
        np.eye(3)
        + 2.0 * E
    )

    detC = float(
        np.linalg.det(C)
    )

    if (
        not np.isfinite(detC)
        or detC <= 0.0
    ):
        raise ValueError(
            "C must be positive definite"
        )

    Cinv = np.linalg.inv(C)

    W = (
        0.5
        * mu
        * (
            np.trace(C)
            - 3.0
            - np.log(detC)
        )
        + 0.25
        * lame
        * (
            detC
            - 1.0
            - np.log(detC)
        )
    )

    alpha = (
        -mu
        + 0.5
        * lame
        * (
            detC
            - 1.0
        )
    )

    S = (
        mu * np.eye(3)
        + alpha * Cinv
    )

    stress = np.array(
        [
            S[0, 0],
            S[1, 1],
            S[2, 2],
            S[0, 1],
            S[0, 2],
            S[1, 2],
        ],
        dtype=float,
    )

    D = np.zeros(
        (6, 6),
        dtype=float,
    )

    for j in range(6):

        dC = np.zeros(
            (3, 3),
            dtype=float,
        )

        if j == 0:
            dC[0, 0] = 2.0

        elif j == 1:
            dC[1, 1] = 2.0

        elif j == 2:
            dC[2, 2] = 2.0

        elif j == 3:
            dC[0, 1] = 1.0
            dC[1, 0] = 1.0

        elif j == 4:
            dC[0, 2] = 1.0
            dC[2, 0] = 1.0

        else:
            dC[1, 2] = 1.0
            dC[2, 1] = 1.0

        ddetC = (
            detC
            * np.trace(
                Cinv @ dC
            )
        )

        dCinv = (
            -Cinv
            @ dC
            @ Cinv
        )

        dS = (
            0.5
            * lame
            * ddetC
            * Cinv
            + alpha
            * dCinv
        )

        D[:, j] = np.array(
            [
                dS[0, 0],
                dS[1, 1],
                dS[2, 2],
                dS[0, 1],
                dS[0, 2],
                dS[1, 2],
            ],
            dtype=float,
        )

    return (
        float(W),
        stress,
        D,
    )

import numpy as np


def hwt10_mixed_interpolation(
    xi: np.ndarray,
):

    xi = np.asarray(
        xi,
        dtype=float,
    )

    if xi.shape != (3,):
        raise ValueError(
            "xi must have shape (3,)"
        )

    x, y, z = xi

    M = np.zeros(
        (6, 18),
        dtype=float,
    )

    values = np.array(
        [x, y, z],
        dtype=float,
    )

    for i in range(6):

        M[
            i,
            3 * i : 3 * i + 3,
        ] = values

    N_sigma = np.concatenate(
        (
            np.eye(6),
            M,
        ),
        axis=1,
    )

    N_epsilon = (
        N_sigma.copy()
    )

    return (
        N_sigma,
        N_epsilon,
    )

import numpy as np


def project_hwt10_internal_fields(
    coords: np.ndarray,
    disp: np.ndarray,
    young: float,
    poisson: float,
):

    q = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.25,0.25,0.25],
        ],
        dtype=float,
    )

    w = np.array(
        [
            1.0/360.0,
            1.0/360.0,
            1.0/360.0,
            1.0/360.0,
            4.0/360.0,
            4.0/360.0,
            4.0/360.0,
            4.0/360.0,
            4.0/360.0,
            4.0/360.0,
            32.0/360.0,
        ],
        dtype=float,
    )

    L = np.zeros(
        (24,24),
        dtype=float,
    )

    a = np.zeros(
        24,
        dtype=float,
    )

    for xi, wi in zip(q, w):

        (
            detJ,
            gradN,
            F,
            E_green,
            B,
        ) = tet10_kinematics(
            coords,
            disp,
            xi,
        )

        (
            N_sigma,
            N_epsilon,
        ) = hwt10_mixed_interpolation(
            xi
        )

        dv = (
            wi * detJ
        )

        L += (
            -dv
            * (
                N_epsilon.T
                @ N_sigma
            )
        )

        a += (
            dv
            * (
                N_sigma.T
                @ E_green
            )
        )

    epsilon_hat = np.linalg.solve(
        L.T,
        -a,
    )

    b = np.zeros(
        24,
        dtype=float,
    )

    for xi, wi in zip(q, w):

        (
            detJ,
            gradN,
            F,
            E_green,
            B,
        ) = tet10_kinematics(
            coords,
            disp,
            xi,
        )

        (
            N_sigma,
            N_epsilon,
        ) = hwt10_mixed_interpolation(
            xi
        )

        physical_strain = (
            N_epsilon
            @ epsilon_hat
        )

        (
            W,
            stress,
            D,
        ) = neo_hooke_green_response(
            physical_strain,
            young,
            poisson,
        )

        dv = (
            wi * detJ
        )

        b += (
            dv
            * (
                N_epsilon.T
                @ stress
            )
        )

    sigma_hat = np.linalg.solve(
        L,
        -b,
    )

    return (
        L,
        epsilon_hat,
        sigma_hat,
    )

import numpy as np


def assemble_hwt10_condensed_tangent(
    coords: np.ndarray,
    disp: np.ndarray,
    young: float,
    poisson: float,
):

    (
        L,
        epsilon_hat,
        sigma_hat,
    ) = project_hwt10_internal_fields(
        coords,
        disp,
        young,
        poisson,
    )

    q = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.25,0.25,0.25],
        ],
        dtype=float,
    )

    w = np.array(
        [
            1/360,1/360,1/360,1/360,
            4/360,4/360,4/360,
            4/360,4/360,4/360,
            32/360,
        ],
        dtype=float,
    )

    G = np.zeros(
        (24,30),
        dtype=float,
    )

    H = np.zeros(
        (24,24),
        dtype=float,
    )

    K_geometric = np.zeros(
        (30,30),
        dtype=float,
    )

    for xi, wi in zip(q, w):

        (
            detJ,
            gradN,
            F,
            E_green,
            B,
        ) = tet10_kinematics(
            coords,
            disp,
            xi,
        )

        (
            N_sigma,
            N_epsilon,
        ) = hwt10_mixed_interpolation(
            xi
        )

        physical_strain = (
            N_epsilon
            @ epsilon_hat
        )

        (
            W,
            stress,
            D,
        ) = neo_hooke_green_response(
            physical_strain,
            young,
            poisson,
        )

        dv = (
            wi * detJ
        )

        G += (
            dv
            * (
                N_sigma.T
                @ B
            )
        )

        H += (
            dv
            * (
                N_epsilon.T
                @ D
                @ N_epsilon
            )
        )

        stress_h = (
            N_sigma
            @ sigma_hat
        )

        for i in range(10):

            gi = gradN[i]

            for k in range(10):

                gk = gradN[k]

                N_ik = np.array(
                    [
                        gi[0]*gk[0],
                        gi[1]*gk[1],
                        gi[2]*gk[2],
                        gi[0]*gk[1]
                        + gi[1]*gk[0],
                        gi[0]*gk[2]
                        + gi[2]*gk[0],
                        gi[1]*gk[2]
                        + gi[2]*gk[1],
                    ],
                    dtype=float,
                )

                scalar = float(
                    N_ik
                    @ stress_h
                )

                K_geometric[
                    3*i:3*i+3,
                    3*k:3*k+3,
                ] += (
                    dv
                    * scalar
                    * np.eye(3)
                )

    left = np.linalg.solve(
        L,
        H,
    )

    H_hat = np.linalg.solve(
        L,
        left.T,
    ).T

    K_material = (
        G.T
        @ H_hat
        @ G
    )

    K_tangent = (
        K_material
        + K_geometric
    )

    return (
        K_tangent,
        K_material,
        K_geometric,
    )

import numpy as np


def solve_hwt10_increment(
    coords: np.ndarray,
    disp: np.ndarray,
    young: float,
    poisson: float,
    load: np.ndarray,
    fixed_dofs: np.ndarray,
    target_dof: int,
) -> float:

    load = np.asarray(
        load,
        dtype=float,
    )

    if load.shape != (30,):
        raise ValueError(
            "load must have shape (30,)"
        )

    fixed_dofs = np.asarray(
        fixed_dofs,
        dtype=int,
    )

    if (
        fixed_dofs.ndim != 1
        or np.unique(
            fixed_dofs
        ).size
        != fixed_dofs.size
    ):
        raise ValueError(
            "invalid fixed_dofs"
        )

    if np.any(
        (fixed_dofs < 0)
        | (fixed_dofs >= 30)
    ):
        raise ValueError(
            "fixed DOF outside range"
        )

    target_dof = int(
        target_dof
    )

    if not (
        0 <= target_dof < 30
    ):
        raise ValueError(
            "invalid target_dof"
        )

    if target_dof in set(
        fixed_dofs.tolist()
    ):
        raise ValueError(
            "target DOF is fixed"
        )

    (
        K_tangent,
        K_material,
        K_geometric,
    ) = assemble_hwt10_condensed_tangent(
        coords,
        disp,
        young,
        poisson,
    )

    free_mask = np.ones(
        30,
        dtype=bool,
    )

    free_mask[
        fixed_dofs
    ] = False

    free = np.flatnonzero(
        free_mask
    )

    K_ff = K_tangent[
        np.ix_(
            free,
            free,
        )
    ]

    du = np.zeros(
        30,
        dtype=float,
    )

    du[free] = np.linalg.solve(
        K_ff,
        load[free],
    )

    return float(
        du[target_dof]
    )
SCICODE_GOLD_EOF
