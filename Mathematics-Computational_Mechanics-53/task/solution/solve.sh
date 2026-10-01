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


def transform_orthotropic_ply(
    theta_deg: float,
    E1: float,
    E2: float,
    nu12: float,
    G12: float,
    G13: float,
    G23: float,
    alpha1: float,
    alpha2: float,
):

    nu21 = nu12 * E2 / E1

    den = (
        1.0
        - nu12 * nu21
    )

    Q11 = E1 / den
    Q22 = E2 / den
    Q12 = nu12 * E2 / den
    Q66 = G12

    theta = np.deg2rad(
        float(theta_deg)
    )

    m = np.cos(theta)
    n = np.sin(theta)

    m2 = m * m
    n2 = n * n

    m4 = m2 * m2
    n4 = n2 * n2

    Qbar_b = np.zeros(
        (3, 3),
        dtype=float,
    )

    Qbar_b[0, 0] = (
        Q11 * m4
        + Q22 * n4
        + 2.0
        * (
            Q12
            + 2.0 * Q66
        )
        * m2 * n2
    )

    Qbar_b[1, 1] = (
        Q11 * n4
        + Q22 * m4
        + 2.0
        * (
            Q12
            + 2.0 * Q66
        )
        * m2 * n2
    )

    Qbar_b[0, 1] = (
        (
            Q11
            + Q22
            - 4.0 * Q66
        )
        * m2 * n2
        + Q12
        * (
            m4 + n4
        )
    )

    Qbar_b[1, 0] = (
        Qbar_b[0, 1]
    )

    Qbar_b[0, 2] = (
        (
            Q11
            - Q12
            - 2.0 * Q66
        )
        * m**3 * n
        -
        (
            Q22
            - Q12
            - 2.0 * Q66
        )
        * m * n**3
    )

    Qbar_b[2, 0] = (
        Qbar_b[0, 2]
    )

    Qbar_b[1, 2] = (
        (
            Q11
            - Q12
            - 2.0 * Q66
        )
        * m * n**3
        -
        (
            Q22
            - Q12
            - 2.0 * Q66
        )
        * m**3 * n
    )

    Qbar_b[2, 1] = (
        Qbar_b[1, 2]
    )

    Qbar_b[2, 2] = (
        (
            Q11
            + Q22
            - 2.0 * Q12
            - 2.0 * Q66
        )
        * m2 * n2
        + Q66
        * (
            m4 + n4
        )
    )

    Qbar_s = np.array(
        [
            [
                G13 * m2
                + G23 * n2,
                (
                    G13 - G23
                )
                * m * n,
            ],
            [
                (
                    G13 - G23
                )
                * m * n,
                G13 * n2
                + G23 * m2,
            ],
        ],
        dtype=float,
    )

    alpha_bar = np.array(
        [
            alpha1 * m2
            + alpha2 * n2,
            alpha1 * n2
            + alpha2 * m2,
            2.0
            * (
                alpha1 - alpha2
            )
            * m * n,
        ],
        dtype=float,
    )

    return (
        Qbar_b,
        Qbar_s,
        alpha_bar,
    )

import numpy as np


def build_symmetric_laminate(
    angles_deg: np.ndarray,
    ply_thickness: float,
    E1: float,
    E2: float,
    nu12: float,
    G12: float,
    G13: float,
    G23: float,
    alpha1: float,
    alpha2: float,
    delta_T: float,
    shear_correction: float = 5.0 / 6.0,
):

    angles_deg = np.asarray(
        angles_deg,
        dtype=float,
    )

    n_ply = angles_deg.size

    if (
        angles_deg.ndim != 1
        or n_ply == 0
    ):
        raise ValueError(
            "angles_deg must be a nonempty vector"
        )

    if not np.array_equal(
        angles_deg,
        angles_deg[::-1],
    ):
        raise ValueError(
            "stacking sequence must be symmetric"
        )

    h = (
        n_ply
        * float(ply_thickness)
    )

    z = np.linspace(
        -0.5 * h,
        0.5 * h,
        n_ply + 1,
    )

    A = np.zeros(
        (3, 3),
        dtype=float,
    )

    B = np.zeros(
        (3, 3),
        dtype=float,
    )

    D = np.zeros(
        (3, 3),
        dtype=float,
    )

    As = np.zeros(
        (2, 2),
        dtype=float,
    )

    NT = np.zeros(
        3,
        dtype=float,
    )

    MT = np.zeros(
        3,
        dtype=float,
    )

    for k, angle in enumerate(
        angles_deg
    ):

        (
            Qbar_b,
            Qbar_s,
            alpha_bar,
        ) = transform_orthotropic_ply(
            angle,
            E1,
            E2,
            nu12,
            G12,
            G13,
            G23,
            alpha1,
            alpha2,
        )

        z0 = z[k]
        z1 = z[k + 1]

        dz = z1 - z0

        A += (
            Qbar_b
            * dz
        )

        B += (
            0.5
            * Qbar_b
            * (
                z1**2
                - z0**2
            )
        )

        D += (
            Qbar_b
            * (
                z1**3
                - z0**3
            )
            / 3.0
        )

        As += (
            shear_correction
            * Qbar_s
            * dz
        )

        thermal = (
            Qbar_b
            @ alpha_bar
            * delta_T
        )

        NT += (
            thermal
            * dz
        )

        MT += (
            0.5
            * thermal
            * (
                z1**2
                - z0**2
            )
        )

    return (
        A,
        B,
        D,
        As,
        NT,
        MT,
    )

import numpy as np


def bw_basis_derivatives(
    x: float,
    y: float,
) -> np.ndarray:

    exponents = (
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
        (3, 1),
        (1, 3),
    )

    derivative_orders = (
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
        (4, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 4),
    )

    out = np.zeros(
        (15, 12),
        dtype=float,
    )

    for row, (dx, dy) in enumerate(
        derivative_orders
    ):

        for col, (px, py) in enumerate(
            exponents
        ):

            if (
                px < dx
                or py < dy
            ):
                continue

            coeff = 1.0

            for k in range(dx):
                coeff *= (
                    px - k
                )

            for k in range(dy):
                coeff *= (
                    py - k
                )

            out[row, col] = (
                coeff
                * x**(px - dx)
                * y**(py - dy)
            )

    return out

import numpy as np


def q4bw_bending_shapes(
    a: float,
    b: float,
    D: np.ndarray,
    As: np.ndarray,
    x: float,
    y: float,
):

    D = np.asarray(
        D,
        dtype=float,
    )

    As = np.asarray(
        As,
        dtype=float,
    )

    As_inv = np.linalg.solve(
        As,
        np.eye(2),
    )

    def reduced_rows(
        data: np.ndarray,
    ):

        (
            H,
            Hx,
            Hy,
            Hxx,
            Hxy,
            Hyy,
            Hxxx,
            Hxxy,
            Hxyy,
            Hyyy,
            Hxxxx,
            Hxxxy,
            Hxxyy,
            Hxyyy,
            Hyyyy,
        ) = data

        M1 = (
            D[0]
            @ np.vstack(
                (
                    Hxxx,
                    Hxyy,
                    2.0 * Hxxy,
                )
            )
            +
            D[2]
            @ np.vstack(
                (
                    Hxxy,
                    Hyyy,
                    2.0 * Hxyy,
                )
            )
        )

        M2 = (
            D[1]
            @ np.vstack(
                (
                    Hxxy,
                    Hyyy,
                    2.0 * Hxyy,
                )
            )
            +
            D[2]
            @ np.vstack(
                (
                    Hxxx,
                    Hxyy,
                    2.0 * Hxxy,
                )
            )
        )

        M1x = (
            D[0]
            @ np.vstack(
                (
                    Hxxxx,
                    Hxxyy,
                    2.0 * Hxxxy,
                )
            )
            +
            D[2]
            @ np.vstack(
                (
                    Hxxxy,
                    Hxyyy,
                    2.0 * Hxxyy,
                )
            )
        )

        M2x = (
            D[1]
            @ np.vstack(
                (
                    Hxxxy,
                    Hxyyy,
                    2.0 * Hxxyy,
                )
            )
            +
            D[2]
            @ np.vstack(
                (
                    Hxxxx,
                    Hxxyy,
                    2.0 * Hxxxy,
                )
            )
        )

        M1y = (
            D[0]
            @ np.vstack(
                (
                    Hxxxy,
                    Hxyyy,
                    2.0 * Hxxyy,
                )
            )
            +
            D[2]
            @ np.vstack(
                (
                    Hxxyy,
                    Hyyyy,
                    2.0 * Hxyyy,
                )
            )
        )

        M2y = (
            D[1]
            @ np.vstack(
                (
                    Hxxyy,
                    Hyyyy,
                    2.0 * Hxyyy,
                )
            )
            +
            D[2]
            @ np.vstack(
                (
                    Hxxxy,
                    Hxyyy,
                    2.0 * Hxxyy,
                )
            )
        )

        Hphiy = (
            -Hx
            - As_inv[0, 0] * M1
            - As_inv[0, 1] * M2
        )

        Hphix = (
            Hy
            + As_inv[1, 0] * M1
            + As_inv[1, 1] * M2
        )

        Hphiy_x = (
            -Hxx
            - As_inv[0, 0] * M1x
            - As_inv[0, 1] * M2x
        )

        Hphiy_y = (
            -Hxy
            - As_inv[0, 0] * M1y
            - As_inv[0, 1] * M2y
        )

        Hphix_x = (
            Hxy
            + As_inv[1, 0] * M1x
            + As_inv[1, 1] * M2x
        )

        Hphix_y = (
            Hyy
            + As_inv[1, 0] * M1y
            + As_inv[1, 1] * M2y
        )

        return (
            Hphiy,
            Hphix,
            Hphiy_x,
            Hphiy_y,
            Hphix_x,
            Hphix_y,
        )

    nodes = np.array(
        [
            [-0.5*a, -0.5*b],
            [ 0.5*a, -0.5*b],
            [ 0.5*a,  0.5*b],
            [-0.5*a,  0.5*b],
        ],
        dtype=float,
    )

    H_i = np.zeros(
        (12, 12),
        dtype=float,
    )

    for node, (
        xn,
        yn,
    ) in enumerate(nodes):

        data_n = (
            bw_basis_derivatives(
                xn,
                yn,
            )
        )

        (
            Hphiy_n,
            Hphix_n,
            _,
            _,
            _,
            _,
        ) = reduced_rows(
            data_n
        )

        H_i[
            3*node
        ] = data_n[0]

        H_i[
            3*node + 1
        ] = Hphiy_n

        H_i[
            3*node + 2
        ] = Hphix_n

    data = (
        bw_basis_derivatives(
            x,
            y,
        )
    )

    (
        Hphiy,
        Hphix,
        Hphiy_x,
        Hphiy_y,
        Hphix_x,
        Hphix_y,
    ) = reduced_rows(data)

    def interpolate(
        row: np.ndarray,
    ) -> np.ndarray:

        return np.linalg.solve(
            H_i.T,
            row,
        )

    Nw = interpolate(
        data[0]
    )

    Nwx = interpolate(
        data[1]
    )

    Nwy = interpolate(
        data[2]
    )

    Nphiy = interpolate(
        Hphiy
    )

    Nphix = interpolate(
        Hphix
    )

    Nphiy_x = interpolate(
        Hphiy_x
    )

    Nphiy_y = interpolate(
        Hphiy_y
    )

    Nphix_x = interpolate(
        Hphix_x
    )

    Nphix_y = interpolate(
        Hphix_y
    )

    Bkappa = np.vstack(
        (
            Nphiy_x,
            -Nphix_y,
            Nphiy_y
            - Nphix_x,
        )
    )

    Bgamma = np.vstack(
        (
            Nwx + Nphiy,
            Nwy - Nphix,
        )
    )

    return (
        Nw,
        Nwx,
        Nwy,
        Nphiy,
        Nphix,
        Bkappa,
        Bgamma,
    )

import numpy as np


def q4bw_point_kinematics(
    a: float,
    b: float,
    D: np.ndarray,
    As: np.ndarray,
    q: np.ndarray,
    x: float,
    y: float,
):

    q = np.asarray(
        q,
        dtype=float,
    )

    wb = q[:12]
    wm = q[12:]

    (
        Nw,
        Nwx,
        Nwy,
        Nphiy,
        Nphix,
        Bkappa,
        Bgamma,
    ) = q4bw_bending_shapes(
        a,
        b,
        D,
        As,
        x,
        y,
    )

    xi = (
        2.0 * x / a
    )

    eta = (
        2.0 * y / b
    )

    dN_dxi = np.array(
        [
            -0.25*(1.0-eta),
             0.25*(1.0-eta),
             0.25*(1.0+eta),
            -0.25*(1.0+eta),
        ],
        dtype=float,
    )

    dN_deta = np.array(
        [
            -0.25*(1.0-xi),
            -0.25*(1.0+xi),
             0.25*(1.0+xi),
             0.25*(1.0-xi),
        ],
        dtype=float,
    )

    dNdx = (
        2.0
        * dN_dxi
        / a
    )

    dNdy = (
        2.0
        * dN_deta
        / b
    )

    Bm = np.zeros(
        (3, 8),
        dtype=float,
    )

    for node in range(4):

        Bm[
            0,
            2*node,
        ] = dNdx[node]

        Bm[
            1,
            2*node + 1,
        ] = dNdy[node]

        Bm[
            2,
            2*node,
        ] = dNdy[node]

        Bm[
            2,
            2*node + 1,
        ] = dNdx[node]

    wx = float(
        Nwx @ wb
    )

    wy = float(
        Nwy @ wb
    )

    epsilon_linear = (
        Bm @ wm
    )

    epsilon_nonlinear = np.array(
        [
            0.5 * wx * wx,
            0.5 * wy * wy,
            wx * wy,
        ],
        dtype=float,
    )

    epsilon0 = (
        epsilon_linear
        + epsilon_nonlinear
    )

    kappa = (
        Bkappa @ wb
    )

    gamma = (
        Bgamma @ wb
    )

    Bepsilon = np.zeros(
        (3, 20),
        dtype=float,
    )

    Bepsilon[
        :,
        12:,
    ] = Bm

    Bepsilon[
        0,
        :12,
    ] = wx * Nwx

    Bepsilon[
        1,
        :12,
    ] = wy * Nwy

    Bepsilon[
        2,
        :12,
    ] = (
        wy * Nwx
        + wx * Nwy
    )

    Bkappa_full = np.zeros(
        (3, 20),
        dtype=float,
    )

    Bkappa_full[
        :,
        :12,
    ] = Bkappa

    Bgamma_full = np.zeros(
        (2, 20),
        dtype=float,
    )

    Bgamma_full[
        :,
        :12,
    ] = Bgamma

    return (
        epsilon0,
        kappa,
        gamma,
        Bepsilon,
        Bkappa_full,
        Bgamma_full,
    )

import numpy as np


def assemble_q4bw_tangent(
    a: float,
    b: float,
    A: np.ndarray,
    D: np.ndarray,
    As: np.ndarray,
    NT: np.ndarray,
    MT: np.ndarray,
    q: np.ndarray,
):

    q = np.asarray(
        q,
        dtype=float,
    )

    gp = np.sqrt(
        3.0 / 5.0
    )

    gauss = np.array(
        [-gp, 0.0, gp],
        dtype=float,
    )

    weights = np.array(
        [
            5.0 / 9.0,
            8.0 / 9.0,
            5.0 / 9.0,
        ],
        dtype=float,
    )

    residual = np.zeros(
        20,
        dtype=float,
    )

    K = np.zeros(
        (20, 20),
        dtype=float,
    )

    potential = 0.0

    jacobian = (
        a * b / 4.0
    )

    for i, xi in enumerate(
        gauss
    ):

        for j, eta in enumerate(
            gauss
        ):

            x = (
                0.5 * a * xi
            )

            y = (
                0.5 * b * eta
            )

            weight = (
                weights[i]
                * weights[j]
                * jacobian
            )

            (
                epsilon0,
                kappa,
                gamma,
                Bepsilon,
                Bkappa,
                Bgamma,
            ) = q4bw_point_kinematics(
                a,
                b,
                D,
                As,
                q,
                x,
                y,
            )

            N = (
                A @ epsilon0
                - NT
            )

            M = (
                D @ kappa
                - MT
            )

            Q = (
                As @ gamma
            )

            residual += (
                weight
                * (
                    Bepsilon.T @ N
                    + Bkappa.T @ M
                    + Bgamma.T @ Q
                )
            )

            K_local = (
                Bepsilon.T
                @ A
                @ Bepsilon
                +
                Bkappa.T
                @ D
                @ Bkappa
                +
                Bgamma.T
                @ As
                @ Bgamma
            )

            (
                Nw,
                Nwx,
                Nwy,
                Nphiy,
                Nphix,
                Bkappa_b,
                Bgamma_b,
            ) = q4bw_bending_shapes(
                a,
                b,
                D,
                As,
                x,
                y,
            )

            K_geo = np.zeros(
                (20, 20),
                dtype=float,
            )

            K_geo[
                :12,
                :12,
            ] = (
                N[0]
                * np.outer(
                    Nwx,
                    Nwx,
                )
                +
                N[1]
                * np.outer(
                    Nwy,
                    Nwy,
                )
                +
                N[2]
                * (
                    np.outer(
                        Nwx,
                        Nwy,
                    )
                    +
                    np.outer(
                        Nwy,
                        Nwx,
                    )
                )
            )

            K += (
                weight
                * (
                    K_local
                    + K_geo
                )
            )

            potential += (
                weight
                * (
                    0.5
                    * epsilon0
                    @ A
                    @ epsilon0
                    +
                    0.5
                    * kappa
                    @ D
                    @ kappa
                    +
                    0.5
                    * gamma
                    @ As
                    @ gamma
                    -
                    epsilon0
                    @ NT
                    -
                    kappa
                    @ MT
                )
            )

    return (
        residual,
        K,
        float(potential),
    )

import numpy as np


def solve_q4bw_increment(
    a: float,
    b: float,
    A: np.ndarray,
    D: np.ndarray,
    As: np.ndarray,
    NT: np.ndarray,
    MT: np.ndarray,
    q: np.ndarray,
    delta_p: np.ndarray,
    fixed_dofs: np.ndarray,
    target_dof: int,
) -> float:

    delta_p = np.asarray(
        delta_p,
        dtype=float,
    )

    fixed_dofs = np.asarray(
        fixed_dofs,
        dtype=int,
    )

    (
        residual,
        K,
        potential,
    ) = assemble_q4bw_tangent(
        a,
        b,
        A,
        D,
        As,
        NT,
        MT,
        q,
    )

    free_mask = np.ones(
        20,
        dtype=bool,
    )

    free_mask[
        fixed_dofs
    ] = False

    free = np.flatnonzero(
        free_mask
    )

    if target_dof not in free:
        raise ValueError(
            "target_dof must be free"
        )

    Kff = K[
        np.ix_(
            free,
            free,
        )
    ]

    dq = np.zeros(
        20,
        dtype=float,
    )

    dq[free] = np.linalg.solve(
        Kff,
        delta_p[free],
    )

    return float(
        dq[target_dof]
    )
SCICODE_GOLD_EOF
