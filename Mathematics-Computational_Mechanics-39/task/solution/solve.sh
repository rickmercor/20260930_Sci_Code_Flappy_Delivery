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


def dual_mortar_q4_shapes(
    xi: float,
    eta: float,
):

    x = float(xi)
    y = float(eta)

    N = 0.25 * np.array(
        [
            (1.0 - x) * (1.0 - y),
            (1.0 + x) * (1.0 - y),
            (1.0 + x) * (1.0 + y),
            (1.0 - x) * (1.0 + y),
        ],
        dtype=float,
    )

    Phi = np.array(
        [
            4.0 * N[0]
            - 2.0 * N[1]
            + N[2]
            - 2.0 * N[3],

            -2.0 * N[0]
            + 4.0 * N[1]
            - 2.0 * N[2]
            + N[3],

            N[0]
            - 2.0 * N[1]
            + 4.0 * N[2]
            - 2.0 * N[3],

            -2.0 * N[0]
            + N[1]
            - 2.0 * N[2]
            + 4.0 * N[3],
        ],
        dtype=float,
    )

    return N, Phi

import math
import numpy as np


def coupling_gauss_rule(
    fe_lengths: np.ndarray,
    mpm_spacing: np.ndarray,
):

    lfe = np.asarray(
        fe_lengths,
        dtype=float,
    )

    hmpm = np.asarray(
        mpm_spacing,
        dtype=float,
    )

    ratio = (
        np.max(lfe)
        / np.min(hmpm)
    )

    nearest_integer = np.rint(
        ratio
    )

    if abs(
        ratio
        - nearest_integer
    ) <= 1.0e-12:
        ratio = float(
            nearest_integer
        )

    n_gauss = int(
        math.floor(
            ratio
        )
        + 2
    )

    points, weights = (
        np.polynomial.legendre.leggauss(
            n_gauss
        )
    )

    return (
        n_gauss,
        points,
        weights,
    )

import numpy as np


def regular_grid_interface_shapes(
    x: float,
    y: float,
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
):

    nx = int(n_cells_x)
    ny = int(n_cells_y)

    xs = np.linspace(
        -0.5 * length_x,
        0.5 * length_x,
        nx + 1,
    )

    ys = np.linspace(
        -0.5 * length_y,
        0.5 * length_y,
        ny + 1,
    )

    i = (
        np.searchsorted(
            xs,
            float(x),
            side="right",
        )
        - 1
    )

    j = (
        np.searchsorted(
            ys,
            float(y),
            side="right",
        )
        - 1
    )

    i = min(
        max(i, 0),
        nx - 1,
    )

    j = min(
        max(j, 0),
        ny - 1,
    )

    tx = (
        (float(x) - xs[i])
        / (xs[i + 1] - xs[i])
    )

    ty = (
        (float(y) - ys[j])
        / (ys[j + 1] - ys[j])
    )

    local = np.array(
        [
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            tx * ty,
            (1.0 - tx) * ty,
        ],
        dtype=float,
    )

    active = np.array(
        [
            j * (nx + 1) + i,
            j * (nx + 1) + i + 1,
            (j + 1) * (nx + 1) + i + 1,
            (j + 1) * (nx + 1) + i,
        ],
        dtype=int,
    )

    N = np.zeros(
        (nx + 1) * (ny + 1),
        dtype=float,
    )

    N[active] = local

    return N

import numpy as np


def assemble_dual_mortar_projection(
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
):

    Lx = float(length_x)
    Ly = float(length_y)

    nx = int(n_cells_x)
    ny = int(n_cells_y)

    spacing = np.array(
        [
            Lx / nx,
            Ly / ny,
        ],
        dtype=float,
    )

    (
        n_gauss,
        points,
        weights,
    ) = coupling_gauss_rule(
        np.array(
            [Lx, Ly],
            dtype=float,
        ),
        spacing,
    )

    n2 = (
        (nx + 1)
        * (ny + 1)
    )

    D = np.zeros(
        (4, 4),
        dtype=float,
    )

    A = np.zeros(
        (4, n2),
        dtype=float,
    )

    jacobian = (
        Lx
        * Ly
        / 4.0
    )

    for i in range(
        n_gauss
    ):

        xi = points[i]

        for j in range(
            n_gauss
        ):

            eta = points[j]

            Nfe, Phi = (
                dual_mortar_q4_shapes(
                    xi,
                    eta,
                )
            )

            x = (
                0.5
                * Lx
                * xi
            )

            y = (
                0.5
                * Ly
                * eta
            )

            Nmpm = (
                regular_grid_interface_shapes(
                    x,
                    y,
                    Lx,
                    Ly,
                    nx,
                    ny,
                )
            )

            weight = (
                weights[i]
                * weights[j]
                * jacobian
            )

            D += (
                weight
                * np.outer(
                    Phi,
                    Nfe,
                )
            )

            A += (
                weight
                * np.outer(
                    Phi,
                    Nmpm,
                )
            )

    P = np.linalg.solve(
        D,
        A,
    )

    return (
        D,
        A,
        P,
        n_gauss,
    )

import numpy as np


def condense_dual_mortar_newton(
    P: np.ndarray,
    Knn: np.ndarray,
    K11: np.ndarray,
    K22: np.ndarray,
    Kn1: np.ndarray,
    Kn2: np.ndarray,
    Rn: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
):

    P = np.asarray(
        P,
        dtype=float,
    )

    Knn = np.asarray(
        Knn,
        dtype=float,
    )

    K11 = np.asarray(
        K11,
        dtype=float,
    )

    K22 = np.asarray(
        K22,
        dtype=float,
    )

    Kn1 = np.asarray(
        Kn1,
        dtype=float,
    )

    Kn2 = np.asarray(
        Kn2,
        dtype=float,
    )

    top_right = (
        Kn2
        + Kn1 @ P
    )

    bottom_left = (
        Kn2.T
        + P.T @ Kn1.T
    )

    bottom_right = (
        K22
        + P.T @ K11 @ P
    )

    K_reduced = np.block(
        [
            [
                Knn,
                top_right,
            ],
            [
                bottom_left,
                bottom_right,
            ],
        ]
    )

    R_reduced = np.concatenate(
        (
            np.asarray(
                Rn,
                dtype=float,
            ),
            np.asarray(
                R2,
                dtype=float,
            )
            - P.T
            @ np.asarray(
                R1,
                dtype=float,
            ),
        )
    )

    return (
        K_reduced,
        R_reduced,
    )

import numpy as np


def solve_dual_mortar_increment(
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
    Knn: np.ndarray,
    K11: np.ndarray,
    K22: np.ndarray,
    Kn1: np.ndarray,
    Kn2: np.ndarray,
    Rn: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
):

    (
        D,
        A,
        P,
        n_gauss,
    ) = assemble_dual_mortar_projection(
        length_x,
        length_y,
        n_cells_x,
        n_cells_y,
    )

    (
        K_reduced,
        R_reduced,
    ) = condense_dual_mortar_newton(
        P,
        Knn,
        K11,
        K22,
        Kn1,
        Kn2,
        Rn,
        R1,
        R2,
    )

    delta_reduced = np.linalg.solve(
        K_reduced,
        -R_reduced,
    )

    n = np.asarray(Knn).shape[0]

    delta_n = (
        delta_reduced[:n]
        .copy()
    )

    delta_2 = (
        delta_reduced[n:]
        .copy()
    )

    delta_1 = (
        P @ delta_2
    )

    residual_inf = float(
        np.max(
            np.abs(
                K_reduced
                @ delta_reduced
                + R_reduced
            )
        )
    )

    return (
        delta_n,
        delta_1,
        delta_2,
        residual_inf,
    )

import numpy as np


def dual_mortar_target_increment(
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
    Knn: np.ndarray,
    K11: np.ndarray,
    K22: np.ndarray,
    Kn1: np.ndarray,
    Kn2: np.ndarray,
    Rn: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    target_nonmortar_dof: int,
) -> float:

    (
        delta_n,
        delta_1,
        delta_2,
        residual_inf,
    ) = solve_dual_mortar_increment(
        length_x,
        length_y,
        n_cells_x,
        n_cells_y,
        Knn,
        K11,
        K22,
        Kn1,
        Kn2,
        Rn,
        R1,
        R2,
    )

    return float(
        delta_1[
            int(target_nonmortar_dof)
        ]
    )
SCICODE_GOLD_EOF
