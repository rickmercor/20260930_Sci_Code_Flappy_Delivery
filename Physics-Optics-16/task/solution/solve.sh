#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def reciprocal_fourier(
    harmonics: np.ndarray,
    fill: float,
    offset: float,
    epsilon_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    harmonics = np.asarray(harmonics)

    if (
        harmonics.ndim != 1
        or harmonics.size == 0
        or not np.issubdtype(harmonics.dtype, np.integer)
    ):
        raise ValueError("harmonics must be a nonempty integer vector")

    parameters = np.asarray([fill, offset, epsilon_high])

    if (
        not np.isrealobj(parameters)
        or not np.all(np.isfinite(parameters))
        or not 0 < fill < 1
        or epsilon_high <= 0
    ):
        raise ValueError("invalid layer parameters")

    offset = float(np.remainder(offset, 1.0))

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        reciprocal_high = np.divide(1.0, np.float64(epsilon_high))

    if not np.isfinite(reciprocal_high):
        raise ValueError(
            "epsilon_high is outside the representable reciprocal domain"
        )

    orders = harmonics[:, None] - harmonics[None, :]
    contrast = reciprocal_high - 1.0

    phase = np.exp(-2j * np.pi * orders * (offset + fill / 2))

    coefficients = (
        (orders == 0)
        + contrast * fill * np.sinc(orders * fill) * phase
    )

    tangent = contrast * np.exp(
        -2j * np.pi * orders * (offset + fill)
    )

    return (
        coefficients.astype(np.complex128),
        tangent.astype(np.complex128),
    )

import numpy as np
from scipy.linalg import solve


def _constant(value):
    array = np.asarray(value, dtype=np.complex128)
    return array, np.zeros_like(array)


def _subtract(left, right):
    return left[0] - right[0], left[1] - right[1]


def _multiply(left, right):
    return (
        left[0] @ right[0],
        left[1] @ right[0] + left[0] @ right[1],
    )


def _solve(left, right):
    value = solve(left[0], right[0])
    tangent = solve(
        left[0],
        right[1] - left[1] @ value,
    )
    return value, tangent


def tm_operators(
    coefficients: np.ndarray,
    tangent: np.ndarray,
    kappa: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return P, dP, Q, dQ, PQ, d(PQ), each of shape (N, N).

    C and dC are finite arrays with consistent shapes.
    C is nonsingular, and kappa is a real vector of shape (N,).
    kappa is held fixed under differentiation.
    """
    identity = _constant(np.eye(len(kappa)))
    wavevector = _constant(np.diag(kappa))

    inverse = _solve(
        (coefficients, tangent),
        identity,
    )

    coupling = _subtract(
        _multiply(
            _multiply(
                wavevector,
                (coefficients, tangent),
            ),
            wavevector,
        ),
        identity,
    )

    product = _multiply(inverse, coupling)

    return (
        inverse[0],
        inverse[1],
        coupling[0],
        coupling[1],
        product[0],
        product[1],
    )

def rotated_root(
    product: np.ndarray,
    tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float]:
    import numpy as np
    from scipy.linalg import solve

    def constant(value):
        array = np.asarray(value, dtype=np.complex128)
        return array, np.zeros_like(array)

    def add(left, right):
        return left[0] + right[0], left[1] + right[1]

    def multiply(left, right):
        return (
            left[0] @ right[0],
            left[1] @ right[0] + left[0] @ right[1],
        )

    def scale(factor, pair):
        return factor * pair[0], factor * pair[1]

    def solve_pair(left, right):
        value = solve(left[0], right[0])
        derivative = solve(
            left[0],
            right[1] - left[1] @ value,
        )
        return value, derivative

    identity = constant(
        np.eye(product.shape[0], dtype=np.complex128)
    )

    phi = -0.25 * np.pi
    root_pair = scale(
        np.exp(1j * phi),
        (product, tangent),
    )
    inverse_pair = identity

    rho_after_two = None
    rho_tangent_after_two = None

    for updates in range(51):
        iterate = multiply(inverse_pair, root_pair)

        residual = float(
            np.linalg.norm(identity[0] - iterate[0], ord="fro")
        )

        if updates == 2:
            residual_matrix = identity[0] - iterate[0]
            residual_tangent = -iterate[1]
            rho_after_two = float(
                np.linalg.norm(residual_matrix, ord="fro")
            )

            if rho_after_two == 0.0:
                raise RuntimeError(
                    "two-update residual norm is zero"
                )

            rho_tangent_after_two = float(
                np.real(
                    np.vdot(
                        residual_matrix,
                        residual_tangent,
                    )
                )
                / rho_after_two
            )

        if residual < 1e-13:
            if (
                rho_after_two is None
                or rho_tangent_after_two is None
            ):
                raise RuntimeError(
                    "iteration converged before two-update "
                    "diagnostics were available"
                )

            root, root_tangent = scale(
                np.exp(-0.5j * phi),
                root_pair,
            )

            return (
                root,
                root_tangent,
                updates,
                residual,
                rho_after_two,
                rho_tangent_after_two,
            )

        if updates == 50:
            break

        squared = multiply(iterate, iterate)
        cubed = multiply(squared, iterate)

        numerator = add(
            add(
                scale(7, identity),
                scale(35, iterate),
            ),
            add(
                scale(21, squared),
                cubed,
            ),
        )

        denominator = add(
            add(
                identity,
                scale(21, iterate),
            ),
            add(
                scale(35, squared),
                scale(7, cubed),
            ),
        )

        correction = solve_pair(denominator, numerator)
        root_pair = multiply(root_pair, correction)
        inverse_pair = multiply(correction, inverse_pair)

    raise RuntimeError(
        "rational square root did not converge in 50 updates"
    )

import numpy as np
from scipy.linalg import solve


def _right_solve(right, left):
    value = solve(left[0].T, right[0].T).T
    tangent = solve(
        left[0].T,
        (right[1] - value @ left[1]).T,
    ).T
    return value, tangent


def transformed_modal(
    coupling: np.ndarray,
    coupling_tangent: np.ndarray,
    root: np.ndarray,
    root_tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _right_solve(
        (coupling, coupling_tangent),
        (root, root_tangent),
    )

import numpy as np
from scipy.linalg import expm_frechet


def layer_propagation(
    root: np.ndarray,
    tangent: np.ndarray,
    depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isreal(depth) or not np.isfinite(depth) or depth < 0:
        raise ValueError(
            "depth must be real, finite and nonnegative"
        )

    return expm_frechet(-depth * root, -depth * tangent)

import numpy as np
from scipy.linalg import solve


def _constant(value):
    array = np.asarray(value, dtype=np.complex128)
    return array, np.zeros_like(array)


def _add(left, right):
    return left[0] + right[0], left[1] + right[1]


def _subtract(left, right):
    return left[0] - right[0], left[1] - right[1]


def _multiply(left, right):
    return (
        left[0] @ right[0],
        left[1] @ right[0] + left[0] @ right[1],
    )


def _solve(left, right):
    value = solve(left[0], right[0])
    tangent = solve(left[0], right[1] - left[1] @ value)
    return value, tangent


def layer_scattering(
    modal: np.ndarray,
    modal_tangent: np.ndarray,
    propagation: np.ndarray,
    propagation_tangent: np.ndarray,
    vacuum_root: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    identity = _constant(np.eye(len(vacuum_root)))

    impedance = _solve(
        (modal, modal_tangent),
        _constant(np.diag(vacuum_root)),
    )

    plus = _add(identity, impedance)
    minus = _subtract(identity, impedance)
    propagation_pair = (propagation, propagation_tangent)

    intermediate = _multiply(
        _multiply(propagation_pair, minus),
        _solve(plus, propagation_pair),
    )

    denominator = _subtract(
        plus,
        _multiply(intermediate, minus),
    )

    reflection = _solve(
        denominator,
        _subtract(_multiply(intermediate, plus), minus),
    )

    transmission = _solve(
        denominator,
        _multiply(
            propagation_pair,
            _subtract(
                plus,
                _multiply(minus, _solve(plus, minus)),
            ),
        ),
    )

    scattering = np.array(
        [
            [reflection[0], transmission[0]],
            [transmission[0], reflection[0]],
        ]
    )

    tangent = np.array(
        [
            [reflection[1], transmission[1]],
            [transmission[1], reflection[1]],
        ]
    )

    return scattering, tangent

import numpy as np
from scipy.linalg import solve


def _constant(value):
    array = np.asarray(value, dtype=np.complex128)
    return array, np.zeros_like(array)


def _add(left, right):
    return left[0] + right[0], left[1] + right[1]


def _subtract(left, right):
    return left[0] - right[0], left[1] - right[1]


def _multiply(left, right):
    return (
        left[0] @ right[0],
        left[1] @ right[0] + left[0] @ right[1],
    )


def _solve(left, right):
    value = solve(left[0], right[0])
    tangent = solve(left[0], right[1] - left[1] @ value)
    return value, tangent


def redheffer_compose(
    left: np.ndarray,
    left_tangent: np.ndarray,
    right: np.ndarray,
    right_tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_blocks = [
        [
            (left[row, column], left_tangent[row, column])
            for column in range(2)
        ]
        for row in range(2)
    ]

    right_blocks = [
        [
            (right[row, column], right_tangent[row, column])
            for column in range(2)
        ]
        for row in range(2)
    ]

    identity = _constant(np.eye(left.shape[-1]))

    feedback = _subtract(
        identity,
        _multiply(left_blocks[1][1], right_blocks[0][0]),
    )

    internal_left = _solve(
        feedback,
        left_blocks[1][0],
    )

    internal_right = _solve(
        feedback,
        _multiply(left_blocks[1][1], right_blocks[0][1]),
    )

    result = [
        [
            _add(
                left_blocks[0][0],
                _multiply(
                    _multiply(
                        left_blocks[0][1],
                        right_blocks[0][0],
                    ),
                    internal_left,
                ),
            ),
            _multiply(
                left_blocks[0][1],
                _add(
                    right_blocks[0][1],
                    _multiply(
                        right_blocks[0][0],
                        internal_right,
                    ),
                ),
            ),
        ],
        [
            _multiply(
                right_blocks[1][0],
                internal_left,
            ),
            _add(
                right_blocks[1][1],
                _multiply(
                    right_blocks[1][0],
                    internal_right,
                ),
            ),
        ],
    ]

    scattering = np.array(
        [[pair[0] for pair in row] for row in result]
    )
    tangent = np.array(
        [[pair[1] for pair in row] for row in result]
    )

    return scattering, tangent

def solve_sensitivity(fill_first: float = 0.43) -> np.ndarray:
    global np, solve, expm_frechet
    import numpy as np
    from scipy.linalg import expm_frechet, solve

    harmonics = np.arange(-3, 4)
    kappa = 0.23 + 0.76 * harmonics
    vacuum_root = np.sqrt(
        (kappa ** 2 - 1).astype(np.complex128)
    )
    layers = [
        (fill_first, 0.0, 4.0, 3.1),
        (0.57, 0.19, 2.89, 2.4),
    ]

    scattering_pairs = []
    update_counts = []
    primary_rho = None
    primary_sensitivity = None

    for layer_index, (
        fill,
        offset,
        epsilon_high,
        depth,
    ) in enumerate(layers):
        coefficients, coefficient_tangent = (
            reciprocal_fourier(
                harmonics,
                fill,
                offset,
                epsilon_high,
            )
        )

        if layer_index != 0:
            coefficient_tangent = np.zeros_like(
                coefficient_tangent
            )

        operators = tm_operators(
            coefficients,
            coefficient_tangent,
            kappa,
        )

        (
            root,
            root_tangent,
            updates,
            _,
            rho_two,
            rho_tangent_two,
        ) = rotated_root(
            operators[4],
            operators[5],
        )

        if layer_index == 0:
            primary_rho = rho_two
            primary_sensitivity = rho_tangent_two

        modal = transformed_modal(
            operators[2],
            operators[3],
            root,
            root_tangent,
        )
        propagation = layer_propagation(
            root,
            root_tangent,
            depth,
        )
        scattering_pairs.append(
            layer_scattering(
                *modal,
                *propagation,
                vacuum_root,
            )
        )
        update_counts.append(updates)

    scattering, tangent = redheffer_compose(
        *scattering_pairs[0],
        *scattering_pairs[1],
    )

    transmission = scattering[1, 0, :, 3]
    transmission_tangent = tangent[1, 0, :, 3]
    reflection = scattering[0, 0, :, 3]
    reflection_tangent = tangent[0, 0, :, 3]

    propagating = np.abs(kappa) < 1
    weights = np.zeros_like(kappa)
    weights[propagating] = (
        np.sqrt(1 - kappa[propagating] ** 2)
        / np.sqrt(1 - kappa[3] ** 2)
    )

    transmitted = weights * np.abs(transmission) ** 2
    reflected = weights * np.abs(reflection) ** 2

    transmitted_tangent = (
        2
        * weights
        * np.real(
            np.conj(transmission) * transmission_tangent
        )
    )
    reflected_tangent = (
        2
        * weights
        * np.real(
            np.conj(reflection) * reflection_tangent
        )
    )

    return np.array(
        [
            primary_sensitivity,
            primary_rho,
            transmitted_tangent[4],
            transmitted[4],
            reflected.sum(),
            transmitted.sum(),
            *update_counts,
            (
                transmitted_tangent
                + reflected_tangent
            ).sum(),
        ],
        dtype=np.float64,
    )
SCICODE_GOLD_EOF
