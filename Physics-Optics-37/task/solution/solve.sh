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


def PadeCoeff(P, t):
    """Return high-precision Pade partial fractions without extra packages.

    Decimal real/imaginary pairs retain internal precision. Taylor
    coefficients satisfy (1+x)*f'' + f'/2 + t**2*f/4 = 0, with
    f(0) = 1 and f'(0) = 1j*t/2. The normalized denominator is obtained
    by Taylor matching, then its simple poles are refined with Newton.
    """
    from decimal import Decimal, localcontext

    if (
        isinstance(P, (bool, np.bool_))
        or not isinstance(P, (int, np.integer))
        or P < 1
    ):
        raise ValueError("P must be a positive integer, not a boolean.")
    raw_parameter = np.asarray(t)
    if raw_parameter.ndim != 0 or raw_parameter.dtype.kind not in "iufc":
        raise ValueError("t must be a finite real or complex scalar.")
    try:
        parameter = complex(raw_parameter)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "t must be representable as a complex scalar."
        ) from error
    if not np.isfinite(parameter):
        raise ValueError("t must be finite.")

    order = int(P)
    if parameter == 0:
        poles = np.zeros(order, dtype=complex)
        residues = np.zeros(order + 1, dtype=complex)
        residues[0] = 1.0
        return poles, residues

    with localcontext() as context:
        context.prec = max(80, 12 * order)

        def number(real=0, imaginary=0):
            return np.array([Decimal(real), Decimal(imaginary)], dtype=object)

        def multiply(first, second):
            return np.array(
                [
                    first[0] * second[0] - first[1] * second[1],
                    first[0] * second[1] + first[1] * second[0],
                ],
                dtype=object,
            )

        def divide(first, second):
            squared_norm = second[0]**2 + second[1]**2
            conjugate = np.array([second[0], -second[1]], dtype=object)
            return multiply(first, conjugate) / squared_norm

        def evaluate(coefficients, point):
            value = coefficients[-1].copy()
            for coefficient in reversed(coefficients[:-1]):
                value = multiply(value, point) + coefficient
            return value

        try:
            scaled_parameter = number(parameter.real, parameter.imag)
            series = [number(1), multiply(number(0, 1), scaled_parameter) / 2]
            squared_parameter = (
                multiply(scaled_parameter, scaled_parameter) / 4
            )
            for degree in range(2, 2 * order + 1):
                factor = Decimal(degree - 1) * (
                    Decimal(degree) - Decimal("1.5")
                )
                series.append(
                    -(
                        factor * series[-1]
                        + multiply(squared_parameter, series[-2])
                    ) / Decimal(degree * (degree - 1))
                )

            coefficients = np.array(
                [
                    [series[order + row - column] for column in range(order)]
                    for row in range(order)
                ],
                dtype=object,
            )
            real = coefficients[:, :, 0]
            imaginary = coefficients[:, :, 1]
            system = np.block([[real, -imaginary], [imaginary, real]])
            right_hand_side = np.array(
                [-value[0] for value in series[order + 1:]]
                + [-value[1] for value in series[order + 1:]],
                dtype=object,
            )
            augmented = np.column_stack((system, right_hand_side))
            size = 2 * order
            for column in range(size):
                pivot = max(
                    range(column, size),
                    key=lambda row: abs(augmented[row, column]),
                )
                if augmented[pivot, column] == 0:
                    raise ValueError("The Pade matching system is singular.")
                augmented[[column, pivot]] = augmented[[pivot, column]]
                augmented[column, column:] /= augmented[column, column]
                for row in range(column + 1, size):
                    augmented[row, column:] -= (
                        augmented[row, column] * augmented[column, column:]
                    )

            solution = np.empty(size, dtype=object)
            for row in reversed(range(size)):
                solution[row] = augmented[row, -1] - sum(
                    augmented[row, row + 1:size] * solution[row + 1:]
                )
            denominator = np.vstack(
                (
                    number(1),
                    np.column_stack((solution[:order], solution[order:])),
                )
            )
            numerator = np.array(
                [
                    sum(
                        (
                            multiply(
                                denominator[index], series[degree - index]
                            )
                            for index in range(degree + 1)
                        ),
                        number(),
                    )
                    for degree in range(order + 1)
                ],
                dtype=object,
            )
            derivative = np.array(
                [
                    degree * denominator[degree]
                    for degree in range(1, order + 1)
                ],
                dtype=object,
            )
            guesses = np.roots(
                np.array([complex(*value) for value in denominator[::-1]])
            )
            if guesses.size != order or not np.all(np.isfinite(guesses)):
                raise ValueError("The Pade poles cannot be represented.")

            tolerance = Decimal(10)**(20 - context.prec)
            roots = []
            for guess in guesses:
                root = number(float(guess.real), float(guess.imag))
                for _iteration in range(150):
                    correction = divide(
                        evaluate(denominator, root), evaluate(derivative, root)
                    )
                    root -= correction
                    if max(abs(correction)) <= (
                        tolerance * max(1, max(abs(root)))
                    ):
                        break
                else:
                    raise ValueError("A Pade pole could not be refined.")
                if (
                    max(abs(root)) == 0
                    or max(abs(evaluate(derivative, root))) == 0
                ):
                    raise ValueError("Simple finite Pade poles are required.")
                if any(
                    max(abs(root - previous))
                    <= tolerance.sqrt() * max(1, max(abs(root)))
                    for previous in roots
                ):
                    raise ValueError(
                        "Distinct Pade poles could not be resolved."
                    )
                roots.append(root)

            decimal_poles = [-divide(number(1), root) for root in roots]
            decimal_residues = [divide(numerator[-1], denominator[-1])] + [
                multiply(
                    pole,
                    divide(
                        evaluate(numerator, root), evaluate(derivative, root)
                    ),
                )
                for pole, root in zip(decimal_poles, roots)
            ]
            poles = np.array([complex(*value) for value in decimal_poles])
            residues = np.array(
                [complex(*value) for value in decimal_residues]
            )
        except (
            ArithmeticError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            raise ValueError(
                "The Pade partial-fraction representation could not "
                "be computed."
            ) from error

    if not np.all(np.isfinite(poles)) or not np.all(np.isfinite(residues)):
        raise ValueError(
            "The Pade coefficients must be finite complex arrays."
        )
    return poles, residues

import numpy as np
from scipy.sparse import coo_matrix


def build_modulation_matrix(ny, mx, B, C):
    """Build the Fourier matrix for the squared refractive index.

    Periodic shifts contribute at 0, +/-mx, and +/-2*mx. Converting
    COO to CSC sums entries when shifts coincide, including at zero.
    """
    if not isinstance(ny, (int, np.integer)) or ny < 1:
        raise ValueError("ny must be a positive integer.")
    if not isinstance(mx, (int, np.integer)):
        raise ValueError("mx must be an integer.")

    mx = int(mx)
    rows = []
    cols = []
    values = []

    for row_index in range(ny):
        rows.append(row_index)
        cols.append(row_index)
        values.append(C**2 + B**2 / 2.0)

        rows.extend([row_index, row_index])
        cols.extend([(row_index + mx) % ny, (row_index - mx) % ny])
        values.extend([B * C, B * C])

        rows.extend([row_index, row_index])
        cols.extend([
            (row_index + 2 * mx) % ny,
            (row_index - 2 * mx) % ny,
        ])
        values.extend([B**2 / 4.0, B**2 / 4.0])

    return coo_matrix(
        (values, (rows, cols)), shape=(ny, ny), dtype=complex
    ).tocsc()

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu


def wjsolver(kx, ky, mx, B, C, bj, k0, U, lu_factors=None):
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
        modulation_matrix = build_modulation_matrix(
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

import numpy as np


def apply_pade(kx, ky, mx, B, C, k0, U, b, d, lu_cache=None):
    """Apply R(X)U with X=M_tilde-I-D and matched pole/residue pairs.

    R(X)=d[0]*I+sum_j d[j+1]*(I+b[j]*X)^(-1).
    Here M_tilde is the positive index-squared modulation matrix and
    D=diag((kx**2+ky[column]**2)/k0**2) for each ky column.
    Cache entries belong to the same ordered b, grid, and material values;
    callers must pass None if any of those values change.
    """
    spectrum = np.asarray(U, dtype=complex)
    poles = np.asarray(b, dtype=complex)
    residues = np.asarray(d, dtype=complex)

    if not np.isfinite(k0) or k0 <= 0:
        raise ValueError("k0 must be finite and positive.")
    if spectrum.ndim != 2 or spectrum.shape[0] != spectrum.shape[1]:
        raise ValueError("U must be a square 2D array.")

    size = spectrum.shape[0]
    if np.asarray(kx).shape != (size,) or np.asarray(ky).shape != (size,):
        raise ValueError("kx and ky must match the field dimensions.")
    if poles.ndim != 1 or residues.shape != (poles.size + 1,):
        raise ValueError("b and d must have shapes (P,) and (P+1,).")
    if not all(
        np.all(np.isfinite(values))
        for values in (spectrum, poles, residues)
    ):
        raise ValueError("The field and coefficients must be finite.")

    if lu_cache is None:
        lu_cache = [None] * poles.size
    if len(lu_cache) != poles.size:
        raise ValueError("lu_cache must contain one entry per pole.")

    updated = residues[0] * spectrum
    for index, pole in enumerate(poles):
        auxiliary, lu_cache[index] = wjsolver(
            kx, ky, mx, B, C, pole, k0, spectrum, lu_cache[index]
        )
        updated += residues[index + 1] * auxiliary

    return updated, lu_cache

import numpy as np


def radial_angular_power(U, kx, ky, k0, nbins=None):
    """Return radial centers, angular density, and polar angles.

    Uniform radial bins cover [0, min(max(abs(kx)), max(abs(ky)), k0)].
    Bins are left-closed and right-open except for the closed final bin.
    Filtering is applied to individual Fourier samples before accumulation.
    Density is peak-normalized radial PSD divided by angular-bin width.
    """
    spectrum = np.asarray(U, dtype=complex)
    first_axis = np.asarray(kx, dtype=float)
    second_axis = np.asarray(ky, dtype=float)

    if first_axis.ndim != 1 or second_axis.ndim != 1:
        raise ValueError("kx and ky must be one-dimensional.")
    if spectrum.shape != (first_axis.size, second_axis.size):
        raise ValueError("U shape must be (len(kx), len(ky)).")
    if first_axis.size < 2 or second_axis.size < 2:
        raise ValueError("kx and ky must each contain at least two values.")
    if not np.isfinite(k0) or k0 <= 0:
        raise ValueError("k0 must be finite and positive.")
    if not all(
        np.all(np.isfinite(values))
        for values in (spectrum, first_axis, second_axis)
    ):
        raise ValueError("The field and wavenumbers must be finite.")

    if nbins is None:
        nbins = min(first_axis.size, second_axis.size)
    if (
        isinstance(nbins, (bool, np.bool_))
        or not isinstance(nbins, (int, np.integer))
        or nbins < 1
    ):
        raise ValueError("nbins must be a positive integer.")

    first_differences = np.diff(first_axis)
    second_differences = np.diff(second_axis)
    for differences in (first_differences, second_differences):
        if differences[0] <= 0 or not np.allclose(
            differences, differences[0], rtol=1e-12, atol=0
        ):
            raise ValueError(
                "Wavenumbers must be increasing and uniformly spaced."
            )

    radial_limit = min(
        np.max(np.abs(first_axis)), np.max(np.abs(second_axis)), k0
    )
    if radial_limit <= 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    edges = np.linspace(0.0, radial_limit, int(nbins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radii = np.hypot(first_axis[:, None], second_axis[None, :])
    retained = (radii <= radial_limit) & (radii <= k0)
    spectral_scale = np.max(np.abs(spectrum[retained]), initial=0.0)

    if spectral_scale > 0:
        weights = np.abs(spectrum[retained] / spectral_scale) ** 2
    else:
        weights = np.zeros(np.count_nonzero(retained), dtype=float)

    annular_power, _ = np.histogram(
        radii[retained], bins=edges, weights=weights
    )
    peak = np.max(annular_power, initial=0.0)
    normalized_radial_psd = (
        annular_power / peak if peak > 0 else np.zeros(int(nbins))
    )
    angle_edges = np.arcsin(np.clip(edges / k0, 0.0, 1.0))
    angles = np.arcsin(np.clip(centers / k0, 0.0, 1.0))
    density = normalized_radial_psd / np.diff(angle_edges)
    return centers, density, angles

import numpy as np


def initial_fourier_transform(U0, dx, dy):
    """Return the centered Fourier field and angular-wavenumber axes.

    The input origin is at (size//2, size//2). The forward transform
    is unnormalized, with dx and dy assigned to axes 0 and 1.
    """
    field = np.asarray(U0, dtype=complex)

    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("U0 must be a square 2D array.")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive.")

    size = field.shape[0]
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(size, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(size, d=dy))
    return spectrum, kx, ky

import numpy as np


def propagate_and_angular_power(
    U0, P, k0, h, zlength, dx, dy, mx, B, C, nbins=None, plot=False
):
    """Propagate to zlength, including a remainder with fresh factors.

    Return (U, normalized_power, kx, ky, k_perp, angular_density, Pangle).
    U is the centered unnormalized Fourier envelope, and normalized_power
    is its Cartesian power divided by its peak. The last three outputs
    follow radial_angular_power. A zero distance returns the initial FFT.
    """
    initial = np.asarray(U0, dtype=complex)
    if (
        initial.ndim != 2
        or initial.shape[0] != initial.shape[1]
        or initial.shape[0] < 2
    ):
        raise ValueError(
            "U0 must be a square 2D array with at least two samples per axis."
        )
    if not np.all(np.isfinite(initial)):
        raise ValueError("U0 must be finite.")
    if (
        isinstance(P, (bool, np.bool_))
        or not isinstance(P, (int, np.integer))
        or P < 1
    ):
        raise ValueError("P must be a positive integer.")
    if not all(np.isfinite(value) and value > 0 for value in (k0, h, dx, dy)):
        raise ValueError("k0, h, dx, and dy must be finite and positive.")
    if not np.isfinite(zlength) or zlength < 0:
        raise ValueError("zlength must be finite and nonnegative.")

    spectrum, first_axis, second_axis = initial_fourier_transform(
        initial, dx, dy
    )
    step_count = int(np.floor(zlength / h))
    remainder = float(zlength - step_count * h)
    cache = None

    if step_count:
        poles, residues = PadeCoeff(P, k0 * h)
        for _ in range(step_count):
            spectrum, cache = apply_pade(
                first_axis, second_axis, mx, B, C, k0, spectrum,
                poles, residues, cache
            )

    if remainder > 0:
        poles, residues = PadeCoeff(P, k0 * remainder)
        spectrum, _ = apply_pade(
            first_axis, second_axis, mx, B, C, k0, spectrum,
            poles, residues, None
        )

    if not np.all(np.isfinite(spectrum)):
        raise ValueError("The propagated field is not finite.")
    peak_amplitude = np.max(np.abs(spectrum), initial=0.0)
    normalized_power = (
        np.abs(spectrum / peak_amplitude) ** 2
        if peak_amplitude > 0
        else np.zeros(spectrum.shape, dtype=float)
    )
    centers, density, angles = radial_angular_power(
        spectrum, first_axis, second_axis, k0, nbins
    )

    if plot:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(1, 2)
        axes[0].pcolormesh(
            first_axis, second_axis, normalized_power.T, shading="auto"
        )
        axes[0].set(xlabel="kx (rad/m)", ylabel="ky (rad/m)")
        axes[1].plot(angles, density)
        axes[1].set(
            xlabel="Polar angle (rad)", ylabel="Angular density (rad^-1)"
        )
        plt.tight_layout()
        plt.show()

    return (
        spectrum, normalized_power, first_axis, second_axis,
        centers, density, angles
    )

import numpy as np


def wide_angle_power_fraction(
    U0, P, k0, h, zlength, dx, dy, mx, B, C,
    nbins=None, theta_min=np.pi / 6
):
    """Return the power fraction selected by polar bin-center angle.

    Uniform radial centers determine the angular-bin widths. Multiplying
    density by these widths recovers common-scale annular powers.
    Select bins whose center angle is at least theta_min. The cutoff
    must be a finite real scalar in [0, pi/2]. Zero retained power gives 0.
    """
    raw_angle = np.asarray(theta_min)
    if raw_angle.ndim != 0 or raw_angle.dtype.kind not in "iuf":
        raise ValueError("theta_min must be a finite real scalar.")

    minimum_angle = float(raw_angle)
    if not np.isfinite(minimum_angle) or not 0 <= minimum_angle <= np.pi / 2:
        raise ValueError("theta_min must lie in [0, pi/2].")

    outputs = propagate_and_angular_power(
        U0, P, k0, h, zlength, dx, dy, mx, B, C, nbins=nbins, plot=False
    )
    centers, density, angles = outputs[4:]
    if len(centers) == 0:
        return 0.0

    radial_edges = 2 * centers[0] * np.arange(len(centers) + 1, dtype=float)
    angular_widths = np.diff(
        np.arcsin(np.clip(radial_edges / k0, 0.0, 1.0))
    )
    weights = density * angular_widths
    total = np.sum(weights)
    if total == 0:
        return 0.0

    return float(np.sum(weights[angles >= minimum_angle]) / total)
SCICODE_GOLD_EOF
