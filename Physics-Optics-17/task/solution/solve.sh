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


def _jet(value, derivative):
    return np.concatenate(
        (
            np.asarray(value, complex)[..., None],
            np.asarray(derivative, complex),
        ),
        axis=-1,
    )


def _mul(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] * b[..., 0]
    out[..., 1:] = a[..., :1] * b[..., 1:] + a[..., 1:] * b[..., :1]
    return out


def _div(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] / b[..., 0]
    out[..., 1:] = (a[..., 1:] - out[..., :1] * b[..., 1:]) / b[..., :1]
    return out


def _exp(a):
    out = np.empty_like(a)
    out[..., 0] = np.exp(a[..., 0])
    out[..., 1:] = out[..., :1] * a[..., 1:]
    return out


def _expm1(a):
    out = _exp(a)
    out[..., 0] = np.expm1(a[..., 0])
    return out


def logarithmic_wave_jets(
    z: "np.ndarray",
    dz: "np.ndarray",
    nmax: int,
    depth: int,
) -> "np.ndarray":
    if (
        not isinstance(nmax, (int, np.integer))
        or not isinstance(depth, (int, np.integer))
        or not 1 <= nmax <= 16
        or not max(nmax + 1, 40) <= depth <= 100
        or np.any(np.asarray(z) == 0)
    ):
        raise ValueError("invalid argument or recurrence order")
    z = _jet(z, dz)
    one = _jet(np.ones_like(z[..., 0]), np.zeros_like(dz))
    regular = np.zeros((depth + 1,) + z.shape, dtype=complex)
    for order in range(depth, 0, -1):
        ratio = _div(order * one, z)
        regular[order - 1] = ratio - _div(one, regular[order] + ratio)
    out = np.empty((nmax + 1, 2) + z.shape, dtype=complex)
    out[:, 0] = regular[: nmax + 1]
    out[0, 1] = 1j * one
    product = -0.5 * _expm1(2j * z)
    for order in range(1, nmax + 1):
        ratio = _div(order * one, z)
        product = _mul(
            product, _mul(ratio - out[order - 1, 0], ratio - out[order - 1, 1])
        )
        out[order, 1] = out[order, 0] + _div(1j * one, product)
    return out

import numpy as np


def _jet(value, derivative):
    return np.concatenate(
        (
            np.asarray(value, complex)[..., None],
            np.asarray(derivative, complex),
        ),
        axis=-1,
    )


def _mul(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] * b[..., 0]
    out[..., 1:] = a[..., :1] * b[..., 1:] + a[..., 1:] * b[..., :1]
    return out


def _div(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] / b[..., 0]
    out[..., 1:] = (a[..., 1:] - out[..., :1] * b[..., 1:]) / b[..., :1]
    return out


def _exp(a):
    out = np.empty_like(a)
    out[..., 0] = np.exp(a[..., 0])
    out[..., 1:] = out[..., :1] * a[..., 1:]
    return out


def _expm1(a):
    out = _exp(a)
    out[..., 0] = np.expm1(a[..., 0])
    return out


def shell_ratio_jets(
    z1: "np.ndarray",
    dz1: "np.ndarray",
    z2: "np.ndarray",
    dz2: "np.ndarray",
    inner: "np.ndarray",
    outer: "np.ndarray",
) -> "np.ndarray":
    a, b = _jet(z1, dz1), _jet(z2, dz2)
    one = _jet(np.ones_like(z1), np.zeros_like(dz1))
    result = np.empty(inner[:, 0].shape, dtype=complex)
    result[0] = _mul(_exp(2j * (b - a)), _div(_expm1(2j * a), _expm1(2j * b)))
    radii_ratio = _div(a, b)
    for order in range(1, len(result)):
        num = _mul(
            _mul(b, outer[order, 0]) + order * one,
            order * one - _mul(b, outer[order - 1, 1]),
        )
        den = _mul(
            _mul(a, inner[order, 0]) + order * one,
            order * one - _mul(a, inner[order - 1, 1]),
        )
        result[order] = _mul(
            result[order - 1],
            _mul(_mul(radii_ratio, radii_ratio), _div(num, den)),
        )
    return result

import numpy as np


def _jet(value, derivative):
    return np.concatenate(
        (
            np.asarray(value, complex)[..., None],
            np.asarray(derivative, complex),
        ),
        axis=-1,
    )


def _mul(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] * b[..., 0]
    out[..., 1:] = a[..., :1] * b[..., 1:] + a[..., 1:] * b[..., :1]
    return out


def _div(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] / b[..., 0]
    out[..., 1:] = (a[..., 1:] - out[..., :1] * b[..., 1:]) / b[..., :1]
    return out


def composite_impedance_jets(
    previous: "np.ndarray",
    m1: "np.ndarray",
    dm1: "np.ndarray",
    m2: "np.ndarray",
    dm2: "np.ndarray",
    inner: "np.ndarray",
    outer: "np.ndarray",
    ratio: "np.ndarray",
) -> "np.ndarray":
    a, b = _jet(m1, dm1), _jet(m2, dm2)
    result = np.empty_like(previous)
    for pol in range(2):
        left, right = (b, a) if pol == 0 else (a, b)
        g1 = _mul(left, previous[:, pol]) - _mul(right, inner[:, 0])
        g2 = _mul(left, previous[:, pol]) - _mul(right, inner[:, 1])
        qg1 = _mul(ratio, g1)
        result[:, pol] = _div(
            _mul(g2, outer[:, 0]) - _mul(qg1, outer[:, 1]),
            g2 - qg1,
        )
    return result

import numpy as np


def _jet(value, derivative):
    return np.concatenate(
        (
            np.asarray(value, complex)[..., None],
            np.asarray(derivative, complex),
        ),
        axis=-1,
    )


def _mul(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] * b[..., 0]
    out[..., 1:] = a[..., :1] * b[..., 1:] + a[..., 1:] * b[..., :1]
    return out


def _div(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] / b[..., 0]
    out[..., 1:] = (a[..., 1:] - out[..., :1] * b[..., 1:]) / b[..., :1]
    return out


def _exp(a):
    out = np.empty_like(a)
    out[..., 0] = np.exp(a[..., 0])
    out[..., 1:] = out[..., :1] * a[..., 1:]
    return out


def host_radial_jets(
    x: "np.ndarray",
    dx: "np.ndarray",
    logarithms: "np.ndarray",
) -> "np.ndarray":
    z = _jet(x, dx)
    one = _jet(np.ones_like(x), np.zeros_like(dx))
    result = np.empty_like(logarithms)
    result[0, 0] = _jet(np.sin(x), np.cos(x)[..., None] * np.asarray(dx))
    result[0, 1] = -1j * _exp(1j * z)
    for order in range(1, len(result)):
        n_over_z = _div(order * one, z)
        for kind in range(2):
            result[order, kind] = _mul(
                result[order - 1, kind],
                n_over_z - logarithms[order - 1, kind],
            )
    return result

import numpy as np


def _jet(value, derivative):
    return np.concatenate(
        (
            np.asarray(value, complex)[..., None],
            np.asarray(derivative, complex),
        ),
        axis=-1,
    )


def _mul(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] * b[..., 0]
    out[..., 1:] = a[..., :1] * b[..., 1:] + a[..., 1:] * b[..., :1]
    return out


def _div(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] / b[..., 0]
    out[..., 1:] = (a[..., 1:] - out[..., :1] * b[..., 1:]) / b[..., :1]
    return out


def layered_coefficient_jets(
    x: "np.ndarray",
    dx: "np.ndarray",
    m: "np.ndarray",
    dm: "np.ndarray",
    composite: "np.ndarray",
    radial: "np.ndarray",
) -> "np.ndarray":
    z, material = _jet(x, dx), _jet(m, dm)
    one = _jet(np.ones_like(x), np.zeros_like(dx))
    result = np.empty((len(composite) - 1, 2) + z.shape, dtype=complex)
    for order in range(1, len(composite)):
        n_over_z = _div(order * one, z)
        for pol in range(2):
            h = (
                _div(composite[order, pol], material)
                if pol == 0
                else _mul(composite[order, pol], material)
            )
            prefactor = h + n_over_z
            num = _mul(prefactor, radial[order, 0]) - radial[order - 1, 0]
            den = _mul(prefactor, radial[order, 1]) - radial[order - 1, 1]
            result[order - 1, pol] = _div(num, den)
    return result

import numpy as np


def _jet(value, derivative):
    return np.concatenate(
        (
            np.asarray(value, complex)[..., None],
            np.asarray(derivative, complex),
        ),
        axis=-1,
    )


def _mul(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] * b[..., 0]
    out[..., 1:] = a[..., :1] * b[..., 1:] + a[..., 1:] * b[..., :1]
    return out


def _div(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] / b[..., 0]
    out[..., 1:] = (a[..., 1:] - out[..., :1] * b[..., 1:]) / b[..., :1]
    return out


def mie_efficiency_jets(
    x: "np.ndarray",
    dx: "np.ndarray",
    coeff: "np.ndarray",
) -> "np.ndarray":
    weight = 2 * np.arange(1, len(coeff) + 1) + 1
    extinction = np.sum(
        weight[:, None, None] * np.real(coeff.sum(axis=1)), axis=0
    )
    squares = np.real(_mul(coeff, np.conj(coeff)))
    scattering = np.sum(weight[:, None, None] * squares.sum(axis=1), axis=0)
    data = np.stack((extinction, scattering, extinction - scattering))
    size = _jet(x, dx)
    factor = _div(
        _jet(2 * np.ones_like(x), np.zeros_like(dx)), _mul(size, size)
    )
    return np.real(_mul(data, factor))

import numpy as np


def _material_design(u):
    lower = np.array([35, 12, 15, 2.8, 1.35, 2.1, 0, 0.015, 0.005])
    upper = np.array([85, 48, 65, 4.2, 2.15, 3.3, 0.07, 0.16, 0.08])
    s = 1 / (1 + np.exp(-u))
    params = lower + (upper - lower) * s
    jac = np.diag((upper - lower) * s * (1 - s))
    radii = np.cumsum(params[:3])
    dr = np.cumsum(jac[:3], axis=0)
    index = params[3:6] + 1j * params[6:9]
    dm = jac[3:6] + 1j * jac[6:9]
    return params, radii, index, dr, dm


def spectral_loss_gradient(
    u: "np.ndarray",
) -> "np.ndarray":
    u = np.asarray(u, dtype=float)
    if u.shape != (9,) or not np.all(np.isfinite(u)) or np.any(abs(u) > 6):
        raise ValueError("u must have nine finite entries in [-6, 6]")
    nmax, depth = 12, 80
    _, radii, index, dr, dm = _material_design(np.asarray(u))
    wavelengths = 430 + 17 * np.arange(25)
    k = 2 * np.pi / wavelengths
    x, dx = radii[:, None] * k, dr[:, None, :] * k[None, :, None]
    ms = index[:, None] * np.ones((3, len(k)))
    dms = dm[:, None, :] * np.ones((3, len(k), 9))
    z = ms[0] * x[0]
    dz = dms[0] * x[0, :, None] + ms[0, :, None] * dx[0]
    core = logarithmic_wave_jets(z, dz, nmax, depth)
    composite = np.stack((core[:, 0], core[:, 0]), axis=1)
    for layer in range(1, 3):
        z1 = ms[layer] * x[layer - 1]
        z2 = ms[layer] * x[layer]
        dz1 = (
            dms[layer] * x[layer - 1, :, None]
            + ms[layer, :, None] * dx[layer - 1]
        )
        dz2 = dms[layer] * x[layer, :, None] + ms[layer, :, None] * dx[layer]
        inner = logarithmic_wave_jets(z1, dz1, nmax, depth)
        outer = logarithmic_wave_jets(z2, dz2, nmax, depth)
        q = shell_ratio_jets(z1, dz1, z2, dz2, inner, outer)
        composite = composite_impedance_jets(
            composite,
            ms[layer - 1],
            dms[layer - 1],
            ms[layer],
            dms[layer],
            inner,
            outer,
            q,
        )
    host = logarithmic_wave_jets(x[-1], dx[-1], nmax, depth)
    radial = host_radial_jets(x[-1], dx[-1], host)
    coeff = layered_coefficient_jets(
        x[-1], dx[-1], ms[-1], dms[-1], composite, radial
    )
    spectra = mie_efficiency_jets(x[-1], dx[-1], coeff).transpose(
        1, 0, 2
    )
    target = 0.15 + 3 * np.exp(-0.5 * ((wavelengths - 620) / 70) ** 2)
    residual = spectra[:, 1, 0] - target
    loss = np.mean(residual**2 + 0.25 * spectra[:, 2, 0] ** 2)
    grad = np.mean(
        2 * residual[:, None] * spectra[:, 1, 1:]
        + 0.5 * spectra[:, 2, :1] * spectra[:, 2, 1:],
        axis=0,
    )
    return np.r_[loss, grad]

import numpy as np


def adam_design_step(
    state: "np.ndarray",
    grad: "np.ndarray",
    step: int,
    learning_rate: float,
) -> "np.ndarray":
    if (
        not isinstance(step, (int, np.integer))
        or not 1 <= step <= 60
        or not 0 < learning_rate <= 0.06
    ):
        raise ValueError("invalid Adam iteration or learning rate")
    u, first, second = np.asarray(state)
    first = 0.9 * first + 0.1 * grad
    second = 0.999 * second + 0.001 * grad**2
    correction = (first / (1 - 0.9**step)) / (
        np.sqrt(second / (1 - 0.999**step)) + 1e-8
    )
    return np.array([u - learning_rate * correction, first, second])

import numpy as np


def _material_design(u):
    lower = np.array([35, 12, 15, 2.8, 1.35, 2.1, 0, 0.015, 0.005])
    upper = np.array([85, 48, 65, 4.2, 2.15, 3.3, 0.07, 0.16, 0.08])
    s = 1 / (1 + np.exp(-u))
    params = lower + (upper - lower) * s
    jac = np.diag((upper - lower) * s * (1 - s))
    radii = np.cumsum(params[:3])
    dr = np.cumsum(jac[:3], axis=0)
    index = params[3:6] + 1j * params[6:9]
    dm = jac[3:6] + 1j * jac[6:9]
    return params, radii, index, dr, dm


def run_layered_design(
    steps: int,
    learning_rate: float,
    initial: "np.ndarray",
) -> "np.ndarray":
    if (
        not isinstance(steps, (int, np.integer))
        or not 0 <= steps <= 60
        or not 0 < learning_rate <= 0.06
    ):
        raise ValueError("invalid update count or learning rate")
    initial = np.asarray(initial, dtype=float)
    if (
        initial.shape != (9,)
        or not np.all(np.isfinite(initial))
        or np.any(abs(initial) > 1)
    ):
        raise ValueError("initial must have nine finite entries in [-1, 1]")
    state = np.array([initial, np.zeros(9), np.zeros(9)])
    initial_loss = spectral_loss_gradient(state[0])[0]
    for step in range(1, steps + 1):
        data = spectral_loss_gradient(state[0])
        state = adam_design_step(state, data[1:], step, learning_rate)
    final = spectral_loss_gradient(state[0])
    params = _material_design(state[0])[0]
    return np.r_[
        np.log10(initial_loss / final[0]),
        initial_loss,
        final[0],
        np.linalg.norm(final[1:]),
        params,
        state[0],
    ]
SCICODE_GOLD_EOF
