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


def _validate_cell(conductivities, capacities, thicknesses, laplace_s):
    arrays = []
    for name, value in (("conductivities", conductivities), ("capacities", capacities), ("thicknesses", thicknesses)):
        try:
            raw = np.asarray(value)
            if np.iscomplexobj(raw) or raw.dtype == object:
                raise TypeError
            a = raw.astype(float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an array of real numbers") from None
        if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)) or np.any(a <= 0.0):
            raise ValueError(f"{name} must be a non-empty one-dimensional array of finite values above zero")
        arrays.append(a)
    if not (arrays[0].size == arrays[1].size == arrays[2].size):
        raise ValueError("the layer arrays must have equal length")
    try:
        s = complex(laplace_s)
    except (TypeError, ValueError):
        raise ValueError("laplace_s must be a number") from None
    if not np.isfinite(s) or s == 0 or s.real < 0.0:
        raise ValueError("laplace_s must be finite, nonzero and have a non-negative real part")
    return arrays[0], arrays[1], arrays[2], s


def _slice_cell(kappa, cap, h, start):
    """Layers of the period (start, start + l) of the laminate, the layer containing start split in two."""
    period = float(np.sum(h))
    edges = np.concatenate(([0.0], np.cumsum(h)))
    y = float(start) % period
    j = int(np.searchsorted(edges, y, side="right")) - 1
    j = min(max(j, 0), h.size - 1)
    if abs(y - edges[j]) <= 1e-15 * period or abs(y - edges[j + 1]) <= 1e-15 * period:
        j = (j + 1) % h.size if abs(y - edges[j + 1]) <= 1e-15 * period else j
        order = [(j + i) % h.size for i in range(h.size)]
        return kappa[order], cap[order], h[order]
    lead, trail = edges[j + 1] - y, y - edges[j]
    order = [(j + i) % h.size for i in range(1, h.size)]
    return (np.concatenate(([kappa[j]], kappa[order], [kappa[j]])),
            np.concatenate(([cap[j]], cap[order], [cap[j]])),
            np.concatenate(([lead], h[order], [trail])))


def cell_transfer_matrix(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    start: float,
) -> dict:
    """Reference implementation."""
    kappa, cap, h, s = _validate_cell(conductivities, capacities, thicknesses, laplace_s)
    try:
        y = float(start)
    except (TypeError, ValueError):
        raise ValueError("start must be a real number") from None
    if not np.isfinite(y):
        raise ValueError("start must be finite")
    kappa, cap, h = _slice_cell(kappa, cap, h, y)
    T = np.eye(2, dtype=complex)
    for a, c, d in zip(kappa, cap, h):
        k = np.sqrt(c * s / a)
        inv_impedance = a * k
        layer = np.array([[np.cosh(k * d), inv_impedance * np.sinh(k * d)],
                          [np.sinh(k * d) / inv_impedance, np.cosh(k * d)]])
        T = layer @ T
    return {"matrix": T, "determinant": complex(np.linalg.det(T)), "trace": complex(T[0, 0] + T[1, 1])}

import numpy as np


def bloch_trace_retrieval(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    start: float,
) -> dict:
    """Reference implementation."""
    out = cell_transfer_matrix(conductivities, capacities, thicknesses, laplace_s, start)  # noqa: F821
    T = out["matrix"]
    s = complex(laplace_s)
    period = float(np.sum(np.asarray(thicknesses, dtype=float)))
    half_trace = 0.5 * out["trace"]
    if abs(T[1, 0]) < 1e-300 or abs(half_trace * half_trace - 1.0) < 1e-14:
        raise ValueError("the retrieval is undefined for this period transfer matrix")
    phase = complex(np.arccosh(half_trace))
    if phase.real < 0.0:
        phase = -phase
    phase = complex(phase.real, (phase.imag + np.pi) % (2.0 * np.pi) - np.pi)
    if phase.imag == -np.pi:
        phase = complex(phase.real, np.pi)
    S = np.sinh(phase)
    chi = (T[0, 0] - T[1, 1]) / (2.0 * T[1, 0])
    kappa = S * period / (T[1, 0] * phase)
    capacity = S * phase / (s * T[1, 0] * period)
    return {"bloch_wavenumber": complex(phase / period), "kappa": complex(kappa), "capacity": complex(capacity),
            "chi": complex(chi), "xi": complex(chi / s)}

import numpy as np


def _validate_cell(conductivities, capacities, thicknesses, laplace_s):
    arrays = []
    for name, value in (("conductivities", conductivities), ("capacities", capacities), ("thicknesses", thicknesses)):
        try:
            raw = np.asarray(value)
            if np.iscomplexobj(raw) or raw.dtype == object:
                raise TypeError
            a = raw.astype(float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an array of real numbers") from None
        if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)) or np.any(a <= 0.0):
            raise ValueError(f"{name} must be a non-empty one-dimensional array of finite values above zero")
        arrays.append(a)
    if not (arrays[0].size == arrays[1].size == arrays[2].size):
        raise ValueError("the layer arrays must have equal length")
    try:
        s = complex(laplace_s)
    except (TypeError, ValueError):
        raise ValueError("laplace_s must be a number") from None
    if not np.isfinite(s) or s == 0 or s.real < 0.0:
        raise ValueError("laplace_s must be finite, nonzero and have a non-negative real part")
    return arrays[0], arrays[1], arrays[2], s


def _decay_integral(z, d):
    """Integral of exp(-z u) over 0 < u < d, stable for small z d."""
    if abs(z * d) < 1e-8:
        return d * (1.0 - 0.5 * z * d)
    return -np.expm1(-z * d) / z


def forced_bloch_cell_response(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    wavenumber: complex,
    heat_source: complex,
    residual_gradient: complex,
    residual_temperature: complex,
) -> dict:
    """Reference implementation."""
    kappa, cap, h, s = _validate_cell(conductivities, capacities, thicknesses, laplace_s)
    try:
        K, r0, z0, f0 = (complex(v) for v in (wavenumber, heat_source, residual_gradient, residual_temperature))
    except (TypeError, ValueError):
        raise ValueError("the wavenumber and the source amplitudes must be numbers") from None
    if not all(np.isfinite(v) for v in (K, r0, z0, f0)):
        raise ValueError("the wavenumber and the source amplitudes must be finite")
    n = h.size
    edges = np.concatenate(([0.0], np.cumsum(h)))
    period = edges[-1]
    k = np.sqrt(cap * s / kappa)
    denominator = kappa * K * K - s * cap
    if np.any(np.abs(denominator) <= 1e-12 * np.maximum(np.abs(kappa * K * K), np.abs(s * cap))):
        raise ValueError("the wavenumber makes a particular solution resonant in a layer")
    A = (kappa * K * z0 - s * cap * f0 - r0) / denominator

    def _state(j, x):
        # temperature and -q rows for the two homogeneous amplitudes, and the particular part, at x in layer j
        grow = np.exp(k[j] * (x - edges[j + 1]))
        decay = np.exp(-k[j] * (x - edges[j]))
        wave = np.exp(K * x)
        return (np.array([grow, decay]), kappa[j] * k[j] * np.array([grow, -decay]),
                A[j] * wave, kappa[j] * (K * A[j] - z0) * wave)

    M = np.zeros((2 * n, 2 * n), dtype=complex)
    rhs = np.zeros(2 * n, dtype=complex)
    row = 0
    for j in range(n - 1):
        tl, ql, tpl, qpl = _state(j, edges[j + 1])
        tr, qr, tpr, qpr = _state(j + 1, edges[j + 1])
        M[row, 2 * j:2 * j + 2], M[row, 2 * j + 2:2 * j + 4], rhs[row] = tl, -tr, tpr - tpl
        M[row + 1, 2 * j:2 * j + 2], M[row + 1, 2 * j + 2:2 * j + 4], rhs[row + 1] = ql, -qr, qpr - qpl
        row += 2
    bloch = np.exp(K * period)
    te, qe, tpe, qpe = _state(n - 1, period)
    t0, q0, tp0, qp0 = _state(0, 0.0)
    M[row, 2 * n - 2:] += te
    M[row, 0:2] -= bloch * t0
    rhs[row] = bloch * tp0 - tpe
    M[row + 1, 2 * n - 2:] += qe
    M[row + 1, 0:2] -= bloch * q0
    rhs[row + 1] = bloch * qp0 - qpe
    try:
        amplitudes = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        raise ValueError("the forced cell problem is singular: the wavenumber lies on the Bloch dispersion") from None
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("the forced cell problem is singular: the wavenumber lies on the Bloch dispersion")

    theta_mean = flux_mean = entropy_mean = 0.0 + 0.0j
    for j in range(n):
        a, b = amplitudes[2 * j], amplitudes[2 * j + 1]
        i_grow = np.exp(-K * edges[j + 1]) * _decay_integral(k[j] - K, h[j])
        i_decay = np.exp(-K * edges[j]) * _decay_integral(k[j] + K, h[j])
        theta_mean += a * i_grow + b * i_decay + A[j] * h[j]
        flux_mean += kappa[j] * (k[j] * a * i_grow - k[j] * b * i_decay + (K * A[j] - z0) * h[j])
        entropy_mean += cap[j] * (a * i_grow + b * i_decay + (A[j] - f0) * h[j])
    return {"mean_temperature": complex(theta_mean / period), "mean_flux": complex(flux_mean / period),
            "mean_entropy": complex(entropy_mean / period)}

import numpy as np


def source_driven_constitutive_matrix(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    wavenumber: complex,
) -> dict:
    """Reference implementation."""
    try:
        K = complex(wavenumber)
    except (TypeError, ValueError):
        raise ValueError("wavenumber must be a number") from None
    kinematic, kinetic = [], []
    for r0, z0, f0 in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        out = forced_bloch_cell_response(conductivities, capacities, thicknesses,  # noqa: F821
                                                 laplace_s, K, r0, z0, f0)
        theta = out["mean_temperature"]
        kinematic.append([K * theta - z0, theta - f0])
        kinetic.append([out["mean_flux"], out["mean_entropy"]])
    B = np.array(kinematic).T
    H = np.array(kinetic).T
    if np.linalg.cond(B[:, :2]) > 1e13:
        raise ValueError("the mean kinematic vectors of the heat source and the residual gradient are dependent")
    L = H[:, :2] @ np.linalg.inv(B[:, :2])
    consistency = float(np.max(np.abs(L @ B[:, 2] - H[:, 2])) / np.max(np.abs(H)))
    return {"kappa": complex(L[0, 0]), "chi": complex(L[0, 1]), "xi": complex(L[1, 0]),
            "capacity": complex(L[1, 1]), "consistency": consistency}

import numpy as np


def _validate_cell(conductivities, capacities, thicknesses, laplace_s):
    arrays = []
    for name, value in (("conductivities", conductivities), ("capacities", capacities), ("thicknesses", thicknesses)):
        try:
            raw = np.asarray(value)
            if np.iscomplexobj(raw) or raw.dtype == object:
                raise TypeError
            a = raw.astype(float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an array of real numbers") from None
        if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)) or np.any(a <= 0.0):
            raise ValueError(f"{name} must be a non-empty one-dimensional array of finite values above zero")
        arrays.append(a)
    if not (arrays[0].size == arrays[1].size == arrays[2].size):
        raise ValueError("the layer arrays must have equal length")
    try:
        s = complex(laplace_s)
    except (TypeError, ValueError):
        raise ValueError("laplace_s must be a number") from None
    if not np.isfinite(s) or s == 0 or s.real < 0.0:
        raise ValueError("laplace_s must be finite, nonzero and have a non-negative real part")
    return arrays[0], arrays[1], arrays[2], s


def _slice_cell(kappa, cap, h, start):
    """Layers of the period (start, start + l) of the laminate, the layer containing start split in two."""
    period = float(np.sum(h))
    edges = np.concatenate(([0.0], np.cumsum(h)))
    y = float(start) % period
    j = int(np.searchsorted(edges, y, side="right")) - 1
    j = min(max(j, 0), h.size - 1)
    if abs(y - edges[j]) <= 1e-15 * period or abs(y - edges[j + 1]) <= 1e-15 * period:
        j = (j + 1) % h.size if abs(y - edges[j + 1]) <= 1e-15 * period else j
        order = [(j + i) % h.size for i in range(h.size)]
        return kappa[order], cap[order], h[order]
    lead, trail = edges[j + 1] - y, y - edges[j]
    order = [(j + i) % h.size for i in range(1, h.size)]
    return (np.concatenate(([kappa[j]], kappa[order], [kappa[j]])),
            np.concatenate(([cap[j]], cap[order], [cap[j]])),
            np.concatenate(([lead], h[order], [trail])))


def willis_symmetry_certificate(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    wavenumber: complex,
    start: float,
) -> dict:
    """Reference implementation."""
    kappa, cap, h, s = _validate_cell(conductivities, capacities, thicknesses, laplace_s)
    try:
        y, K = float(start), complex(wavenumber)
    except (TypeError, ValueError):
        raise ValueError("start must be a real number and wavenumber a number") from None
    if not np.isfinite(y):
        raise ValueError("start must be finite")
    here = source_driven_constitutive_matrix(kappa, cap, h, s, K)  # noqa: F821
    back = source_driven_constitutive_matrix(kappa, cap, h, s, -K)  # noqa: F821
    mirror = source_driven_constitutive_matrix(kappa[::-1], cap[::-1], h[::-1], s, -K)  # noqa: F821
    cut = source_driven_constitutive_matrix(*_slice_cell(kappa, cap, h, y), s, K)  # noqa: F821
    scale = max(abs(here["kappa"]), abs(here["chi"]), abs(s * here["xi"]), abs(s * here["capacity"]))
    adjoint = max(abs(here["chi"] - s * back["xi"]), abs(here["kappa"] - back["kappa"]),
                  abs(s * (here["capacity"] - back["capacity"])))
    reflected = max(abs(mirror["chi"] + here["chi"]), abs(s * (mirror["xi"] + here["xi"])),
                    abs(mirror["kappa"] - here["kappa"]), abs(s * (mirror["capacity"] - here["capacity"])))
    shifted = max(abs(cut["kappa"] - here["kappa"]), abs(cut["chi"] - here["chi"]),
                  abs(s * (cut["xi"] - here["xi"])), abs(s * (cut["capacity"] - here["capacity"])))
    return {"adjoint_residual": float(adjoint / scale), "mirror_residual": float(reflected / scale),
            "translation_residual": float(shifted / scale), "chi": here["chi"]}

import numpy as np


def local_directional_impedance(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
) -> dict:
    """Reference implementation."""
    local = source_driven_constitutive_matrix(conductivities, capacities, thicknesses,  # noqa: F821
                                                      laplace_s, 0.0)
    s = complex(laplace_s)
    kappa, chi, xi, cap = local["kappa"], local["chi"], local["xi"], local["capacity"]
    k = np.sqrt(cap * s / kappa)
    if k.real < 0.0:
        k = -k
    inverse = kappa * k
    if abs(inverse - chi) < 1e-14 * abs(inverse) or abs(inverse + chi) < 1e-14 * abs(inverse):
        raise ValueError("a thermal impedance is undefined")
    gap = abs(chi - s * xi) / abs(inverse)
    return {"kappa": kappa, "chi": chi, "xi": xi, "capacity": cap, "local_wavenumber": complex(k),
            "impedance_forward": complex(1.0 / (inverse - chi)), "impedance_backward": complex(1.0 / (-inverse - chi)),
            "pair_residual": float(gap)}

import numpy as np


def _dispersion(conductivities, capacities, thicknesses, s, K):
    out = forced_bloch_cell_response(conductivities, capacities, thicknesses, s, K, 1.0, 0.0, 0.0)  # noqa: F821
    if out["mean_temperature"] == 0:
        raise ValueError("the mean temperature of the heat-source response vanishes")
    return -1.0 / out["mean_temperature"]


def effective_bloch_dispersion_root(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    first_guess: complex,
    second_guess: complex,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation."""
    if isinstance(tolerance, (bool, complex, np.complexfloating)):
        raise ValueError("tolerance must be a real number")
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError):
        raise ValueError("tolerance must be a real number") from None
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and above zero")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be an integer of at least 1")
    try:
        k0, k1 = complex(first_guess), complex(second_guess)
    except (TypeError, ValueError):
        raise ValueError("the starting wavenumbers must be numbers") from None
    if not (np.isfinite(k0) and np.isfinite(k1)) or k0 == k1:
        raise ValueError("the starting wavenumbers must be finite and distinct")
    s = complex(laplace_s)
    d0 = _dispersion(conductivities, capacities, thicknesses, s, k0)
    first = d0
    d1 = _dispersion(conductivities, capacities, thicknesses, s, k1)
    for iteration in range(1, max_iterations + 1):
        if d1 == d0:
            raise ValueError("two successive values of the dispersion function coincide")
        k2 = k1 - d1 * (k1 - k0) / (d1 - d0)
        if abs(k2 - k1) <= tolerance * abs(k2):
            return {"root": complex(k2), "dispersion_at_guess": complex(first), "iterations": iteration}
        k0, d0 = k1, d1
        k1 = k2
        d1 = _dispersion(conductivities, capacities, thicknesses, s, k1)
    raise ValueError("the secant iteration did not converge")

import numpy as np


def willis_heat_coupling_report(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    angular_frequency: float,
    wavenumber: float,
    reference_conductivity: float,
    reference_capacity: float,
    root_tolerance: float,
    max_iterations: int,
    certificate_threshold: float,
) -> dict:
    """Reference implementation."""
    scalars = {}
    for name, value in (("angular_frequency", angular_frequency), ("wavenumber", wavenumber),
                        ("reference_conductivity", reference_conductivity), ("reference_capacity", reference_capacity),
                        ("certificate_threshold", certificate_threshold)):
        if isinstance(value, (complex, np.complexfloating, bool)):
            raise ValueError(f"{name} must be a real number")
        try:
            scalars[name] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a real number") from None
        if not np.isfinite(scalars[name]) or (name != "wavenumber" and scalars[name] <= 0.0):
            raise ValueError(f"{name} must be finite" + ("" if name == "wavenumber" else " and above zero"))
    angular_frequency, wavenumber = scalars["angular_frequency"], scalars["wavenumber"]
    reference_conductivity, reference_capacity = scalars["reference_conductivity"], scalars["reference_capacity"]
    try:
        raw = [np.asarray(v) for v in (thicknesses, conductivities, capacities)]
        if any(np.iscomplexobj(v) or v.dtype == object for v in raw):
            raise TypeError
        h = raw[0].astype(float)
        kappa = raw[1].astype(float) / reference_conductivity
        cap = raw[2].astype(float) / reference_capacity
    except (TypeError, ValueError):
        raise ValueError("the layer data must be arrays of real numbers") from None
    if h.ndim != 1 or h.size == 0 or not np.all(np.isfinite(h)) or np.any(h <= 0.0):
        raise ValueError("thicknesses must be a non-empty one-dimensional array of finite values above zero")
    period = float(np.sum(h))
    h = h / period
    omega_bar = angular_frequency * reference_capacity * period ** 2 / reference_conductivity
    k_bar = wavenumber * period
    s, K = 1j * omega_bar, 1j * k_bar

    transfer = cell_transfer_matrix(kappa, cap, h, s, 0.0)  # noqa: F821
    retrieval = bloch_trace_retrieval(kappa, cap, h, s, 0.0)  # noqa: F821
    effective = source_driven_constitutive_matrix(kappa, cap, h, s, K)  # noqa: F821
    heated = forced_bloch_cell_response(kappa, cap, h, s, K, 1.0, 0.0, 0.0)  # noqa: F821
    dispersion = (effective["kappa"] * K * K + (effective["chi"] - s * effective["xi"]) * K
                  - s * effective["capacity"])
    balance = abs(heated["mean_temperature"] * dispersion + 1.0)
    certificate = willis_symmetry_certificate(kappa, cap, h, s, K, 0.5)  # noqa: F821
    worst = max(certificate["adjoint_residual"], certificate["mirror_residual"], certificate["translation_residual"],
                effective["consistency"], abs(transfer["determinant"] - 1.0), balance)
    if worst > certificate_threshold:
        raise ValueError("the effective matrix failed its symmetry certificate")
    local = local_directional_impedance(kappa, cap, h, s)  # noqa: F821
    seed = local["local_wavenumber"]
    root = effective_bloch_dispersion_root(kappa, cap, h, s, 0.9 * seed, 1.1 * seed,  # noqa: F821
                                                   root_tolerance, max_iterations)
    bloch = retrieval["bloch_wavenumber"]
    image = root["root"] - 2j * np.pi * np.round(root["root"].imag / (2.0 * np.pi) - bloch.imag / (2.0 * np.pi))
    mismatch = abs(image - bloch) / abs(bloch)
    if mismatch > 1e-5:
        raise ValueError("the effective dispersion root disagrees with the trace-formula Bloch wavenumber")
    chi = effective["chi"]
    return {
        "chi_real": float(chi.real), "chi_imag": float(chi.imag),
        "dimensionless_frequency": float(omega_bar), "dimensionless_wavenumber": float(k_bar),
        "adjoint_residual": certificate["adjoint_residual"], "mirror_residual": certificate["mirror_residual"],
        "translation_residual": certificate["translation_residual"], "root_mismatch": float(mismatch),
        "chi": chi, "kappa": effective["kappa"], "xi": effective["xi"], "capacity": effective["capacity"],
        "bloch_wavenumber": bloch, "dispersion_root": image, "retrieval_chi": retrieval["chi"],
        "local_chi": local["chi"], "impedance_forward": local["impedance_forward"],
        "impedance_backward": local["impedance_backward"],
    }
SCICODE_GOLD_EOF
