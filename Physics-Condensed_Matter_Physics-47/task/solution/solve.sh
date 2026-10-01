#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def build_ness_symbols(mass: float, coupling: float, beta_mean: float,
                               bias: float, nq: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np

    parameters = []
    for value in (mass, coupling, beta_mean, bias):
        try:
            scalar = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Thermal and chain parameters must be real scalars") from exc
        if scalar.ndim != 0 or scalar.dtype.kind not in "iuf":
            raise ValueError("Thermal and chain parameters must be real scalars")
        number = float(scalar)
        if not np.isfinite(number):
            raise ValueError("Thermal and chain parameters must be finite")
        parameters.append(number)
    mass, coupling, beta_mean, bias = parameters
    if mass <= 0 or coupling <= 0 or beta_mean <= abs(bias):
        raise ValueError("Require positive mass, coupling and reservoir temperatures")
    if (isinstance(nq, (bool, np.bool_)) or
            not isinstance(nq, (int, np.integer)) or nq < 8 or nq % 2):
        raise ValueError("nq must be an even integer at least eight")

    q = -np.pi + (2.0 * np.pi / nq) * (np.arange(nq) + 0.5)
    omega = np.hypot(mass, 2.0 * np.sqrt(coupling) * np.sin(q / 2.0))
    # Negative exponentials avoid overflow for cold reservoirs; expm1 retains
    # the small thermal denominator for hot reservoirs.
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        arguments = np.array([beta_mean - bias, beta_mean + bias])[:, None] * omega
        decays = np.exp(-arguments)
        denominators = -np.expm1(-arguments)
        coth = 1.0 + 2.0 * decays / denominators
        coth_tangent = (omega * (decays / denominators)) * (2.0 / denominators)
        coth_tangent[1] *= -1.0
        sums = np.stack((coth.sum(axis=0), coth_tangent.sum(axis=0)))
        differences = np.stack((coth[0] - coth[1],
                                coth_tangent[0] - coth_tangent[1]))
        symbols = np.empty((2, nq, 3), dtype=float)
        symbols[:, :, 0] = (sums / 4.0) / omega
        symbols[:, :, 1] = (omega / 4.0) * sums
        symbols[:, :, 2] = np.sign(q) * differences / 4.0
    if not np.all(np.isfinite(symbols)):
        raise ValueError("The requested symbols are not finite in double precision")
    return symbols

def fourier_covariance(symbols: "np.ndarray", positions: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np

    try:
        raw_symbols = np.asarray(symbols)
        raw_positions = np.asarray(positions)
    except (TypeError, ValueError) as exc:
        raise ValueError("symbols and positions must be numeric arrays") from exc
    if (raw_symbols.ndim != 3 or raw_symbols.shape[0] != 2 or
            raw_symbols.shape[2] != 3 or raw_symbols.shape[1] < 8 or
            raw_symbols.shape[1] % 2 or raw_symbols.dtype.kind not in "iuf"):
        raise ValueError("symbols must be real with shape (2, even nq>=8, 3)")
    symbol_values = np.array(raw_symbols, dtype=float, copy=True)
    if not np.all(np.isfinite(symbol_values)) or np.any(symbol_values[0, :, :2] <= 0):
        raise ValueError("symbols must be finite with strictly positive qq and pp values")
    if (raw_positions.ndim != 1 or raw_positions.size < 1 or
            raw_positions.dtype.kind not in "iuf"):
        raise ValueError("positions must be a nonempty real integer-coordinate vector")
    # Inspect object views too, so booleans in a mixed Python sequence are not
    # silently converted to integer coordinates during numeric coercion.
    if any(isinstance(value, (bool, np.bool_))
           for value in np.asarray(positions, dtype=object).flat):
        raise ValueError("Boolean site positions are invalid")
    if (not np.all(np.isfinite(raw_positions)) or
            np.any(raw_positions != np.floor(raw_positions))):
        raise ValueError("positions must contain distinct finite integer coordinates")
    coordinates = [int(value) for value in raw_positions]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("positions must contain distinct finite integer coordinates")
    nq = symbol_values.shape[1]
    if max(coordinates) - min(coordinates) >= nq:
        raise ValueError("The position span must be strictly less than nq")

    q = -np.pi + (2.0 * np.pi / nq) * (np.arange(nq) + 0.5)
    differences = np.array([[left - right for right in coordinates]
                            for left in coordinates], dtype=float)
    cosine = np.cos(q[:, None, None] * differences)
    sine = np.sin(q[:, None, None] * differences)
    m = len(coordinates)
    covariance_jet = np.empty((2, 2 * m, 2 * m), dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        for order in range(2):
            qq = np.einsum("kij,k->ij", cosine, symbol_values[order, :, 0] / nq)
            pp = np.einsum("kij,k->ij", cosine, symbol_values[order, :, 1] / nq)
            mixed = -np.einsum("kij,k->ij", sine, symbol_values[order, :, 2] / nq)
            covariance_jet[order, :m, :m] = qq
            covariance_jet[order, m:, m:] = pp
            covariance_jet[order, :m, m:] = mixed
            covariance_jet[order, m:, :m] = mixed.T
    if not np.all(np.isfinite(covariance_jet)):
        raise ValueError("The requested covariance is not finite in double precision")
    return covariance_jet

def partial_transpose_jet(covariance_jet: "np.ndarray",
                                  transpose_sites: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np

    try:
        raw_jet = np.asarray(covariance_jet)
        raw_sites = np.asarray(transpose_sites)
    except (TypeError, ValueError) as exc:
        raise ValueError("The covariance jet and site indices must be numeric arrays") from exc
    if (raw_jet.ndim != 3 or raw_jet.shape[0] != 2 or
            raw_jet.shape[1] != raw_jet.shape[2] or raw_jet.shape[1] < 2 or
            raw_jet.shape[1] % 2 or raw_jet.dtype.kind not in "iuf"):
        raise ValueError("covariance_jet must be real with shape (2, 2*m, 2*m)")
    jet = np.array(raw_jet, dtype=float, copy=True)
    if (not np.all(np.isfinite(jet)) or
            not np.allclose(jet, jet.transpose(0, 2, 1), rtol=0.0, atol=1e-10)):
        raise ValueError("Both covariance slices must be finite and symmetric")
    try:
        np.linalg.cholesky(jet[0] / 2.0 + jet[0].T / 2.0)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The covariance value must be positive definite") from exc
    m = jet.shape[1] // 2
    if raw_sites.ndim != 1 or raw_sites.dtype.kind not in "iuf":
        raise ValueError("transpose_sites must be a one-dimensional integer-coordinate array")
    if any(isinstance(value, (bool, np.bool_))
           for value in np.asarray(transpose_sites, dtype=object).flat):
        raise ValueError("Boolean transpose indices are invalid")
    sites = np.array(raw_sites, dtype=float, copy=True)
    if (not np.all(np.isfinite(sites)) or np.any(sites != np.floor(sites)) or
            np.any(sites < 0) or np.any(sites >= m) or
            np.unique(sites).size != sites.size):
        raise ValueError("transpose_sites must contain distinct indices in [0, m)")
    signs = np.ones(2 * m, dtype=float)
    signs[m + sites.astype(int)] = -1.0
    transposed_jet = jet * signs[None, :, None] * signs[None, None, :]
    return transposed_jet

def symplectic_absolute_jet(covariance_jet: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from scipy.linalg import solve_sylvester

    raw = np.asarray(covariance_jet)
    if np.iscomplexobj(raw):
        raise ValueError("The covariance jet must be real.")
    try:
        jet = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("The covariance jet must be numerical.") from exc
    if (jet.ndim != 3 or jet.shape[0] != 2 or jet.shape[1] != jet.shape[2]
            or jet.shape[1] < 2 or jet.shape[1] % 2 or not np.isfinite(jet).all()):
        raise ValueError("The covariance jet must have shape (2, 2*m, 2*m).")
    if not np.allclose(jet, jet.transpose(0, 2, 1), atol=1e-10, rtol=0):
        raise ValueError("Both covariance matrices must be symmetric.")
    g, dg = (jet + jet.transpose(0, 2, 1)) / 2
    values, vectors = np.linalg.eigh(g)
    if values[0] <= 0:
        raise ValueError("The covariance must be positive definite.")
    c = (vectors * np.sqrt(values)) @ vectors.T
    dc = solve_sylvester(c, c, dg)
    m = g.shape[0] // 2
    j = np.block([[np.zeros((m, m)), np.eye(m)], [-np.eye(m), np.zeros((m, m))]])
    a = c @ j @ c
    da = dc @ j @ c + c @ j @ dc
    square = a @ a.T
    values, vectors = np.linalg.eigh((square + square.T) / 2)
    if values[0] <= 0:
        raise ValueError("The symplectic absolute square must be positive definite.")
    b = (vectors * np.sqrt(values)) @ vectors.T
    db = solve_sylvester(b, b, da @ a.T + a @ da.T)
    result = np.stack((c, dc, b, db))
    return (result + result.transpose(0, 2, 1)) / 2

def balance_contour_jet(covariance_jet: "np.ndarray", absolute_jet: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from scipy.linalg import solve_sylvester

    arrays = []
    for raw, leading in ((covariance_jet, 2), (absolute_jet, 4)):
        raw = np.asarray(raw)
        if np.iscomplexobj(raw):
            raise ValueError("Both jets must be real.")
        try:
            a = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Both jets must be numerical.") from exc
        if (a.ndim != 3 or a.shape[0] != leading or a.shape[1] != a.shape[2]
                or a.shape[1] < 2 or a.shape[1] % 2 or not np.isfinite(a).all()):
            raise ValueError("Invalid jet shape or entries.")
        if not np.allclose(a, a.transpose(0, 2, 1), atol=1e-10, rtol=0):
            raise ValueError("Jet matrices must be symmetric.")
        arrays.append((a + a.transpose(0, 2, 1)) / 2)
    if arrays[0].shape[1:] != arrays[1].shape[1:]:
        raise ValueError("The two jets must have matching matrix dimensions.")
    g, dg = arrays[0]
    c, dc, b, db = arrays[1]
    if any(np.linalg.eigvalsh(a)[0] <= 0 for a in (g, c, b)):
        raise ValueError("G, C and B must be positive definite.")
    x = np.linalg.solve(b, c)
    metric = c @ x
    dmetric = dc @ x + x.T @ dc - x.T @ db @ x
    values, vectors = np.linalg.eigh((metric + metric.T) / 2)
    if values[0] <= 0:
        raise ValueError("The polar metric must be positive definite.")
    s = (vectors * np.sqrt(values)) @ vectors.T
    ds = solve_sylvester(s, s, (dmetric + dmetric.T) / 2)
    r = np.linalg.solve(s, np.eye(s.shape[0]))
    dr = -r @ ds @ r
    phi = r @ g @ r
    dphi = dr @ g @ r + r @ dg @ r + r @ g @ dr
    result = np.stack((phi, dphi))
    return (result + result.transpose(0, 2, 1)) / 2

def active_log_jet(contour_jet: "np.ndarray") -> "np.ndarray":
    import numpy as np

    raw = np.asarray(contour_jet)
    if np.iscomplexobj(raw):
        raise ValueError("contour_jet must be real")
    try:
        z = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("contour_jet must be numeric") from exc
    if z.ndim != 3 or z.shape[0] != 2 or z.shape[1] != z.shape[2] or z.shape[1] < 2 or z.shape[1] % 2:
        raise ValueError("contour_jet has an invalid shape")
    if not np.isfinite(z).all() or not np.allclose(z, z.transpose(0, 2, 1), atol=1e-10, rtol=0):
        raise ValueError("contour_jet must be finite and symmetric")
    phi, h = (z + z.transpose(0, 2, 1)) / 2
    lam, u = np.linalg.eigh(phi)
    if np.min(lam) <= 0:
        raise ValueError("Phi must be positive definite")
    boundary = np.abs(lam - 0.5) <= 1e-10
    lam[boundary] = 0.5
    values = np.maximum(0.0, -np.log(2 * lam))
    hhat = u.T @ h @ u
    coeff = np.zeros((len(lam), len(lam)))
    for i, a in enumerate(lam):
        for j, b in enumerate(lam):
            if boundary[i] and boundary[j]:
                continue
            if a < 0.5 and b < 0.5:
                gap = a - b
                if abs(gap) <= 1e-12 * max(a, b):
                    coeff[i, j] = -2 / (a + b)
                else:
                    coeff[i, j] = -np.log1p(gap / b) / gap
            elif a != b:
                coeff[i, j] = (values[i] - values[j]) / (a - b)
    tangent = coeff * hhat
    ids = np.flatnonzero(boundary)
    if len(ids):
        block = -2 * hhat[np.ix_(ids, ids)]
        ev, v = np.linalg.eigh(block)
        tangent[np.ix_(ids, ids)] = (v * np.maximum(ev, 0)) @ v.T
    value = (u * values) @ u.T
    direction = u @ tangent @ u.T
    out = np.stack((value, direction))
    return (out + out.transpose(0, 2, 1)) / 2

def negativity_width_response(log_jet: "np.ndarray", positions: "np.ndarray", center: float) -> float:
    import numpy as np
    from numbers import Real

    raw = np.asarray(log_jet)
    sites = np.asarray(positions)
    if (np.iscomplexobj(raw) or sites.dtype.kind not in "iuf"
            or any(isinstance(v, (bool, np.bool_)) for v in np.asarray(positions, dtype=object).flat)):
        raise ValueError("inputs must be real and sites nonboolean")
    try:
        z = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("arrays must be numeric") from exc
    if z.ndim != 3 or z.shape[0] != 2 or z.shape[1] != z.shape[2] or z.shape[1] < 2 or z.shape[1] % 2:
        raise ValueError("log_jet has an invalid shape")
    if not np.isfinite(z).all() or not np.allclose(z, z.transpose(0, 2, 1), atol=1e-10, rtol=0):
        raise ValueError("log_jet must be finite and symmetric")
    m = z.shape[1] // 2
    if sites.shape != (m,) or not np.isfinite(sites).all() or not np.all(sites == np.floor(sites)):
        raise ValueError("positions must be distinct finite integer sites")
    coordinates = [int(v) for v in sites]
    if len(set(coordinates)) != m:
        raise ValueError("positions must be distinct finite integer sites")
    if isinstance(center, (bool, np.bool_)) or not isinstance(center, Real) or not np.isfinite(center):
        raise ValueError("center must be a finite real scalar")
    diagonals = np.diagonal(z, axis1=1, axis2=2)
    e, de = (diagonals[:, :m] + diagonals[:, m:]) / 2
    total = float(e.sum())
    if total <= 1e-14:
        raise ValueError("total negativity must exceed 1e-14")
    origin = int(center) if center == int(center) else float(center)
    with np.errstate(over="ignore", invalid="ignore"):
        r2 = np.array([v - origin for v in coordinates], dtype=float) ** 2
    moment = float(r2 @ e)
    dmoment = float(r2 @ de)
    dtotal = float(de.sum())
    response = float((dmoment * total - moment * dtotal) / total**2)
    if not np.isfinite(response):
        raise ValueError("The spatial response must be representable as a finite float")
    return response

def compute_ness_negativity_response(mass: float, coupling: float, beta_mean: float, bias: float, nq: int, positions: "np.ndarray", transpose_sites: "np.ndarray", center: float) -> float:
    symbols = build_ness_symbols(mass, coupling, beta_mean, bias, nq)
    covariance = fourier_covariance(symbols, positions)
    transposed = partial_transpose_jet(covariance, transpose_sites)
    absolute = symplectic_absolute_jet(transposed)
    contour = balance_contour_jet(transposed, absolute)
    logarithm = active_log_jet(contour)
    return negativity_width_response(logarithm, positions, center)
SCICODE_GOLD_EOF
