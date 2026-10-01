"""
The heterogeneous conduction problem is solved by replacing it with an auxiliary problem posed on a homogeneous reference medium of conductivity c0 carrying a polarisation field,

$$J = c0 * E + tau, tau = [c(x) - c0] * E,$$

and iterating until the polarisation is the one the real material produces. Each sweep does two things in the two places each is cheap. In real space the constitutive law is applied point by point, which a heterogeneous material makes trivial and a convolution would not. In Fourier space the equilibrium of the auxiliary problem is imposed mode by mode through the Green operator, which a homogeneous reference medium makes trivial and a heterogeneous one would not. The sweep is

- form the current from the present field, J = c(x) * E, at every grid point;
- form the polarisation tau = J - c0 * E and transform it;
- replace every retained non-zero mode of the field by -Gamma(xi) applied to the transformed polarisation, set every mode outside the retained set to zero, and set the mean mode to the applied mean field;
- transform back.

Two features of that sweep are specific to this construction. The first is that the modes outside the retained set are set to zero rather than left alone: the represented field is a partial Fourier series of order M + 1 per direction, and the polarisation formed from it in real space is not, so its transform carries content above the truncation that the field is not permitted to hold. Discarding that content is the aliasing correction, and without it the update is not consistent with the space the field lives in. The second is that the mean mode is prescribed rather than solved: the operator annihilates it, and it carries the applied loading.

Convergence is measured by how far the current is from equilibrium. Equilibrium means the current is divergence free, and the divergence of a mode is the symbol beta contracted with that mode of the transformed current, so the natural residual is the root mean square divergence normalised by the mean current,

$$residual = [L / (2 * pi)] * sqrt(mean(|div J|^2)) / |mean(J)|,$$

the prefactor making it dimensionless so that the same tolerance means the same thing whatever the cell size. The sum runs over the retained modes and no others. That restriction is not a convenience: the represented field spans exactly those modes, so equilibrium against exactly those modes is what the discrete problem asks for, and the modes above the truncation carry a divergence that no admissible field can cancel. Including them would leave a residual that never falls below a floor set by the truncation itself. The residual is evaluated before any update and again after each one, so the count reported is the number of evaluations rather than the number of updates, and a starting field already in equilibrium reports one.

Three properties of the converged state are worth knowing, because each says something the iteration count does not. The converged field does not depend on c0, which only sets how fast the iteration contracts: at the fixed point every retained mode of the field is parallel to the direction the operator projects onto, the reference terms cancel, and what remains is the statement that the transformed current is orthogonal to that direction at every retained mode. The macroscopic conductivity follows from the mean modes alone,

$$c_num = mean(J) . Ebar / |Ebar|^2,$$

because the mean of a periodic fluctuation vanishes. And the energy of the converged field, evaluated with the same sampled conductivity the iteration used and averaged over the grid, equals c_num times |Ebar| squared exactly, which is the discrete form of the Hill-Mandel relation. It holds by Parseval: the grid average of J . E is the sum over frequencies of the transformed current against the transformed field, and the orthogonality just described annihilates every term of that sum except the mean one. This says nothing about the local energy density, whose own Fourier coefficients are a convolution and are not small; it is the average alone that collapses. That identity is returned as a defect and is the sharpest available check that the fixed point really is one. Its size tracks the stopping residual rather than the machine precision, so it certifies convergence and is not a constant.

Returns
-------
dict, the converged electric field and its transform, the macroscopic conductivity read from the mean mode, and the convergence diagnostics.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def moulinec_suquet_field(
    conductivity,
    symbol,
    green,
    retained,
    reference: float,
    mean_field,
    period: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Iterate the auxiliary problem to a divergence-free current on the retained modes.

    Parameters
    ----------
    conductivity : array
        Sampled conductivities of shape (n, n) in siemens per metre.
    symbol : array
        Symbol beta of shape (2, n, n) in reciprocal metre.
    green : array
        Green operator of shape (2, 2, n, n) in metre per siemens.
    retained : array
        Boolean mask of shape (n, n) marking the retained non-zero modes.
    reference : float
        Reference conductivity in siemens per metre, above zero.
    mean_field : sequence
        Two components of the applied mean electric field in volt per metre.
    period : float
        Cell edge in metre, above zero.
    tolerance : float
        Residual at which the iteration stops, above zero.
    max_iterations : int
        Largest number of sweeps permitted, above zero.

    Returns
    -------
    dict
        Under the keys electric_field, field_transform, iterations, residual, macroscopic_conductivity and hill_mandel_defect. electric_field is a real array of shape (2, n, n) and field_transform its complex unnormalised transform of the same shape, the leading axis running over the two directions as in symbol. The remaining four entries are scalars.

    Raises
    ------
    ValueError
        When the array shapes disagree, when the conductivity is not everywhere above zero, when the reference conductivity, the period or the tolerance fails to be finite and above zero, when the mean field is not two finite components of which at least one is non-zero, when no mode is retained, or when the residual has not fallen below the tolerance after max_iterations sweeps.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _positive_float(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError("%s must be finite and above zero" % label)
    return out


def _oracle_moulinec_suquet_field(
    conductivity,
    symbol,
    green,
    retained,
    reference: float,
    mean_field,
    period: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation."""
    c = np.asarray(conductivity, dtype=float)
    beta = np.asarray(symbol)
    gamma = np.asarray(green)
    keep = np.asarray(retained, dtype=bool)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("conductivity must be a square two-dimensional array")
    n = c.shape[0]
    if beta.shape != (2, n, n):
        raise ValueError("symbol must have shape (2, n, n)")
    if gamma.shape != (2, 2, n, n):
        raise ValueError("green must have shape (2, 2, n, n)")
    if keep.shape != (n, n):
        raise ValueError("retained must have shape (n, n)")
    if not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        raise ValueError("conductivity must be finite and everywhere above zero")
    if not keep.any():
        raise ValueError("at least one mode must be retained")

    c0 = _positive_float(reference, "reference")
    length = _positive_float(period, "period")
    tol = _positive_float(tolerance, "tolerance")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)):
        raise ValueError("max_iterations must be an integer")
    cap = int(max_iterations)
    if cap <= 0:
        raise ValueError("max_iterations must be above zero")

    mean = np.asarray(mean_field, dtype=float).ravel()
    if mean.size != 2 or not np.all(np.isfinite(mean)):
        raise ValueError("mean_field must be two finite components")
    mean_norm_sq = float(mean @ mean)
    if mean_norm_sq <= 0.0:
        raise ValueError("mean_field must not vanish")

    scale = length / (2.0 * np.pi)
    field = np.zeros((2, n, n))
    field[0] = mean[0]
    field[1] = mean[1]
    transform = None
    residual = float("inf")
    taken = 0
    for step in range(1, cap + 1):
        taken = step
        current = c * field
        current_hat = np.fft.fft2(current, axes=(1, 2))
        divergence = beta[0] * current_hat[0] + beta[1] * current_hat[1]
        mean_current = np.array([current_hat[0, 0, 0].real, current_hat[1, 0, 0].real])
        residual = scale * float(np.sqrt(np.sum(np.abs(divergence[keep]) ** 2))
                                 / np.linalg.norm(mean_current))
        if residual < tol:
            break
        if step == cap:
            # the residual has been measured and failed; a further update would be discarded
            break
        polarisation = current_hat - c0 * np.fft.fft2(field, axes=(1, 2))
        transform = np.zeros((2, n, n), complex)
        for a in range(2):
            transform[a] = -(gamma[a, 0] * polarisation[0] + gamma[a, 1] * polarisation[1])
        transform[0][~keep] = 0.0
        transform[1][~keep] = 0.0
        transform[0, 0, 0] = mean[0] * n * n
        transform[1, 0, 0] = mean[1] * n * n
        field = np.real(np.fft.ifft2(transform, axes=(1, 2)))
    if residual >= tol:
        raise ValueError("the residual did not fall below the tolerance within max_iterations sweeps")
    if transform is None:
        # the starting uniform field already satisfies equilibrium on the retained modes
        transform = np.fft.fft2(field, axes=(1, 2))

    mean_current = np.array([current_hat[0, 0, 0].real, current_hat[1, 0, 0].real]) / n ** 2
    c_num = float(mean_current @ mean) / mean_norm_sq
    energy = float(np.mean(c * (field[0] ** 2 + field[1] ** 2)))
    defect = abs(energy - c_num * mean_norm_sq) / (c_num * mean_norm_sq)
    return {
        "electric_field": field,
        "field_transform": transform,
        "iterations": taken,
        "residual": residual,
        "macroscopic_conductivity": c_num,
        "hill_mandel_defect": defect,
    }

# =============================================================================
# TEST CASES
# =============================================================================

FLAT = """
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""

SETUP = """
import numpy as np
def lattice(N, M, L, c0):
    n = N + 1
    idx = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    wave = 2.0 * np.pi * idx / L
    beta = np.zeros((2, n, n), complex)
    beta[0] = wave[:, None]
    beta[1] = wave[None, :]
    keep = np.abs(idx) <= M // 2
    ret = keep[:, None] & keep[None, :]
    ret[0, 0] = False
    w = np.real(np.sum(np.conj(beta) * beta, axis=0))
    g = np.zeros((2, 2, n, n), complex)
    live = w > 0.0
    for a in range(2):
        for b in range(2):
            num = np.conj(beta[a]) * beta[b]
            g[a, b][live] = num[live] / (c0 * w[live])
    return beta, g, ret
def board(N, c1, c2, L):
    n = N + 1
    x = np.arange(n) * (L / n)
    ins = x < 0.5 * L
    return np.where(ins[:, None] & ins[None, :], c2, c1)
"""


def test_cases():
    return [
        {
            # the converged macroscopic conductivity, the Hill-Mandel defect and the
            # field extremes, across three truncations at one grid size
            "setup": SETUP + """
def digest(fn, N, M, c1=1.0, c2=100.0, L=1.0, E=(1.0, 0.0)):
    c0 = 0.5 * (c1 + c2)
    beta, g, ret = lattice(N, M, L, c0)
    out = fn(board(N, c1, c2, L), beta, g, ret, c0, E, L, 1.0e-9, 20000)
    f = np.asarray(out["electric_field"])
    return (round(out["macroscopic_conductivity"], 9), out["iterations"],
            int(out["hill_mandel_defect"] < 1.0e-10),
            round(float(f[0].min()), 9), round(float(f[0].max()), 9),
            round(float(f[1].min()), 9), round(float(f[1].max()), 9))
""" + FLAT,
            "call": "flat((digest(moulinec_suquet_field, 16, 16), digest(moulinec_suquet_field, 16, 12), digest(moulinec_suquet_field, 16, 8)))",
            "gold_call": "flat((digest(_oracle_moulinec_suquet_field, 16, 16), digest(_oracle_moulinec_suquet_field, 16, 12), digest(_oracle_moulinec_suquet_field, 16, 8)))",
        },
        {
            # invariances: the converged macroscopic conductivity must not move with the
            # reference medium, the cell size or the strength of the loading, and a
            # homogeneous material must return its own conductivity exactly
            "setup": SETUP + """
def run(fn, N, M, c1, c2, L, E, c0):
    beta, g, ret = lattice(N, M, L, c0)
    return fn(board(N, c1, c2, L), beta, g, ret, c0, E, L, 1.0e-9, 20000)
def digest(fn):
    a = run(fn, 12, 8, 1.0, 100.0, 1.0, (1.0, 0.0), 50.5)["macroscopic_conductivity"]
    b = run(fn, 12, 8, 1.0, 100.0, 1.0, (1.0, 0.0), 200.0)["macroscopic_conductivity"]
    d = run(fn, 12, 8, 1.0, 100.0, 4.0e-6, (1.0, 0.0), 50.5)["macroscopic_conductivity"]
    e = run(fn, 12, 8, 1.0, 100.0, 1.0, (7.0, 0.0), 50.5)["macroscopic_conductivity"]
    h = run(fn, 12, 8, 3.0, 3.0, 1.0, (1.0, 0.0), 3.0)["macroscopic_conductivity"]
    return (round(a, 9), round(b - a, 9), round(d - a, 9), round(e - a, 9), round(h, 12))
def convention(fn):
    # the residual is evaluated before any update, so a uniform material, whose starting field
    # is already in equilibrium, must report exactly one evaluation and no updates at all
    out = run(fn, 12, 8, 3.0, 3.0, 1.0, (1.0, 0.0), 3.0)
    return (out["iterations"], int(out["residual"] < 1.0e-9))
""" + FLAT,
            "call": "flat((digest(moulinec_suquet_field), convention(moulinec_suquet_field)))",
            "gold_call": "flat((digest(_oracle_moulinec_suquet_field), convention(_oracle_moulinec_suquet_field)))",
        },
        {
            # equilibrium itself: at convergence the transformed current contracted with
            # the symbol must vanish on every retained mode, and the mean mode of the
            # field must be the applied loading
            "setup": SETUP + """
def digest(fn, N, M):
    c0 = 50.5
    L = 1.0
    beta, g, ret = lattice(N, M, L, c0)
    c = board(N, 1.0, 100.0, L)
    out = fn(c, beta, g, ret, c0, (1.0, 0.0), L, 1.0e-9, 20000)
    f = np.asarray(out["electric_field"])
    t = np.asarray(out["field_transform"])
    n = N + 1
    jh = np.fft.fft2(c * f, axes=(1, 2))
    div = beta[0] * jh[0] + beta[1] * jh[1]
    worst = float(np.max(np.abs(div[ret]))) * L / (2.0 * np.pi) / abs(jh[0, 0, 0])
    outside = float(np.max(np.abs(t[0][~ret]))) + float(np.max(np.abs(t[1][~ret])))
    outside -= abs(t[0, 0, 0]) + abs(t[1, 0, 0])
    return (int(worst < 1.0e-8), round(float(t[0, 0, 0].real) / n ** 2, 10),
            round(float(t[1, 0, 0].real) / n ** 2, 10), round(outside, 8),
            int(out["iterations"] > 0), int(out["residual"] < 1.0e-9))
""" + FLAT,
            "call": "flat((digest(moulinec_suquet_field, 10, 10), digest(moulinec_suquet_field, 10, 6)))",
            "gold_call": "flat((digest(_oracle_moulinec_suquet_field, 10, 10), digest(_oracle_moulinec_suquet_field, 10, 6)))",
        },
        {
            "setup": SETUP + """
def verdict(fn, bad=None):
    N, M, L, c0 = 8, 8, 1.0, 50.5
    beta, g, ret = lattice(N, M, L, c0)
    c = board(N, 1.0, 100.0, L)
    kw = dict(conductivity=c, symbol=beta, green=g, retained=ret, reference=c0,
              mean_field=(1.0, 0.0), period=L, tolerance=1.0e-9, max_iterations=20000)
    if bad is not None:
        kw.update(bad)
    try:
        fn(**kw)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(moulinec_suquet_field), verdict(moulinec_suquet_field, {'reference': 0.0}), verdict(moulinec_suquet_field, {'period': -1.0}), verdict(moulinec_suquet_field, {'tolerance': 0.0}), verdict(moulinec_suquet_field, {'max_iterations': 3}), verdict(moulinec_suquet_field, {'mean_field': (0.0, 0.0)}), verdict(moulinec_suquet_field, {'mean_field': (1.0, 0.0, 0.0)}), verdict(moulinec_suquet_field, {'retained': __import__('numpy').zeros((9, 9), bool)})))",
            "gold_call": "flat((verdict(_oracle_moulinec_suquet_field), verdict(_oracle_moulinec_suquet_field, {'reference': 0.0}), verdict(_oracle_moulinec_suquet_field, {'period': -1.0}), verdict(_oracle_moulinec_suquet_field, {'tolerance': 0.0}), verdict(_oracle_moulinec_suquet_field, {'max_iterations': 3}), verdict(_oracle_moulinec_suquet_field, {'mean_field': (0.0, 0.0)}), verdict(_oracle_moulinec_suquet_field, {'mean_field': (1.0, 0.0, 0.0)}), verdict(_oracle_moulinec_suquet_field, {'retained': __import__('numpy').zeros((9, 9), bool)})))",
        },
    ]
