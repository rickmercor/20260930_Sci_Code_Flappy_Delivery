"""
For sources $r = r_0 e^{K x}$, $\zeta = \zeta_0 e^{K x}$ and $\varphi = \varphi_0 e^{K x}$ acting on the periodic laminate whose cell is listed in order along $+x$ from $x = 0$, return the amplitudes of the ensemble means over a uniformly distributed translation of the laminate, the sources being held fixed: $\langle \theta \rangle = \Theta e^{K x}$, $\langle -q \rangle = Q e^{K x}$ and $\langle \theta_R \eta \rangle = H e^{K x}$. Units are those of step 01.

In every layer the residual temperature gradient $\zeta$ and the residual temperature $\varphi$ enter the constitutive relations as $-q = \kappa (\partial_x \theta - \zeta)$ and $\theta_R \eta = c (\theta - \varphi)$, and the heat input $r$ enters energy conservation as $-\partial_x q + r = s \, \theta_R \eta$, where $\theta_R \eta$ is the reference temperature times the entropy increment per unit volume; $\theta$ and $-q$ are continuous at every interface. The laminate at translation $y$ is the cell pattern shifted by $y$, with $y$ uniformly distributed over one period, and each mean is the average over $y$ at fixed $x$. The response is undefined when $K$ lies on the free Bloch dispersion of the laminate or when $\kappa K^2 = s c$ in some layer; the Raises section states the tests applied.

Returns
-------
dict holding the complex scalars mean_temperature ($\Theta$), mean_flux ($Q$, the amplitude of the mean of $-q$) and mean_entropy ($H$, the amplitude of the mean of $\theta_R \eta$).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    r"""Ensemble-mean response of a periodic laminate to a heat source and to residual fields of one wavenumber.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities $\kappa_j$ in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities $c_j$.
    thicknesses : np.ndarray
        Layer thicknesses $h_j$.
    laplace_s : complex
        Laplace variable $s$.
    wavenumber : complex
        Wavenumber $K$ of the source dependence $e^{K x}$.
    heat_source : complex
        Amplitude $r_0$ of the heat input $r$.
    residual_gradient : complex
        Amplitude $\zeta_0$ of the residual temperature gradient $\zeta$.
    residual_temperature : complex
        Amplitude $\varphi_0$ of the residual temperature $\varphi$.

    Returns
    -------
    dict
        Under the keys mean_temperature, mean_flux and mean_entropy.

    Raises
    ------
    ValueError
        When an input is invalid for the transfer matrix, when the wavenumber or a source amplitude is not a number or is not finite, when $|\kappa_j K^2 - s c_j| \le 10^{-12} \max(|\kappa_j K^2|, |s c_j|)$ in some layer $j$, or when the linear solve of the forced cell problem fails or returns non-finite amplitudes, which happens only when the wavenumber lies on the Bloch dispersion to working precision.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_forced_bloch_cell_response(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
def flat(x):
    # flatten to a tuple of plain real terminals, complex numbers split into real and imaginary parts, -0.0 read as 0.0
    if isinstance(x, dict):
        return flat([x[k] for k in sorted(x)])
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, complex):
        return (x.real + 0.0, x.imag + 0.0)
    if isinstance(x, bool):
        return (int(x),)
    if isinstance(x, float):
        return (x + 0.0,)
    return (x,)
"""
    SETUP = """
import numpy as np
def graded():
    return (np.array([1.38, 400.0, 35.0, 148.0]), np.array([1.65, 3.45, 3.06, 1.66]), np.array([0.35, 0.15, 0.30, 0.20]))
def tri():
    return (np.array([1.38, 719.0, 400.0]), np.array([1.65, 1.78, 3.45]), np.array([0.3, 0.4, 0.3]))
def bi():
    return (np.array([1.38, 719.0, 1.38]), np.array([1.65, 1.78, 1.65]), np.array([0.3, 0.4, 0.3]))
def four():
    return (np.array([0.19, 21.9, 148.0, 1.38]), np.array([1.73, 2.36, 1.66, 1.65]), np.array([0.2, 0.3, 0.1, 0.4]))
def hom():
    return (np.array([3.0, 3.0]), np.array([2.0, 2.0]), np.array([0.4, 0.6]))
RES = np.sqrt(1.65 * 10j / 1.38)
def digest(out):
    return tuple(complex(np.round(out[key].real, 7), np.round(out[key].imag, 7)) for key in ("mean_temperature", "mean_flux", "mean_entropy"))
def verdict(fn, **kw):
    kap, cap, h = tri()
    args = dict(conductivities=kap, capacities=cap, thicknesses=h, laplace_s=10j, wavenumber=0.7j, heat_source=1.0, residual_gradient=0.0, residual_temperature=0.0)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT
    return [
        {
            # normal: unit heat source on the four-layer cell of the task at $s = 15 i$, $K = 2.5 i$
            "setup": SETUP,
            "call": "flat(digest(forced_bloch_cell_response(*graded(), 15j, 2.5j, 1.0, 0.0, 0.0)))",
            "gold_call": "flat(digest(_oracle_forced_bloch_cell_response(*graded(), 15j, 2.5j, 1.0, 0.0, 0.0)))",
        },
        {
            # normal: unit residual gradient on the same cell and drive
            "setup": SETUP,
            "call": "flat(digest(forced_bloch_cell_response(*graded(), 15j, 2.5j, 0.0, 1.0, 0.0)))",
            "gold_call": "flat(digest(_oracle_forced_bloch_cell_response(*graded(), 15j, 2.5j, 0.0, 1.0, 0.0)))",
        },
        {
            # normal: unit residual temperature on the same cell and drive
            "setup": SETUP,
            "call": "flat(digest(forced_bloch_cell_response(*graded(), 15j, 2.5j, 0.0, 0.0, 1.0)))",
            "gold_call": "flat(digest(_oracle_forced_bloch_cell_response(*graded(), 15j, 2.5j, 0.0, 0.0, 1.0)))",
        },
        {
            # normal: unit heat source on an asymmetric three-layer cell at a purely oscillatory wavenumber
            "setup": SETUP,
            "call": "flat(digest(forced_bloch_cell_response(*tri(), 10j, 0.7j, 1.0, 0.0, 0.0)))",
            "gold_call": "flat(digest(_oracle_forced_bloch_cell_response(*tri(), 10j, 0.7j, 1.0, 0.0, 0.0)))",
        },
        {
            # boundary: the combination $\zeta_0 = K$, $\varphi_0 = 1$ without heat input
            "setup": SETUP,
            "call": "flat(digest(forced_bloch_cell_response(*tri(), 10j, 1.1 + 0.4j, 0.0, 1.1 + 0.4j, 1.0)))",
            "gold_call": "flat(digest(_oracle_forced_bloch_cell_response(*tri(), 10j, 1.1 + 0.4j, 0.0, 1.1 + 0.4j, 1.0)))",
        },
        {
            # boundary: a homogeneous cell driven by all three sources
            "setup": SETUP,
            "call": "flat(digest(forced_bloch_cell_response(*hom(), 5.0j, 1.1 + 0.4j, 0.7, -0.2 + 0.1j, 0.3j)))",
            "gold_call": "flat(digest(_oracle_forced_bloch_cell_response(*hom(), 5.0j, 1.1 + 0.4j, 0.7, -0.2 + 0.1j, 0.3j)))",
        },
        {
            # boundary: $K = 0$, a uniform heat source
            "setup": SETUP,
            "call": "flat(digest(forced_bloch_cell_response(*tri(), 40j, 0.0, 1.0, 0.0, 0.0)))",
            "gold_call": "flat(digest(_oracle_forced_bloch_cell_response(*tri(), 40j, 0.0, 1.0, 0.0, 0.0)))",
        },
        {
            # boundary: $K = 0$, a uniform residual gradient
            "setup": SETUP,
            "call": "flat(digest(forced_bloch_cell_response(*tri(), 40j, 0.0, 0.0, 1.0, 0.0)))",
            "gold_call": "flat(digest(_oracle_forced_bloch_cell_response(*tri(), 40j, 0.0, 0.0, 1.0, 0.0)))",
        },
        {
            # edge: complex $K$ with a large real part, heat source and residual gradient together
            "setup": SETUP,
            "call": "flat(digest(forced_bloch_cell_response(*tri(), 40j, 6.0 + 2.0j, 1.0, 0.5, 0.0)))",
            "gold_call": "flat(digest(_oracle_forced_bloch_cell_response(*tri(), 40j, 6.0 + 2.0j, 1.0, 0.5, 0.0)))",
        },
        {
            # edge: complex $K$ with a large imaginary part, residual temperature
            "setup": SETUP,
            "call": "flat(digest(forced_bloch_cell_response(*tri(), 40j, -3.0 + 9.0j, 0.0, 0.0, 1.0)))",
            "gold_call": "flat(digest(_oracle_forced_bloch_cell_response(*tri(), 40j, -3.0 + 9.0j, 0.0, 0.0, 1.0)))",
        },
        {
            # invalid input: a negative thickness
            "setup": SETUP,
            "call": "verdict(forced_bloch_cell_response, thicknesses=np.array([0.3, -0.4, 0.3]))",
            "gold_call": "verdict(_oracle_forced_bloch_cell_response, thicknesses=np.array([0.3, -0.4, 0.3]))",
        },
        {
            # invalid input: a wavenumber that is not a number
            "setup": SETUP,
            "call": "verdict(forced_bloch_cell_response, wavenumber=None)",
            "gold_call": "verdict(_oracle_forced_bloch_cell_response, wavenumber=None)",
        },
        {
            # invalid input: a non-finite source amplitude
            "setup": SETUP,
            "call": "verdict(forced_bloch_cell_response, heat_source=np.nan)",
            "gold_call": "verdict(_oracle_forced_bloch_cell_response, heat_source=np.nan)",
        },
        {
            # invalid input: a wavenumber resonant in the first layer
            "setup": SETUP,
            "call": "verdict(forced_bloch_cell_response, wavenumber=RES)",
            "gold_call": "verdict(_oracle_forced_bloch_cell_response, wavenumber=RES)",
        },
        {
            # invalid input: a wavenumber resonant in the first layer to $10^{-13}$ relative
            "setup": SETUP,
            "call": "verdict(forced_bloch_cell_response, wavenumber=RES * (1.0 + 1e-13))",
            "gold_call": "verdict(_oracle_forced_bloch_cell_response, wavenumber=RES * (1.0 + 1e-13))",
        },
    ]
