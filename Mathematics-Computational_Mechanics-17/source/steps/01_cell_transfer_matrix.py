"""
Work in units in which the period, a reference conductivity and a reference volumetric heat capacity are one. The laminate repeats a cell of layers listed in order along $+x$, the first occupying $0 < x < h_1$. Return the transfer matrix that carries the state vector $(-q, \theta)$ from the start of the period beginning at $x = \mathrm{start}$, taken modulo the period, to the end of that period, together with its determinant and its trace.

Each layer is a homogeneous conductor with conductivity $\kappa$ and volumetric heat capacity $c$. For a time dependence $e^{s t}$, with $s = i \omega$ for a time-harmonic field, the temperature increment $\theta$ and the heat flux $q$ along $+x$ obey Fourier's law $-q = \kappa \, \partial_x \theta$ and energy conservation $-\partial_x q = s \, c \, \theta$ inside every layer, and both are continuous at every interface. When the period begins inside a layer, that layer is split: the part between the start and its right face comes first and the part between its left face and the start comes last.

Returns
-------
dict holding matrix, the complex 2 by 2 array that maps $(-q, \theta)$ at the start of the period to its value one period later, and the complex scalars determinant and trace.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cell_transfer_matrix(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    start: float,
) -> dict:
    r"""Transfer matrix of the flux-temperature state $(-q, \theta)$ across one period of a periodic laminate.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities $\kappa_j$ in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities $c_j$.
    thicknesses : np.ndarray
        Layer thicknesses $h_j$.
    laplace_s : complex
        Laplace variable $s$ of the time dependence $e^{s t}$.
    start : float
        Position at which the period begins.

    Returns
    -------
    dict
        Under the keys matrix, determinant and trace.

    Raises
    ------
    ValueError
        When the layer arrays are not one-dimensional, non-empty, finite, above zero and of equal length, when the Laplace variable is not finite, is zero or has a negative real part, or when start is not finite.
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


def _oracle_cell_transfer_matrix(
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
    return (np.array([5.0, 5.0]), np.array([2.0, 2.0]), np.array([0.25, 0.75]))
def digest(out):
    return (np.round(out["matrix"], 6), np.round(out["determinant"], 9), np.round(out["trace"], 6))
def verdict(fn, **kw):
    kap, cap, h = tri()
    args = dict(conductivities=kap, capacities=cap, thicknesses=h, laplace_s=10j, start=0.0)
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
            # normal: the four-layer cell of the task at $s = 15 i$, period starting at the first interface
            "setup": SETUP,
            "call": "flat(digest(cell_transfer_matrix(*graded(), 15j, 0.0)))",
            "gold_call": "flat(digest(_oracle_cell_transfer_matrix(*graded(), 15j, 0.0)))",
        },
        {
            # normal: asymmetric three-layer cell at a time-harmonic frequency
            "setup": SETUP,
            "call": "flat(digest(cell_transfer_matrix(*tri(), 10j, 0.0)))",
            "gold_call": "flat(digest(_oracle_cell_transfer_matrix(*tri(), 10j, 0.0)))",
        },
        {
            # edge: the period begins inside the first layer, which is split into a leading and a trailing part
            "setup": SETUP,
            "call": "flat(digest(cell_transfer_matrix(*tri(), 10j, 0.15)))",
            "gold_call": "flat(digest(_oracle_cell_transfer_matrix(*tri(), 10j, 0.15)))",
        },
        {
            # edge: a start beyond one period is taken modulo the period
            "setup": SETUP,
            "call": "flat(digest(cell_transfer_matrix(*tri(), 10j, 2.15)))",
            "gold_call": "flat(digest(_oracle_cell_transfer_matrix(*tri(), 10j, 2.15)))",
        },
        {
            # boundary: the period begins exactly on an interior interface
            "setup": SETUP,
            "call": "flat(digest(cell_transfer_matrix(*tri(), 10j, 0.3)))",
            "gold_call": "flat(digest(_oracle_cell_transfer_matrix(*tri(), 10j, 0.3)))",
        },
        {
            # boundary: a homogeneous period, cut inside the second slab
            "setup": SETUP,
            "call": "flat(digest(cell_transfer_matrix(*hom(), 7.5j, 0.6)))",
            "gold_call": "flat(digest(_oracle_cell_transfer_matrix(*hom(), 7.5j, 0.6)))",
        },
        {
            # boundary: a homogeneous period with a real, non-oscillatory Laplace variable
            "setup": SETUP,
            "call": "flat(digest(cell_transfer_matrix(*hom(), 2.0, 0.6)))",
            "gold_call": "flat(digest(_oracle_cell_transfer_matrix(*hom(), 2.0, 0.6)))",
        },
        {
            # invalid input: a negative conductivity
            "setup": SETUP,
            "call": "verdict(cell_transfer_matrix, conductivities=np.array([-1.38, 719.0, 400.0]))",
            "gold_call": "verdict(_oracle_cell_transfer_matrix, conductivities=np.array([-1.38, 719.0, 400.0]))",
        },
        {
            # invalid input: a Laplace variable with negative real part
            "setup": SETUP,
            "call": "verdict(cell_transfer_matrix, laplace_s=-1.0 + 2.0j)",
            "gold_call": "verdict(_oracle_cell_transfer_matrix, laplace_s=-1.0 + 2.0j)",
        },
        {
            # invalid input: a non-finite start
            "setup": SETUP,
            "call": "verdict(cell_transfer_matrix, start=np.inf)",
            "gold_call": "verdict(_oracle_cell_transfer_matrix, start=np.inf)",
        },
    ]
