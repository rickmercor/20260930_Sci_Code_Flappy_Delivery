"""
Certify the effective matrix of the laminate at $(K, s)$ by computing it also at $-K$, for the mirrored cell (layers in reverse order) at $-K$, and for the period cut at $x = \mathrm{start}$, and report three relative residuals, all measured on the scaled matrix with rows $(\kappa_{\mathrm{eff}}, \chi_{\mathrm{eff}})$ and $(s L_{21}, s \, c_{\mathrm{eff}})$, $L_{21}$ being the lower-left entry of the matrix of step 04. The adjoint residual is the largest of $|\chi_{\mathrm{eff}}(K) - s L_{21}(-K)|$, $|\kappa_{\mathrm{eff}}(K) - \kappa_{\mathrm{eff}}(-K)|$ and $|s \, c_{\mathrm{eff}}(K) - s \, c_{\mathrm{eff}}(-K)|$; the mirror residual is the largest of $|\chi^{m}(-K) + \chi_{\mathrm{eff}}(K)|$, $|s L_{21}^{m}(-K) + s L_{21}(K)|$, $|\kappa^{m}(-K) - \kappa_{\mathrm{eff}}(K)|$ and $|s \, c^{m}(-K) - s \, c_{\mathrm{eff}}(K)|$, the superscript $m$ marking the mirrored cell; the translation residual is the largest entry-wise modulus of the difference between the scaled matrices of the two cuts. Each is divided by the largest entry modulus of the scaled matrix at $(K, s)$. Units are those of step 01.

All effective matrices are those of step 04. The mirrored cell lists the same layers in reverse order. The alternative period begins at $x = \mathrm{start}$, taken modulo the period, with the layer containing the start split as in step 01.

Returns
-------
dict holding the floats adjoint_residual, mirror_residual and translation_residual, and the complex scalar chi, the coupling $\chi_{\mathrm{eff}}(K)$ of the cell as listed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def willis_symmetry_certificate(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    wavenumber: complex,
    start: float,
) -> dict:
    r"""Adjoint-pair, mirror and translation residuals of the source-driven effective matrix.

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
        Wavenumber $K$.
    start : float
        Position at which the alternative period begins.

    Returns
    -------
    dict
        Under the keys adjoint_residual, mirror_residual, translation_residual and chi.

    Raises
    ------
    ValueError
        When an input is invalid for the effective matrix at $K$ or at $-K$, or when start is not finite.
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


def _oracle_willis_symmetry_certificate(
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
    here = _oracle_source_driven_constitutive_matrix(kappa, cap, h, s, K)  # noqa: F821
    back = _oracle_source_driven_constitutive_matrix(kappa, cap, h, s, -K)  # noqa: F821
    mirror = _oracle_source_driven_constitutive_matrix(kappa[::-1], cap[::-1], h[::-1], s, -K)  # noqa: F821
    cut = _oracle_source_driven_constitutive_matrix(*_slice_cell(kappa, cap, h, y), s, K)  # noqa: F821
    scale = max(abs(here["kappa"]), abs(here["chi"]), abs(s * here["xi"]), abs(s * here["capacity"]))
    adjoint = max(abs(here["chi"] - s * back["xi"]), abs(here["kappa"] - back["kappa"]),
                  abs(s * (here["capacity"] - back["capacity"])))
    reflected = max(abs(mirror["chi"] + here["chi"]), abs(s * (mirror["xi"] + here["xi"])),
                    abs(mirror["kappa"] - here["kappa"]), abs(s * (mirror["capacity"] - here["capacity"])))
    shifted = max(abs(cut["kappa"] - here["kappa"]), abs(cut["chi"] - here["chi"]),
                  abs(s * (cut["xi"] - here["xi"])), abs(s * (cut["capacity"] - here["capacity"])))
    return {"adjoint_residual": float(adjoint / scale), "mirror_residual": float(reflected / scale),
            "translation_residual": float(shifted / scale), "chi": here["chi"]}

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
def single():
    return (np.array([2.0]), np.array([1.0]), np.array([1.0]))
def digest(out):
    return tuple(int(0.0 <= out[key] < 1e-11) for key in ("adjoint_residual", "mirror_residual", "translation_residual")) + (
            complex(np.round(out["chi"].real, 7), np.round(out["chi"].imag, 7)),)
def verdict(fn, **kw):
    kap, cap, h = tri()
    args = dict(conductivities=kap, capacities=cap, thicknesses=h, laplace_s=10j, wavenumber=0.7j, start=0.2)
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
            # normal: the four-layer cell of the task at $s = 15 i$, $K = 2.5 i$, alternative period at mid-period
            "setup": SETUP,
            "call": "flat(digest(willis_symmetry_certificate(*graded(), 15j, 2.5j, 0.5)))",
            "gold_call": "flat(digest(_oracle_willis_symmetry_certificate(*graded(), 15j, 2.5j, 0.5)))",
        },
        {
            # normal: asymmetric three-layer cell, period cut inside the diamond layer
            "setup": SETUP,
            "call": "flat(digest(willis_symmetry_certificate(*tri(), 10j, 0.7j, 0.55)))",
            "gold_call": "flat(digest(_oracle_willis_symmetry_certificate(*tri(), 10j, 0.7j, 0.55)))",
        },
        {
            # edge: a four-layer cell with a complex wavenumber and a cut near the end of the period
            "setup": SETUP,
            "call": "flat(digest(willis_symmetry_certificate(*four(), 2.0 + 25j, 1.5 - 2.0j, 0.97)))",
            "gold_call": "flat(digest(_oracle_willis_symmetry_certificate(*four(), 2.0 + 25j, 1.5 - 2.0j, 0.97)))",
        },
        {
            # boundary: the local limit $K = 0$
            "setup": SETUP,
            "call": "flat(digest(willis_symmetry_certificate(*tri(), 10j, 0.0, 0.3)))",
            "gold_call": "flat(digest(_oracle_willis_symmetry_certificate(*tri(), 10j, 0.0, 0.3)))",
        },
        {
            # boundary: a single homogeneous layer, zero coupling
            "setup": SETUP,
            "call": "flat(digest(willis_symmetry_certificate(*single(), 3j, 0.4j, 0.5)))",
            "gold_call": "flat(digest(_oracle_willis_symmetry_certificate(*single(), 3j, 0.4j, 0.5)))",
        },
        {
            # invalid input: a non-finite start
            "setup": SETUP,
            "call": "verdict(willis_symmetry_certificate, start=np.nan)",
            "gold_call": "verdict(_oracle_willis_symmetry_certificate, start=np.nan)",
        },
        {
            # invalid input: a wavenumber whose negative is resonant in the diamond layer
            "setup": SETUP,
            "call": "verdict(willis_symmetry_certificate, wavenumber=-np.sqrt(1.78 * 10j / 719.0))",
            "gold_call": "verdict(_oracle_willis_symmetry_certificate, wavenumber=-np.sqrt(1.78 * 10j / 719.0))",
        },
    ]
