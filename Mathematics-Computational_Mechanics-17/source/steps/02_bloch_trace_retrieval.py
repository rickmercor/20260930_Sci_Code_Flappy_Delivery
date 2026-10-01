"""
From the transfer matrix of the period beginning at $x = \mathrm{start}$ (step 01), return the Bloch wavenumber $K_B$ of the free thermal waves $p(x) e^{\pm K_B x + s t}$ of the laminate, $p$ periodic, taking the root with non-negative real part and with the imaginary part of $K_B l$ in $(-\pi, \pi]$, and the parameters $\kappa_R$, $c_R$ and $\chi_R$ of the uniform bianisotropic slab of length $l$ whose entropy coupling is $\chi_R / s$, whose transfer matrix equals that of the period and whose wavenumber equals that $K_B$. Units are those of step 01, and $l$ is the sum of the thicknesses.

The slab is a uniform medium of length $l$ obeying the local bianisotropic relations $-q = \kappa_R \, \partial_x \theta + \chi_R \, \theta$ and $\theta_R \eta = (\chi_R / s) \, \partial_x \theta + c_R \, \theta$ together with energy conservation $-\partial_x q = s \, \theta_R \eta$, where $\theta_R \eta$ is the reference temperature times the entropy increment per unit volume; its state vector is $(-q, \theta)$ as in step 01. The retrieved parameters describe the chosen interval and change when the period is cut elsewhere.

Returns
-------
dict holding the complex scalars bloch_wavenumber ($K_B$), kappa ($\kappa_R$), capacity ($c_R$), chi ($\chi_R$) and xi (the entropy coupling $\chi_R / s$).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bloch_trace_retrieval(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    start: float,
) -> dict:
    r"""Bloch wavenumber and one-period transfer-matrix retrieval of local bianisotropic parameters.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities $\kappa_j$ in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities $c_j$.
    thicknesses : np.ndarray
        Layer thicknesses $h_j$; the period $l$ is their sum.
    laplace_s : complex
        Laplace variable $s$.
    start : float
        Position at which the period begins.

    Returns
    -------
    dict
        Under the keys bloch_wavenumber ($K_B$), kappa ($\kappa_R$), capacity ($c_R$), chi ($\chi_R$) and xi ($\xi_R = \chi_R / s$).

    Raises
    ------
    ValueError
        When an input is invalid for the transfer matrix, or when the period transfer matrix $T$ has a vanishing lower-left entry or $\operatorname{tr} T = \pm 2$, at which the retrieval is undefined.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bloch_trace_retrieval(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    start: float,
) -> dict:
    """Reference implementation."""
    out = _oracle_cell_transfer_matrix(conductivities, capacities, thicknesses, laplace_s, start)  # noqa: F821
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
def hom3():
    return (np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 3.0]), np.array([0.2, 0.5, 0.3]))
def digest(out):
    return tuple(complex(np.round(out[key].real, 7), np.round(out[key].imag, 7)) for key in ("bloch_wavenumber", "kappa", "capacity", "chi", "xi"))
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
            "call": "flat(digest(bloch_trace_retrieval(*graded(), 15j, 0.0)))",
            "gold_call": "flat(digest(_oracle_bloch_trace_retrieval(*graded(), 15j, 0.0)))",
        },
        {
            # normal: asymmetric silica, diamond, copper cell at dimensionless frequency 10
            "setup": SETUP,
            "call": "flat(digest(bloch_trace_retrieval(*tri(), 10j, 0.0)))",
            "gold_call": "flat(digest(_oracle_bloch_trace_retrieval(*tri(), 10j, 0.0)))",
        },
        {
            # edge: the same cell cut inside the diamond layer
            "setup": SETUP,
            "call": "flat(digest(bloch_trace_retrieval(*tri(), 10j, 0.2)))",
            "gold_call": "flat(digest(_oracle_bloch_trace_retrieval(*tri(), 10j, 0.2)))",
        },
        {
            # edge: the same cell cut at 0.15, silica at both ends, a nearly vanishing coupling
            "setup": SETUP,
            "call": "flat(digest(bloch_trace_retrieval(*tri(), 10j, 0.15)))",
            "gold_call": "flat(digest(_oracle_bloch_trace_retrieval(*tri(), 10j, 0.15)))",
        },
        {
            # boundary: symmetric silica, diamond, silica cell cut at its end
            "setup": SETUP,
            "call": "flat(digest(bloch_trace_retrieval(*bi(), 10j, 0.0)))",
            "gold_call": "flat(digest(_oracle_bloch_trace_retrieval(*bi(), 10j, 0.0)))",
        },
        {
            # boundary: the symmetric cell cut at its midpoint
            "setup": SETUP,
            "call": "flat(digest(bloch_trace_retrieval(*bi(), 10j, 0.5)))",
            "gold_call": "flat(digest(_oracle_bloch_trace_retrieval(*bi(), 10j, 0.5)))",
        },
        {
            # boundary: a homogeneous period
            "setup": SETUP,
            "call": "flat(digest(bloch_trace_retrieval(*hom3(), 4.0j, 0.37)))",
            "gold_call": "flat(digest(_oracle_bloch_trace_retrieval(*hom3(), 4.0j, 0.37)))",
        },
        {
            # invalid input: a negative conductivity
            "setup": SETUP,
            "call": "verdict(bloch_trace_retrieval, conductivities=np.array([1.0, -2.0, 3.0]))",
            "gold_call": "verdict(_oracle_bloch_trace_retrieval, conductivities=np.array([1.0, -2.0, 3.0]))",
        },
        {
            # invalid input: layer arrays of unequal length
            "setup": SETUP,
            "call": "verdict(bloch_trace_retrieval, capacities=np.array([1.0, 2.0]))",
            "gold_call": "verdict(_oracle_bloch_trace_retrieval, capacities=np.array([1.0, 2.0]))",
        },
        {
            # invalid input: a non-finite start
            "setup": SETUP,
            "call": "verdict(bloch_trace_retrieval, start=np.nan)",
            "gold_call": "verdict(_oracle_bloch_trace_retrieval, start=np.nan)",
        },
    ]
