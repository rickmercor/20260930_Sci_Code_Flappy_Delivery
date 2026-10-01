"""
Evaluate the effective matrix of step 04 at $K = 0$ and return its entries, the local wavenumber $k_{\mathrm{eff}}$, with non-negative real part, of source-free disturbances of the local effective medium, and the forward and backward thermal impedances $Z_{+}$ and $Z_{-}$, defined as the temperature divided by the heat flux $q$ (not $-q$) of a disturbance $\theta \propto e^{-k_{\mathrm{eff}} x}$ decaying along $+x$ and of a disturbance $\theta \propto e^{k_{\mathrm{eff}} x}$ decaying along $-x$. Report also the pair residual $|\chi_{\mathrm{eff}} - s L_{21}| / |Z^{-1}|$, with $L_{21}$ the lower-left entry and $Z^{-1} = \kappa_{\mathrm{eff}} k_{\mathrm{eff}}$. Units are those of step 01.

The local effective medium uses the entries at $K = 0$ as constant coefficients in $-\langle q \rangle = \kappa_{\mathrm{eff}} \, \partial_x \langle \theta \rangle + \chi_{\mathrm{eff}} \langle \theta \rangle$ and $\langle \theta_R \eta \rangle = L_{21} \, \partial_x \langle \theta \rangle + c_{\mathrm{eff}} \langle \theta \rangle$, together with energy conservation without sources, $-\partial_x \langle q \rangle = s \langle \theta_R \eta \rangle$. Its thermal impedance may depend on the direction of propagation although every layer is direction-independent.

Returns
-------
dict holding the complex scalars kappa, chi, xi, capacity, local_wavenumber, impedance_forward and impedance_backward, and the float pair_residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def local_directional_impedance(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
) -> dict:
    r"""Local ($K = 0$) source-driven effective parameters and the direction-dependent thermal impedances $Z_{\pm}$.

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

    Returns
    -------
    dict
        Under the keys kappa, chi, xi ($\xi_{\mathrm{eff}} = L_{21}$ at $K = 0$), capacity, local_wavenumber ($k_{\mathrm{eff}}$), impedance_forward ($Z_{+}$), impedance_backward ($Z_{-}$) and pair_residual.

    Raises
    ------
    ValueError
        When an input is invalid for the effective matrix at $K = 0$, or when an impedance is undefined because its denominator vanishes.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_local_directional_impedance(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
) -> dict:
    """Reference implementation."""
    local = _oracle_source_driven_constitutive_matrix(conductivities, capacities, thicknesses,  # noqa: F821
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
    return (np.array([4.0]), np.array([2.5]), np.array([1.0]))
KEYS = ("kappa", "chi", "xi", "capacity", "local_wavenumber", "impedance_forward", "impedance_backward")
def digest(out):
    return (tuple(complex(np.round(out[key].real, 7), np.round(out[key].imag, 7)) for key in KEYS), int(out["pair_residual"] < 1e-10))
def verdict(fn, **kw):
    kap, cap, h = tri()
    args = dict(conductivities=kap, capacities=cap, thicknesses=h, laplace_s=10j)
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
            # normal: the four-layer cell of the task at $s = 15 i$
            "setup": SETUP,
            "call": "flat(digest(local_directional_impedance(*graded(), 15j)))",
            "gold_call": "flat(digest(_oracle_local_directional_impedance(*graded(), 15j)))",
        },
        {
            # normal: asymmetric silica, diamond, copper cell, direction-dependent impedance
            "setup": SETUP,
            "call": "flat(digest(local_directional_impedance(*tri(), 10j)))",
            "gold_call": "flat(digest(_oracle_local_directional_impedance(*tri(), 10j)))",
        },
        {
            # boundary: a symmetric cell, no local coupling
            "setup": SETUP,
            "call": "flat(digest(local_directional_impedance(*bi(), 10j)))",
            "gold_call": "flat(digest(_oracle_local_directional_impedance(*bi(), 10j)))",
        },
        {
            # boundary: a single homogeneous layer
            "setup": SETUP,
            "call": "flat(digest(local_directional_impedance(*single(), 6j)))",
            "gold_call": "flat(digest(_oracle_local_directional_impedance(*single(), 6j)))",
        },
        {
            # edge: a slow, strongly damped Laplace variable on a four-layer cell
            "setup": SETUP,
            "call": "flat(digest(local_directional_impedance(*four(), 0.5 + 0.2j)))",
            "gold_call": "flat(digest(_oracle_local_directional_impedance(*four(), 0.5 + 0.2j)))",
        },
        {
            # invalid input: a zero Laplace variable
            "setup": SETUP,
            "call": "verdict(local_directional_impedance, laplace_s=0.0)",
            "gold_call": "verdict(_oracle_local_directional_impedance, laplace_s=0.0)",
        },
        {
            # invalid input: a non-finite conductivity
            "setup": SETUP,
            "call": "verdict(local_directional_impedance, conductivities=np.array([1.38, np.inf, 400.0]))",
            "gold_call": "verdict(_oracle_local_directional_impedance, conductivities=np.array([1.38, np.inf, 400.0]))",
        },
    ]
