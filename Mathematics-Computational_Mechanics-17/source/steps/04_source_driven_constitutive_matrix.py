"""
At complex wavenumber $K$ and Laplace variable $s$, return the source-driven effective matrix $L_{\mathrm{eff}}$ of the laminate, the 2 by 2 matrix with rows $(\kappa_{\mathrm{eff}}, \chi_{\mathrm{eff}})$ and $(L_{21}, c_{\mathrm{eff}})$, $L_{21}$ being the coefficient of the mean temperature gradient in the mean entropy, that maps the mean kinematic amplitudes $(K \Theta - \zeta_0, \Theta - \varphi_0)$ to the mean kinetic amplitudes $(Q, H)$ of step 03 for every combination of the three sources, and a consistency figure: the largest modulus of the deviation of $L_{\mathrm{eff}} (K \Theta - \zeta_0, \Theta - \varphi_0)$ from $(Q, H)$ for a unit residual temperature, divided by the largest modulus of $Q$ and $H$ over the responses to a unit heat source, a unit residual gradient and a unit residual temperature. Units are those of step 01.

With kinetic fields $h = (-q, \theta_R \eta)$, kinematic fields $b = (\partial_x \theta, \theta)$ and residual fields $m = (\zeta, \varphi)$, the effective matrix relates ensemble means as $\langle h \rangle = L_{\mathrm{eff}} (\langle b \rangle - m)$. For fields proportional to $e^{K x + s t}$ it depends on both $K$ and $s$; its off-diagonal entries $\chi_{\mathrm{eff}}$ and $L_{21}$ are the bianisotropic couplings of the effective conductor, which the individual layers do not have.

Returns
-------
dict holding the complex scalars kappa, chi, xi and capacity, the entries of $L_{\mathrm{eff}}$ with rows (kappa, chi) and (xi, capacity), and the float consistency.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def source_driven_constitutive_matrix(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    wavenumber: complex,
) -> dict:
    r"""Nonlocal effective constitutive matrix $L_{\mathrm{eff}}(K, s)$ of a periodic laminate from source-driven ensemble means.

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
        Wavenumber $K$ of the dependence $e^{K x}$.

    Returns
    -------
    dict
        Under the keys kappa ($\kappa_{\mathrm{eff}}$), chi ($\chi_{\mathrm{eff}}$), xi ($\xi_{\mathrm{eff}} = L_{21}$), capacity ($c_{\mathrm{eff}}$) and consistency.

    Raises
    ------
    ValueError
        When an input is invalid for the forced cell problem, when the wavenumber is not a number, or when the mean kinematic vectors of the unit heat source and the unit residual gradient are linearly dependent to working precision.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_source_driven_constitutive_matrix(
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
        out = _oracle_forced_bloch_cell_response(conductivities, capacities, thicknesses,  # noqa: F821
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
    return (np.array([3.0, 3.0]), np.array([2.0, 2.0]), np.array([0.3, 0.7]))
def digest(out):
    return (tuple(complex(np.round(out[key].real, 7), np.round(out[key].imag, 7)) for key in ("kappa", "chi", "xi", "capacity")),
            int(out["consistency"] < 1e-10))
def verdict(fn, **kw):
    kap, cap, h = tri()
    args = dict(conductivities=kap, capacities=cap, thicknesses=h, laplace_s=10j, wavenumber=0.7j)
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
            # normal: the four-layer cell of the task at $s = 15 i$, $K = 2.5 i$
            "setup": SETUP,
            "call": "flat(digest(source_driven_constitutive_matrix(*graded(), 15j, 2.5j)))",
            "gold_call": "flat(digest(_oracle_source_driven_constitutive_matrix(*graded(), 15j, 2.5j)))",
        },
        {
            # normal: asymmetric silica, diamond, copper cell at an oscillatory wavenumber
            "setup": SETUP,
            "call": "flat(digest(source_driven_constitutive_matrix(*tri(), 10j, 0.7j)))",
            "gold_call": "flat(digest(_oracle_source_driven_constitutive_matrix(*tri(), 10j, 0.7j)))",
        },
        {
            # edge: the same cell at a complex wavenumber
            "setup": SETUP,
            "call": "flat(digest(source_driven_constitutive_matrix(*tri(), 10j, 1.0 + 0.3j)))",
            "gold_call": "flat(digest(_oracle_source_driven_constitutive_matrix(*tri(), 10j, 1.0 + 0.3j)))",
        },
        {
            # boundary: the local limit $K = 0$ of the asymmetric cell, which keeps a coupling
            "setup": SETUP,
            "call": "flat(digest(source_driven_constitutive_matrix(*tri(), 10j, 0.0)))",
            "gold_call": "flat(digest(_oracle_source_driven_constitutive_matrix(*tri(), 10j, 0.0)))",
        },
        {
            # boundary: a symmetric cell, whose coupling is odd in $K$
            "setup": SETUP,
            "call": "flat(digest(source_driven_constitutive_matrix(*bi(), 10j, 0.9j)))",
            "gold_call": "flat(digest(_oracle_source_driven_constitutive_matrix(*bi(), 10j, 0.9j)))",
        },
        {
            # boundary: the symmetric cell at $K = 0$, zero coupling
            "setup": SETUP,
            "call": "flat(digest(source_driven_constitutive_matrix(*bi(), 10j, 0.0)))",
            "gold_call": "flat(digest(_oracle_source_driven_constitutive_matrix(*bi(), 10j, 0.0)))",
        },
        {
            # boundary: a homogeneous cell, diagonal matrix
            "setup": SETUP,
            "call": "flat(digest(source_driven_constitutive_matrix(*hom(), 5.0j, 1.2 + 0.5j)))",
            "gold_call": "flat(digest(_oracle_source_driven_constitutive_matrix(*hom(), 5.0j, 1.2 + 0.5j)))",
        },
        {
            # edge: a large damped Laplace variable with a large decaying wavenumber
            "setup": SETUP,
            "call": "flat(digest(source_driven_constitutive_matrix(*tri(), 3.0 + 60j, 5.0 - 2.0j)))",
            "gold_call": "flat(digest(_oracle_source_driven_constitutive_matrix(*tri(), 3.0 + 60j, 5.0 - 2.0j)))",
        },
        {
            # invalid input: a wavenumber that is not a number
            "setup": SETUP,
            "call": "verdict(source_driven_constitutive_matrix, wavenumber=None)",
            "gold_call": "verdict(_oracle_source_driven_constitutive_matrix, wavenumber=None)",
        },
        {
            # invalid input: a wavenumber given as text
            "setup": SETUP,
            "call": "verdict(source_driven_constitutive_matrix, wavenumber=\"abc\")",
            "gold_call": "verdict(_oracle_source_driven_constitutive_matrix, wavenumber=\"abc\")",
        },
        {
            # invalid input: a zero heat capacity
            "setup": SETUP,
            "call": "verdict(source_driven_constitutive_matrix, capacities=np.array([1.65, 0.0, 3.45]))",
            "gold_call": "verdict(_oracle_source_driven_constitutive_matrix, capacities=np.array([1.65, 0.0, 3.45]))",
        },
        {
            # invalid input: a wavenumber resonant in the copper layer
            "setup": SETUP,
            "call": "verdict(source_driven_constitutive_matrix, wavenumber=np.sqrt(3.45 * 10j / 400.0))",
            "gold_call": "verdict(_oracle_source_driven_constitutive_matrix, wavenumber=np.sqrt(3.45 * 10j / 400.0))",
        },
    ]
