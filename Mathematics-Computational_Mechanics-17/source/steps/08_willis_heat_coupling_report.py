"""
Given the physical layer data of the cell, listed in order along $+x$, an angular frequency and a wavenumber for fields varying as $e^{i k x + i \omega t}$, and the reference scales, return the normalised source-driven effective matrix at that frequency and wavenumber, whose coupling $\chi_{\mathrm{eff}} l / \kappa_0$ is the reported result, together with the certificates and contrasts of the other stages.

The inputs are normalised by the period $l$, the sum of the thicknesses, by the reference conductivity $\kappa_0$ and by the reference volumetric heat capacity $c_0$: layer data become $\kappa / \kappa_0$, $c / c_0$ and $h / l$, the Laplace variable is $s = i \omega c_0 l^2 / \kappa_0$ and the drive wavenumber is $K = i k l$. Couplings are reported as $\chi_{\mathrm{eff}} l / \kappa_0$ and $L_{21} / (c_0 l)$, $L_{21}$ being the lower-left entry of the matrix of step 04, and the direct terms as $\kappa_{\mathrm{eff}} / \kappa_0$ and $c_{\mathrm{eff}} / c_0$.



The report runs every earlier stage on the normalised cell: the transfer matrix of step 01 and the retrieval of step 02 for the period beginning at the first listed interface; the effective matrix of step 04 at $(K, s)$; the heat-source response of step 03 at $(K, s)$, whose mean temperature $\Theta_r$ enters the check $|\Theta_r D + 1|$ with $D = \kappa_{\mathrm{eff}} K^2 + (\chi_{\mathrm{eff}} - s L_{21}) K - s \, c_{\mathrm{eff}}$ built from that matrix; the certificate of step 05 with the alternative period beginning at $x = l / 2$; the local limit of step 06; and the dispersion root of step 07, seeded at 0.9 and 1.1 times the local wavenumber of step 06. It raises a ValueError when any residual of step 05, the consistency of step 04, $|\det T - 1|$ from step 01 or $|\Theta_r D + 1|$ exceeds the certificate threshold, and when the root, shifted by the multiple of $2 \pi i$ that brings its imaginary part nearest that of $K_B l$, differs from the $K_B l$ of step 02 by more than $10^{-5}$ relative.

Returns
-------
dict holding the floats chi_real and chi_imag (the real and imaginary parts of $\chi_{\mathrm{eff}} l / \kappa_0$), dimensionless_frequency ($\omega c_0 l^2 / \kappa_0$), dimensionless_wavenumber ($k l$), adjoint_residual, mirror_residual, translation_residual and root_mismatch (the relative distance between the shifted dispersion root and $K_B l$), and the complex scalars chi, kappa, xi and capacity (the normalised effective matrix at $(k, \omega)$), bloch_wavenumber ($K_B l$), dispersion_root (the zero of step 07, times $l$, shifted by a multiple of $2 \pi i$ to the image nearest $K_B l$), retrieval_chi (the retrieved $\chi_R l / \kappa_0$ of step 02), local_chi (the normalised $\chi_{\mathrm{eff}}$ at $K = 0$), impedance_forward and impedance_backward (the local impedances of step 06, which in these units are $Z \kappa_0 / l$).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    r"""Report the source-driven bianisotropic coupling $\tilde\chi l / \kappa_0$ of a periodic laminate at a given frequency and wavenumber.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities in $\mathrm{W\,m^{-1}\,K^{-1}}$, in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities in $\mathrm{J\,m^{-3}\,K^{-1}}$.
    thicknesses : np.ndarray
        Layer thicknesses in $\mathrm{m}$.
    angular_frequency : float
        Angular frequency $\omega$ in $\mathrm{rad\,s^{-1}}$.
    wavenumber : float
        Wavenumber $k$ in $\mathrm{rad\,m^{-1}}$ of the dependence $e^{i k x + i \omega t}$.
    reference_conductivity : float
        Reference conductivity $\kappa_0$.
    reference_capacity : float
        Reference volumetric heat capacity $c_0$.
    root_tolerance : float
        Relative tolerance of the dispersion root.
    max_iterations : int
        Largest number of secant updates.
    certificate_threshold : float
        Largest accepted certificate residual.

    Returns
    -------
    dict
        Under the keys chi_real, chi_imag, dimensionless_frequency, dimensionless_wavenumber, adjoint_residual, mirror_residual, translation_residual, root_mismatch, chi, kappa, xi, capacity, bloch_wavenumber, dispersion_root, retrieval_chi, local_chi, impedance_forward and impedance_backward.

    Raises
    ------
    ValueError
        When any input is invalid for the stage that uses it, when a scalar input is not a real number or a layer array holds non-real values, when the frequency, the wavenumber, a reference scale or the certificate threshold is out of range, when a certificate residual exceeds the threshold, or when the dispersion root disagrees with the trace-formula Bloch wavenumber.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_willis_heat_coupling_report(
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

    transfer = _oracle_cell_transfer_matrix(kappa, cap, h, s, 0.0)  # noqa: F821
    retrieval = _oracle_bloch_trace_retrieval(kappa, cap, h, s, 0.0)  # noqa: F821
    effective = _oracle_source_driven_constitutive_matrix(kappa, cap, h, s, K)  # noqa: F821
    heated = _oracle_forced_bloch_cell_response(kappa, cap, h, s, K, 1.0, 0.0, 0.0)  # noqa: F821
    dispersion = (effective["kappa"] * K * K + (effective["chi"] - s * effective["xi"]) * K
                  - s * effective["capacity"])
    balance = abs(heated["mean_temperature"] * dispersion + 1.0)
    certificate = _oracle_willis_symmetry_certificate(kappa, cap, h, s, K, 0.5)  # noqa: F821
    worst = max(certificate["adjoint_residual"], certificate["mirror_residual"], certificate["translation_residual"],
                effective["consistency"], abs(transfer["determinant"] - 1.0), balance)
    if worst > certificate_threshold:
        raise ValueError("the effective matrix failed its symmetry certificate")
    local = _oracle_local_directional_impedance(kappa, cap, h, s)  # noqa: F821
    seed = local["local_wavenumber"]
    root = _oracle_effective_bloch_dispersion_root(kappa, cap, h, s, 0.9 * seed, 1.1 * seed,  # noqa: F821
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
def tri_si(scale=1.0):
    return (np.array([1.38, 719.0, 400.0]), np.array([1.65e6, 1.78e6, 3.45e6]), scale * np.array([0.3e-6, 0.4e-6, 0.3e-6]))
def bi_si():
    return (np.array([1.38, 719.0, 1.38]), np.array([1.65e6, 1.78e6, 1.65e6]), np.array([0.3e-6, 0.4e-6, 0.3e-6]))
def hom_si():
    return (np.array([3.0, 3.0]), np.array([2.0e6, 2.0e6]), np.array([0.4e-6, 0.6e-6]))
KEYS = ("chi", "kappa", "xi", "capacity", "bloch_wavenumber", "retrieval_chi", "local_chi", "impedance_forward", "impedance_backward")
def digest(out):
    return (tuple(complex(np.round(out[key].real, 6), np.round(out[key].imag, 6)) for key in KEYS),
            round(out["chi_real"], 6), round(out["chi_imag"], 6), round(out["dimensionless_frequency"], 9),
            round(out["dimensionless_wavenumber"], 9), int(out["root_mismatch"] < 1e-5))
def verdict(fn, **kw):
    kap, cap, h = tri_si()
    args = dict(conductivities=kap, capacities=cap, thicknesses=h, angular_frequency=1.0e7, wavenumber=7.0e5, reference_conductivity=1.0, reference_capacity=1.0e6, root_tolerance=1e-6, max_iterations=60, certificate_threshold=1e-9)
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
            # normal: silica, diamond, copper laminate with a 1 micrometre period, dimensionless frequency 10 and $k l = 0.7$
            "setup": SETUP,
            "call": "flat(digest(willis_heat_coupling_report(*tri_si(), 1.0e7, 7.0e5, 1.0, 1.0e6, 1e-6, 60, 1e-9)))",
            "gold_call": "flat(digest(_oracle_willis_heat_coupling_report(*tri_si(), 1.0e7, 7.0e5, 1.0, 1.0e6, 1e-6, 60, 1e-9)))",
        },
        {
            # edge: every thickness doubled, the frequency quartered and the wavenumber halved: the same dimensionless problem
            "setup": SETUP,
            "call": "flat(digest(willis_heat_coupling_report(*tri_si(2.0), 2.5e6, 3.5e5, 1.0, 1.0e6, 1e-6, 60, 1e-9)))",
            "gold_call": "flat(digest(_oracle_willis_heat_coupling_report(*tri_si(2.0), 2.5e6, 3.5e5, 1.0, 1.0e6, 1e-6, 60, 1e-9)))",
        },
        {
            # boundary: $k = 0$, the local limit
            "setup": SETUP,
            "call": "flat(digest(willis_heat_coupling_report(*tri_si(), 1.0e7, 0.0, 1.0, 1.0e6, 1e-6, 60, 1e-9)))",
            "gold_call": "flat(digest(_oracle_willis_heat_coupling_report(*tri_si(), 1.0e7, 0.0, 1.0, 1.0e6, 1e-6, 60, 1e-9)))",
        },
        {
            # boundary: a symmetric cell at $k \neq 0$
            "setup": SETUP,
            "call": "flat(digest(willis_heat_coupling_report(*bi_si(), 1.0e7, 9.0e5, 1.0, 1.0e6, 1e-6, 60, 1e-9)))",
            "gold_call": "flat(digest(_oracle_willis_heat_coupling_report(*bi_si(), 1.0e7, 9.0e5, 1.0, 1.0e6, 1e-6, 60, 1e-9)))",
        },
        {
            # boundary: the symmetric cell at $k = 0$, zero coupling
            "setup": SETUP,
            "call": "flat(digest(willis_heat_coupling_report(*bi_si(), 1.0e7, 0.0, 1.0, 1.0e6, 1e-6, 60, 1e-9)))",
            "gold_call": "flat(digest(_oracle_willis_heat_coupling_report(*bi_si(), 1.0e7, 0.0, 1.0, 1.0e6, 1e-6, 60, 1e-9)))",
        },
        {
            # boundary: a homogeneous laminate
            "setup": SETUP,
            "call": "flat(digest(willis_heat_coupling_report(*hom_si(), 4.0e6, 7.0e5, 1.0, 1.0e6, 1e-6, 60, 1e-9)))",
            "gold_call": "flat(digest(_oracle_willis_heat_coupling_report(*hom_si(), 4.0e6, 7.0e5, 1.0, 1.0e6, 1e-6, 60, 1e-9)))",
        },
        {
            # invalid input: a zero angular frequency
            "setup": SETUP,
            "call": "verdict(willis_heat_coupling_report, angular_frequency=0.0)",
            "gold_call": "verdict(_oracle_willis_heat_coupling_report, angular_frequency=0.0)",
        },
        {
            # invalid input: a complex wavenumber
            "setup": SETUP,
            "call": "verdict(willis_heat_coupling_report, wavenumber=7.0e5 + 1.0j)",
            "gold_call": "verdict(_oracle_willis_heat_coupling_report, wavenumber=7.0e5 + 1.0j)",
        },
        {
            # invalid input: a layer array with a complex entry
            "setup": SETUP,
            "call": "verdict(willis_heat_coupling_report, conductivities=np.array([1.38, 719.0 + 1.0j, 400.0]))",
            "gold_call": "verdict(_oracle_willis_heat_coupling_report, conductivities=np.array([1.38, 719.0 + 1.0j, 400.0]))",
        },
        {
            # invalid input: a non-finite reference capacity
            "setup": SETUP,
            "call": "verdict(willis_heat_coupling_report, reference_capacity=np.inf)",
            "gold_call": "verdict(_oracle_willis_heat_coupling_report, reference_capacity=np.inf)",
        },
        {
            # invalid input: a zero iteration limit
            "setup": SETUP,
            "call": "verdict(willis_heat_coupling_report, max_iterations=0)",
            "gold_call": "verdict(_oracle_willis_heat_coupling_report, max_iterations=0)",
        },
    ]
