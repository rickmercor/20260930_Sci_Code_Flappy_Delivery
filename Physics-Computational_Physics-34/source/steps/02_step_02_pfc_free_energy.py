"""
Return the discrete phase-field-crystal free energy of a state $\\phi$ on the periodic rectangular box. Two of the four terms are evaluated pointwise and two as spectral quadratic forms, and **the split is graded**: the quartic and quadratic terms are summed on the grid, while $\\int|\\nabla\\phi|^2$ and $\\int(\\Delta\\phi)^2$ are evaluated by Parseval as $-\\int\\phi\\,\\Delta\\phi$ and $\\int\\phi\\,\\Delta^2\\phi$. Forming $|\\nabla\\phi|^2$ pointwise from $\\mathrm{i}k_a\\hat\\phi$ instead is a different discretisation: on an even grid the Nyquist column carries a first derivative that is not the adjoint of the second, so the two disagree by about $5\\times10^{-9}$ on an evolved $64^2$ state - enough to destroy the exact discrete energy law that the whole scheme rests on. This is the *original* energy of the model, with no stabilisation parameter in it.

The classical PFC free energy is a Swift-Hohenberg functional,




$$E[\\phi]=\\int_\\Omega\\Big(\\tfrac14\\phi^4+\\tfrac{\\alpha}{2}\\phi^2-|\\nabla\\phi|^2+\\tfrac12(\\Delta\\phi)^2\\Big)\\,d\\boldsymbol{x},$$




whose linear part $-|\\nabla\\phi|^2+\\tfrac12(\\Delta\\phi)^2$ is minimised at a finite wave number, which is why the minimisers are periodic rather than uniform. The conserved gradient flow of this functional is the sixth-order PFC equation. On the discrete side the volume integral is $h_xh_y\\sum_{\\Omega_h}$, and Parseval turns a quadratic form in derivatives into a weighted sum of $|\\hat\\phi|^2$: for the `numpy` unnormalised transform, $h_xh_y\\sum_{\\Omega_h}fg=h_xh_y(N_xN_y)^{-1}\\sum_{\\boldsymbol{k}}\\hat f\\overline{\\hat g}$.




 **Formulas**




With $\\kappa[\\boldsymbol{p}]=|\\boldsymbol{k}|^2$ from step 1, $\\hat\\phi=\\mathrm{fft2}(\\phi)$ and $w=h_xh_y/(N_xN_y)$,




$$\\int|\\nabla\\phi|^2 = w\\sum_{\\boldsymbol{p}}\\kappa\\,|\\hat\\phi|^2, \\qquad \\int(\\Delta\\phi)^2 = w\\sum_{\\boldsymbol{p}}\\kappa^2\\,|\\hat\\phi|^2,$$




$$E[\\phi]\\;=\\;h_xh_y\\sum_{\\Omega_h}\\Big(\\tfrac14\\phi^4+\\tfrac{\\alpha}{2}\\phi^2\\Big)\\;-\\;\\int|\\nabla\\phi|^2\\;+\\;\\tfrac12\\int(\\Delta\\phi)^2 .$$

Returns
-------
A Python `float`: the discrete free energy. It may be of either sign.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pfc_free_energy(phi: "np.ndarray", L: "float | Sequence[float]",
                    alpha: float) -> float:
    """phi: real 2-D array of shape (Nx, Ny), the atomic density field.
    L: float, or length-2 sequence (Lx, Ly), the per-direction box lengths.
    alpha: float, the temperature parameter 1 - eps.
    Return the discrete PFC free energy E[phi] as a float."""
    # Implement per the formulas above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pfc_free_energy(phi: "np.ndarray", L: "float | Sequence[float]",
                            alpha: float) -> float:
    phi = np.asarray(phi, float)
    if phi.ndim != 2:
        raise ValueError("phi must be a two-dimensional array")
    nv, Lv, hx, hy = _geom(phi, L)
    ksq = _oracle_spectral_wavenumbers(nv, Lv)
    ph = np.fft.fft2(phi)
    pw = np.abs(ph) ** 2
    scale = hx * hy / (nv[0] * nv[1])
    grad2 = scale * np.sum(ksq * pw)            # \int |grad phi|^2 = -\int phi Lap phi
    lap2 = scale * np.sum(ksq ** 2 * pw)        # \int (Lap phi)^2
    bulk = hx * hy * np.sum(0.25 * phi ** 4 + 0.5 * float(alpha) * phi ** 2)
    return float(bulk - grad2 + 0.5 * lap2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: the seed field of the benchmark instance on its production grid.
        {"setup": "import numpy as np\nN = 64\nL = 32.0 * np.pi\n"
                  "x = np.arange(N) * (L / N)\n"
                  "s = (2 * np.pi * x / L)[:, None]\nt = (2 * np.pi * x / L)[None, :]\n"
                  "phi = (-0.27 + 0.06 * np.cos(16 * s + 0.3) * np.cos(9 * t)\n"
                  "       + 0.05 * np.sin(11 * s) * np.cos(13 * t + 0.7)\n"
                  "       + 0.04 * np.cos(7 * s - 14 * t + 1.1)\n"
                  "       + 0.03 * np.sin(19 * s + 5 * t) * np.sin(6 * t)\n"
                  "       + 0.02 * np.cos(23 * s) * np.sin(21 * t + 0.4))\n",
         "call": "pfc_free_energy(phi, L, 0.75)",
         "gold_call": "_oracle_pfc_free_energy(phi, L, 0.75)"},
        # normal: a constant field, for which both derivative terms vanish and the
        # energy is exactly the box area times the pointwise bulk density.
        {"setup": "import numpy as np\nphi = np.full((16, 16), -0.27)\n",
         "call": "pfc_free_energy(phi, 8.0, 0.75)",
         "gold_call": "_oracle_pfc_free_energy(phi, 8.0, 0.75)"},
        # boundary: a field with FULL Nyquist content on an even grid. This is the
        # case in which a pointwise |grad phi|^2 disagrees with the Parseval form;
        # the alternating sign pattern is exactly the Nyquist mode in both axes.
        {"setup": "import numpy as np\nN = 16\ni = np.arange(N)\n"
                  "phi = 0.4 * ((-1.0) ** i)[:, None] * ((-1.0) ** i)[None, :] - 0.27\n",
         "call": "pfc_free_energy(phi, 4.0 * np.pi, 0.9)",
         "gold_call": "_oracle_pfc_free_energy(phi, 4.0 * np.pi, 0.9)"},
        # boundary: a RECTANGULAR grid on an anisotropic box, so the volume element
        # and both wave-number vectors differ between the two directions.
        {"setup": "import numpy as np\n"
                  "phi = -0.27 + 0.1 * np.cos(np.linspace(0, 6.0, 20))[:, None] "
                  "* np.sin(np.linspace(0, 9.0, 28))[None, :]\n"
                  "Lv = (20.0 * np.pi, 32.0 * np.pi)\n",
         "call": "pfc_free_energy(phi, Lv, 0.6)",
         "gold_call": "_oracle_pfc_free_energy(phi, Lv, 0.6)"},
        # boundary: alpha = 0, which removes the quadratic term entirely and leaves
        # the energy dominated by the negative gradient term.
        {"setup": "import numpy as np\nN = 24\nL = 12.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "phi = 0.3 * np.cos(4 * u)[:, None] + 0.2 * np.sin(3 * u)[None, :]\n",
         "call": "pfc_free_energy(phi, L, 0.0)",
         "gold_call": "_oracle_pfc_free_energy(phi, L, 0.0)"},
        # edge: odd mode counts, where no Nyquist mode exists at all.
        {"setup": "import numpy as np\nu = np.arange(9) * (2 * np.pi / 9)\n"
                  "v = np.arange(7) * (2 * np.pi / 7)\n"
                  "phi = 0.5 * np.cos(u)[:, None] * np.cos(2 * v)[None, :] + 0.1\n",
         "call": "pfc_free_energy(phi, (3.0, 5.0), 1.25)",
         "gold_call": "_oracle_pfc_free_energy(phi, (3.0, 5.0), 1.25)"},
        # edge: a large-amplitude field, where the quartic term dominates and the
        # sign of the answer flips relative to the small-amplitude cases.
        {"setup": "import numpy as np\nu = np.arange(12) * (2 * np.pi / 12)\n"
                  "phi = 3.0 * np.cos(u)[:, None] * np.sin(u)[None, :]\n",
         "call": "pfc_free_energy(phi, 2.0, 0.75)",
         "gold_call": "_oracle_pfc_free_energy(phi, 2.0, 0.75)"},
    ]
