"""
Implement compute_core_coupling, which evaluates the asymptotic core-cavity
near-degeneracy coupling coefficient gamma_c between two distinct quadrupolar
mixed modes of an evolved solar-like star.

When two mixed modes of adjacent radial orders are separated by a frequency
interval comparable to the rotational splitting, rotation couples them and
their rotational multiplets become asymmetric. The coupling is carried by the
off-diagonal elements of the rotational operator, whose core-cavity part is
the coefficient computed here. It is defined, for modes i and j with
unperturbed angular frequencies omega_i, omega_j and total inertias I_i, I_j
and with L = sqrt(l (l + 1)) = sqrt(6), as

    gamma_c(i, j) = (1 / sqrt(I_i I_j)) * integral over the g-mode cavity of
                    rho r^2 (L^2 - 1) xi_h,i(r) xi_h,j(r) dr,

with the horizontal displacements taken in their JWKB form in the gravity
cavity,

    xi_h(r) = -A / sqrt(L omega) * rho^(-1/2) r^(-3/2) (N^2 / omega^2)^(1/4)
              * sin( L * integral_0^r (N / omega) dr' / r' - pi / 4 ),

where N is the Brunt-Vaisala frequency and A the mode amplitude. Inertias
are taken with the standard inner product I = integral of rho r^2 (xi_r^2 +
L^2 xi_h^2) dr over the star, so that in the g cavity, where xi_r << xi_h,
the g-cavity inertia is I_g = L^2 integral_g rho r^2 xi_h^2 dr; the
amplitude of each mode is normalized so that I_g,i = zeta_i I_i, with the
squared sine of the inertia integral replaced by its average value one half.
In the asymptotic description the g-cavity phase variable s = integral_0^r N
dr' / r' runs from 0 to 2 pi^2 / DeltaPi over the cavity, with DeltaPi =
L * delta_pi the reduced period spacing. Only the inertia (normalization)
integrals are averaged in this way; the coupling integral defining gamma_c
is evaluated exactly over the full range of s, without averaging its
integrand, so the coefficient does not reduce exactly to the diagonal Ledoux
term for coincident modes. The coefficient is symmetric in (i, j) and
dimensionless.

Returns
-------
float, the dimensionless coefficient gamma_c(i, j), symmetric in (i, j)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_core_coupling(nu_i: float, nu_j: float, zeta_i: float, zeta_j: float, delta_pi: float) -> float:
    '''Asymptotic core near-degeneracy coupling coefficient gamma_c between two l = 2 mixed modes.

    Parameters
    ----------
    nu_i : float
        Unperturbed frequency of the first mode, in microhertz, strictly positive.
    nu_j : float
        Unperturbed frequency of the second mode, in microhertz, strictly
        positive and different from nu_i.
    zeta_i : float
        Trapping fraction of the first mode, in (0, 1].
    zeta_j : float
        Trapping fraction of the second mode, in (0, 1].
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive (the reduced spacing sqrt(6) delta_pi is formed
        internally).

    Returns
    -------
    gamma_c : float
        Dimensionless coupling coefficient, symmetric under exchange of the
        two modes.

    Raises
    ------
    ValueError
        If nu_i or nu_j is not strictly positive, if nu_i equals nu_j, if a
        trapping fraction is outside (0, 1], or if delta_pi is not strictly
        positive.
    '''
    return gamma_c

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _degree_norm() -> float:
    """L = sqrt(l (l + 1)) for the quadrupolar modes (l = 2)."""
    return np.sqrt(6.0)


def _oracle_compute_core_coupling(nu_i: float, nu_j: float, zeta_i: float, zeta_j: float, delta_pi: float) -> float:
    """Reference implementation of the asymptotic gamma_c."""
    nu_i, nu_j, zeta_i, zeta_j, delta_pi = (float(v) for v in (nu_i, nu_j, zeta_i, zeta_j, delta_pi))
    if nu_i <= 0.0 or nu_j <= 0.0:
        raise ValueError("mode frequencies must be strictly positive")
    if nu_i == nu_j:
        raise ValueError("the coupling coefficient is defined for two distinct modes")
    if not (0.0 < zeta_i <= 1.0 and 0.0 < zeta_j <= 1.0):
        raise ValueError("trapping fractions must lie in (0, 1]")
    if delta_pi <= 0.0:
        raise ValueError("delta_pi must be strictly positive")
    big_l = _degree_norm()
    reduced_pi = big_l * delta_pi
    f_i, f_j = nu_i * 1e-6, nu_j * 1e-6  # hertz
    prefactor = reduced_pi * (big_l ** 2 - 1.0) / (np.pi * big_l ** 3) * np.sqrt(zeta_i * zeta_j)
    diff_term = f_i * f_j / (f_j - f_i) * np.sin(np.pi * big_l / reduced_pi * (f_j - f_i) / (f_i * f_j))
    sum_term = f_i * f_j / (f_j + f_i) * (np.cos(np.pi * big_l / reduced_pi * (f_j + f_i) / (f_i * f_j)) - 1.0)
    return float(prefactor * (diff_term + sum_term))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    invalid_setup = """
def run_model():
    try:
        compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Normal: the task's two observed modes straddling the acoustic mode ---
        {
            "setup": "nu_i, nu_j = 366.805, 368.675\nzeta_i, zeta_j = 0.6890591277, 0.3454152054\ndelta_pi = 60.850\n",
            "call": "compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
            "gold_call": "_oracle_compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
        },
        # --- Normal: a g-dominated mode paired with the lower observed mode (larger separation) ---
        {
            "setup": "nu_i, nu_j = 359.3149, 366.805\nzeta_i, zeta_j = 0.9858814291, 0.6890591277\ndelta_pi = 60.850\n",
            "call": "compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
            "gold_call": "_oracle_compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
        },
        # --- Boundary: reversed argument order (symmetry of the coefficient) ---
        {
            "setup": "nu_i, nu_j = 368.675, 366.805\nzeta_i, zeta_j = 0.3454152054, 0.6890591277\ndelta_pi = 60.850\n",
            "call": "compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
            "gold_call": "_oracle_compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
        },
        # --- Edge: nearly degenerate pair (separation 1 nHz), where the difference term is
        #     close to its small-argument limit and must be evaluated without loss of accuracy ---
        {
            "setup": "nu_i, nu_j = 380.382, 380.383\nzeta_i, zeta_j = 0.5, 0.5\ndelta_pi = 64.422\n",
            "call": "compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
            "gold_call": "_oracle_compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
        },
        # --- Edge: pure gravity modes one period spacing apart with unit trapping ---
        {
            "setup": "delta_pi = 60.850\nnu_i = 1e6 / (60.850 * 3000.5)\nnu_j = 1e6 / (60.850 * 2999.5)\nzeta_i, zeta_j = 1.0, 1.0\n",
            "call": "compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
            "gold_call": "_oracle_compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
        },
        # --- Normal: modes of adjacent radial orders far apart (the most p-dominated mode of the
        #     task's window paired with the upper observed mode) ---
        {
            "setup": "nu_i, nu_j = 341.4943423495, 368.675\nzeta_i, zeta_j = 0.1309702260, 0.3454152054\ndelta_pi = 60.850\n",
            "call": "compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
            "gold_call": "_oracle_compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
        },
        # --- Normal: two neighbouring modes of a strongly coupled star at low frequency ---
        {
            "setup": "nu_i, nu_j = 150.11226753, 151.42329751\nzeta_i, zeta_j = 0.9204, 0.8877\ndelta_pi = 64.422\n",
            "call": "compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
            "gold_call": "_oracle_compute_core_coupling(nu_i, nu_j, zeta_i, zeta_j, delta_pi)",
        },
        # --- Invalid: identical modes ---
        {
            "setup": "nu_i, nu_j = 366.805, 366.805\nzeta_i, zeta_j = 0.6890591277, 0.6890591277\ndelta_pi = 60.850\n" + invalid_setup,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
