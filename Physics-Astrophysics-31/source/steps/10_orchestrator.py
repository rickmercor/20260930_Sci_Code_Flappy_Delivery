"""
Step 10 (FINAL ORCHESTRATOR): the two-channel radiative balance height.

Contract
--------
Run the whole chain end to end and return one scalar: the lowest height at
which the total wave heating deposited by BOTH channels first rises to equal
the local optically thin radiative loss rate of the bundle.

The chain is: the prescribed background on a uniform height grid (step 01);
the cross-sectional density split (step 02); the two propagation speeds
(step 03); the two perpendicular correlation lengths (step 04); the
dissipation closures (step 05); the steady-state profile and deposition rate of
the transverse displacement channel (step 06); the same for the two-population
Alfven channel (step 07); the radiative loss rate (step 08); and the first
crossing of their sum with the losses (step 09).

The driver composes the functions of steps 01-09; it does not re-implement
them. Called with no arguments it runs the benchmark configuration of the
problem statement.

Conventions
-----------
The grid is uniform with $n_points$ nodes spanning 0 to $z_top$ Mm. Units
throughout are those of the earlier steps. Before returning, certify that both
channels deposit a non-negative rate everywhere, that the total heating is
below the losses at the base and above them at the top, and that the returned
height lies strictly inside the domain; raise ValueError if any of these fails.

Returns
-------
float
The balance height in Mm.

Returns
-------
A Python float: the balance height in Mm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def two_channel_balance_height(
        n_points: int = 4001, z_top: float = 120.0,
        n_base: float = 1.5, H: float = 42.0, B0: float = 12.5,
        R0: float = 0.8, zeta0: float = 3.6, Lzeta: float = 400.0,
        T_base: float = 0.62, dT: float = 0.83, LT: float = 45.0,
        A_He: float = 0.1, f: float = 0.16, lambda0: float = 1.15,
        lambda_exponent: float = -0.5, W_kink_base: float = 0.9,
        W_alfven_out_base: float = 0.5, W_alfven_in_top: float = 0.025,
        tol: float = 1e-15, itmax: int = 500) -> float:
    '''Height at which two-channel wave heating first balances radiative losses.

    Runs the whole chain: the prescribed background, the cross-sectional
    density structure, the two propagation speeds, the two perpendicular
    correlation lengths, the dissipation closures, the steady-state profile of
    each wave channel, the radiative losses, and the first crossing of total
    heating with losses.

    Called with no arguments it runs the benchmark configuration of the
    problem statement.

    Parameters
    ----------
    n_points : int, optional
        Number of nodes on the uniform height grid. Integer, at least 4.
    z_top : float, optional
        Top of the domain in Mm. Finite and strictly positive.
    n_base : float, optional
        Base hydrogen number density in 1e15 m^-3. Finite and > 0.
    H : float, optional
        Density scale height in Mm. Finite and > 0.
    B0 : float, optional
        Base field strength in G. Finite and > 0.
    R0 : float, optional
        Base strand radius in Mm. Finite and > 0.
    zeta0 : float, optional
        Base density contrast. Finite and > 1.
    Lzeta : float, optional
        Relaxation length of the contrast in Mm. Finite and > 0.
    T_base : float, optional
        Base temperature in MK. Finite and > 0.
    dT : float, optional
        Temperature rise across the domain in MK. Finite and >= 0.
    LT : float, optional
        Height scale of the temperature rise in Mm. Finite and > 0.
    A_He : float, optional
        Helium-to-hydrogen abundance ratio. Finite and >= 0.
    f : float, optional
        Strand-interior area filling factor, 0 < f < 1.
    lambda0 : float, optional
        Radiative loss function at 1 MK in 1e-35 W m^3. Finite and > 0.
    lambda_exponent : float, optional
        Power-law index of the radiative loss function. Finite.
    W_kink_base : float, optional
        Transverse-channel energy density injected at the base, in mJ m^-3.
        Finite and non-negative.
    W_alfven_out_base : float, optional
        Outward Alfven energy density injected at the base, in mJ m^-3.
        Finite and non-negative.
    W_alfven_in_top : float, optional
        Inward Alfven energy density entering at the top, in mJ m^-3. Finite
        and non-negative.
    tol : float, optional
        Relative convergence tolerance of the Alfven relaxation. Finite, > 0.
    itmax : int, optional
        Maximum number of relaxation sweeps. Integer, at least 1.

    Returns
    -------
    float
        The balance height in Mm.

    Raises
    ------
    ValueError
        If any certification fails, or if any argument fails the validation of
        the step it is passed to.
    '''
    return z_star  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_two_channel_balance_height(
        n_points: int = 4001, z_top: float = 120.0,
        n_base: float = 1.5, H: float = 42.0, B0: float = 12.5,
        R0: float = 0.8, zeta0: float = 3.6, Lzeta: float = 400.0,
        T_base: float = 0.62, dT: float = 0.83, LT: float = 45.0,
        A_He: float = 0.1, f: float = 0.16, lambda0: float = 1.15,
        lambda_exponent: float = -0.5, W_kink_base: float = 0.9,
        W_alfven_out_base: float = 0.5, W_alfven_in_top: float = 0.025,
        tol: float = 1e-15, itmax: int = 500) -> float:
    if not isinstance(n_points, (int, np.integer)) or int(n_points) < 4:
        raise ValueError("n_points must be an integer of at least 4")
    top = float(z_top)
    if not np.isfinite(top) or top <= 0.0:
        raise ValueError("z_top must be finite and strictly positive")

    z = np.linspace(0.0, top, int(n_points))

    background = _oracle_stratified_background(
        z, n_base, H, B0, R0, zeta0, Lzeta, T_base, dT, LT, A_He)
    n_H, rho_avg, temperature, field, radius, contrast = background

    components = _oracle_cross_section_structure(rho_avg, contrast, f)
    rho_e, rho_i = components

    speeds = _oracle_channel_speeds(field, rho_avg, rho_i, rho_e)
    v_alfven, v_kink = speeds

    lengths = _oracle_perpendicular_correlation_lengths(
        radius, contrast, f, temperature, field)
    L_kink, L_alfven = lengths

    kink = _oracle_transverse_channel_profile(
        z, v_kink, rho_avg, rho_e, L_kink, W_kink_base)
    alfven = _oracle_alfven_channel_profile(
        z, v_alfven, rho_avg, L_alfven, W_alfven_out_base, W_alfven_in_top,
        tol, itmax)

    # the heating is re-formed from the closures of step 05 on the relaxed
    # profiles, so every earlier step is on the path to the answer
    rates = _oracle_deposition_rates(
        kink[0], alfven[0], alfven[1], rho_avg, rho_e, L_kink, L_alfven)
    heating = rates[0] + rates[1] + rates[2]
    if not np.allclose(heating, kink[1] + alfven[2], rtol=1e-12, atol=0.0):
        raise ValueError("the two channels disagree with the closures of step 05")

    losses = _oracle_radiative_loss_profile(
        n_H, temperature, rho_avg, rho_e, rho_i, f, lambda0, lambda_exponent)

    if np.any(kink[1] < 0.0) or np.any(alfven[2] < 0.0):
        raise ValueError("a channel returned a negative deposition rate")
    if not heating[0] < losses[0]:
        raise ValueError("heating already exceeds the losses at the base")
    if not heating[-1] > losses[-1]:
        raise ValueError("heating never reaches the losses inside the domain")

    z_star = _oracle_first_balance_height(z, heating, losses)
    if not (z[0] < z_star < z[-1]):
        raise ValueError("the balance height is not strictly inside the domain")
    return float(z_star)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    '''Differential test cases for this step.'''
    benchmark = "import numpy as np\nkwargs = {'n_points': 801}\n"
    coarse = "import numpy as np\nkwargs = {'n_points': 241, 'z_top': 120.0}\n"
    edge = (
        "import numpy as np\n"
        "def _code(fn):\n"
        "    try:\n"
        "        fn(n_points=3)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )
    return [
        {"setup": benchmark,
         "call": "two_channel_balance_height(**kwargs)",
         "gold_call": "_oracle_two_channel_balance_height(**kwargs)"},
        {"setup": coarse,
         "call": "two_channel_balance_height(**kwargs)",
         "gold_call": "_oracle_two_channel_balance_height(**kwargs)"},
        {"setup": edge,
         "call": "_code(two_channel_balance_height)",
         "gold_call": "_code(_oracle_two_channel_balance_height)"},
    ]
