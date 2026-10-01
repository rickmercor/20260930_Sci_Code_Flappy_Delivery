"""
Evaluate the right-polarized resistive whistler mode and one-period attenuation.

Use the exp(i(kx-omega t)) convention and write omega = omega_r + i*omega_i. Define



omega_H = ion_skin_depth * wavenumber^2 * background_field / density,

omega_A^2 = wavenumber^2 * background_field^2 / density,

D = sqrt(omega_H^2 + 4*omega_A^2).



The first-order resistive right-polarized whistler frequency is



omega_r = (omega_H + D) / 2,

omega_i = -(resistivity * wavenumber^2 / 2) * (1 + omega_H / D).



Then compute



period = 2*pi / omega_r,

one_period_amplitude = exp(omega_i * period).



Return [omega_r, omega_i, period, one_period_amplitude].

Returns
-------
Return one length-4 real NumPy array in documented order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def resistive_whistler_mode(density, background_field, ion_skin_depth,
                            resistivity, wavenumber):
    """Evaluate one right-polarized resistive whistler mode.

    Parameters
    ----------
    density : float
        Positive modal density.
    background_field : float
        Positive background magnetic field.
    ion_skin_depth : float
        Nonnegative Hall coefficient.
    resistivity : float
        Nonnegative resistive coefficient.
    wavenumber : float
        Positive modal wavenumber.
    Returns
    -------
    ndarray, shape (4,)
        [Re(omega),Im(omega),period,one_period_amplitude].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_resistive_whistler_mode(density, background_field, ion_skin_depth,
                                    resistivity, wavenumber):
    """Evaluate the right-polarized first-order resistive dispersion relation (55)."""
    import math
    import numpy as np
    rho = float(density)
    h0 = float(background_field)
    di = float(ion_skin_depth)
    r = float(resistivity)
    k = float(wavenumber)
    if not all(math.isfinite(x) for x in (rho, h0, di, r, k)):
        raise ValueError("inputs must be finite")
    if rho <= 0 or h0 <= 0 or di < 0 or r < 0 or k <= 0:
        raise ValueError("invalid physical domain")
    hall = di * k * k * h0 / rho
    alfven2 = k * k * h0 * h0 / rho
    root = math.sqrt(hall * hall + 4.0 * alfven2)
    real = 0.5 * (hall + root)
    imag = -0.5 * r * k * k * (1.0 + hall / root)
    period = 2.0 * math.pi / real
    return np.array([real, imag, period, math.exp(imag * period)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'resistive_whistler_mode(1.,.4,.5,.004,3.)', 'gold_call': '_oracle_resistive_whistler_mode(1.,.4,.5,.004,3.)'}, {'setup': '', 'call': 'resistive_whistler_mode(.35,.7,1.,.008,5.)', 'gold_call': '_oracle_resistive_whistler_mode(.35,.7,1.,.008,5.)'}, {'setup': '', 'call': 'resistive_whistler_mode(.8,.3,0.,0.,2.5)', 'gold_call': '_oracle_resistive_whistler_mode(.8,.3,0.,0.,2.5)'}, {'setup': '', 'call': 'resistive_whistler_mode(.6,.9,1.2,.02,7.)', 'gold_call': '_oracle_resistive_whistler_mode(.6,.9,1.2,.02,7.)'}, {'setup': '', 'call': 'resistive_whistler_mode(1.4,.2,.1,.001,1.5)', 'gold_call': '_oracle_resistive_whistler_mode(1.4,.2,.1,.001,1.5)'}, {'setup': '', 'call': 'resistive_whistler_mode(.25,1.1,1.5,.015,8.)', 'gold_call': '_oracle_resistive_whistler_mode(.25,1.1,1.5,.015,8.)'}, {'setup': '', 'call': 'resistive_whistler_mode(2.,.6,.7,.003,4.5)', 'gold_call': '_oracle_resistive_whistler_mode(2.,.6,.7,.003,4.5)'}]
