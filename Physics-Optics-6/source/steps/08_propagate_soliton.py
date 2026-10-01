"""
Integrate the driven, damped, dispersive cavity with its full delayed nonlinear response until the circulating pulse reaches a stationary state, and return the comb power spectrum.

The intracavity field obeys an equation with three distinct pieces: a linear one that is diagonal in the mode index, a constant drive, and a nonlinear one that is diagonal in fast time. The nonlinear piece multiplies the field by a purely real quantity, the response of the medium to the instantaneous intensity, so over a fixed step it is an exact phase rotation that leaves the local power untouched.

The response has two channels. One follows the intensity instantaneously and carries the complementary fraction of the nonlinearity; the other is the circular convolution of the intensity with the sampled delayed response and carries the remaining fraction. Once the delayed channel is switched on the pulse walks steadily around the round trip rather than sitting still in the fast-time window, so the stationary object to return is the power spectrum and not the field.

Returns
-------
np.ndarray of shape (n_modes,), float64: the comb power spectrum of the advanced field, the squared modulus of the mode amplitudes in the standard transform ordering.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_soliton(psi0: "np.ndarray", kernel_spectrum: "np.ndarray",
                      propagators: "np.ndarray", f_R: float, f_pump: float,
                      dtau: float, n_steps: int) -> "np.ndarray":
    '''Advance the intracavity field by n_steps and return the resulting comb power spectrum.

    The field obeys, in the dimensionless variables of this pipeline,

        dpsi/dtau = -(1 + i*zeta)*psi + i*d2psi/dtheta2
                    + i*psi*((1 - f_R)*|psi|^2 + f_R*(h conv |psi|^2)) + f_pump,

    with theta the fast time in units of tau0 and no numerical factor on the curvature term.
    That equation is what propagators and psi0 must have been built for.

    One step is a linear half step, a nonlinear phase rotation through the full step, and a
    second linear half step. A linear half step multiplies the transformed field by row zero of
    propagators and adds row one multiplied by the transformed drive, the drive being the
    constant f_pump in fast time. The nonlinear phase is dtau multiplied by the medium response,
    which is one minus f_R multiplied by the intensity, plus f_R multiplied by the real part of
    the circular convolution of the intensity with the sampled delayed response.

    propagators must have been built with the same dtau, the same number of samples and the
    same detuning as the field being advanced.

    Parameters
    ----------
    psi0 : np.ndarray
        Complex starting field of shape (n_modes,) on the fast-time grid. Not modified.
    kernel_spectrum : np.ndarray
        Complex array of shape (n_modes,), the scaled transform of the sampled delayed
        response on the same grid.
    propagators : np.ndarray
        Complex array of shape (2, n_modes) holding the field and drive half-step operators.
    f_R : float
        Fraction of the nonlinear response carried by the delayed channel.
    f_pump : float
        Dimensionless drive amplitude, constant in fast time. The dimensionless pump power is
        its square.
    dtau : float
        Integration step in dimensionless slow time.
    n_steps : int
        Number of steps to take.

    Returns
    -------
    spectrum : np.ndarray
        Float array of shape (n_modes,) holding the squared modulus of the mode amplitudes,
        each amplitude being the discrete Fourier transform of the final field divided by the
        number of samples, in the standard transform ordering.
    '''
    return spectrum  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_propagate_soliton(psi0: "np.ndarray", kernel_spectrum: "np.ndarray",
                              propagators: "np.ndarray", f_R: float, f_pump: float,
                              dtau: float, n_steps: int) -> "np.ndarray":
    psi = np.asarray(psi0, dtype=complex).copy()
    n = psi.size
    field_op = propagators[0]
    drive_op = propagators[1]
    drive_hat = np.zeros(n, dtype=complex)
    drive_hat[0] = f_pump * n
    for _ in range(int(n_steps)):
        psi = np.fft.ifft(field_op * np.fft.fft(psi) + drive_op * drive_hat)
        intensity = np.abs(psi) ** 2
        delayed = np.real(np.fft.ifft(kernel_spectrum * np.fft.fft(intensity)))
        psi = psi * np.exp(1j * dtau * ((1.0 - f_R) * intensity + f_R * delayed))
        psi = np.fft.ifft(field_op * np.fft.fft(psi) + drive_op * drive_hat)
    return np.abs(np.fft.fft(psi) / n) ** 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    common = """import numpy as np
tau0 = 97.34287e-15
tau1 = 2.0 * np.pi * 12.2e-15
tau2 = 32.0e-15
window = 980.392e-15 / tau0
n = 128
zeta = 17.795878
dtau = 2.0e-4
kh = _oracle_raman_kernel_spectrum(n, window, tau0, tau1, tau2)
pr = _oracle_lle_propagators(zeta, n, window, dtau)
seed_a = _oracle_soliton_seed(zeta, n, window)
seed_b = _oracle_soliton_seed(zeta, n, window)
def run_and_check(fn, field):
    before = np.array(field, copy=True)
    out = fn(field, kh, pr, FR_CASE, np.sqrt(20.0), dtau, NSTEPS)
    if not np.array_equal(field, before):
        raise AssertionError("psi0 must not be modified")
    return out
"""
    return [
        # Normal: a short run with the delayed channel active, long enough for the spectrum to
        # become visibly asymmetric.
        {
            "setup": common + "FR_CASE, NSTEPS = 0.0217, 2000\n",
            "call": "run_and_check(propagate_soliton, seed_a)",
            "gold_call": "run_and_check(_oracle_propagate_soliton, seed_b)",
        },
        # Boundary: zero steps, which must return the spectrum of the seed itself.
        {
            "setup": common + "FR_CASE, NSTEPS = 0.0217, 0\n",
            "call": "run_and_check(propagate_soliton, seed_a)",
            "gold_call": "run_and_check(_oracle_propagate_soliton, seed_b)",
        },
        # Edge: the delayed channel switched off entirely, so the medium response is purely
        # instantaneous and the spectrum must stay symmetric about the pump.
        {
            "setup": common + "FR_CASE, NSTEPS = 0.0, 1500\n",
            "call": "run_and_check(propagate_soliton, seed_a)",
            "gold_call": "run_and_check(_oracle_propagate_soliton, seed_b)",
        },
    ]
