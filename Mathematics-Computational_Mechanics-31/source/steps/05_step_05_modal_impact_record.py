"""
Compute each elastic eigenspace's receptor history under the difference of two Gaussian traction pulses of common amplitude and width, centred at pulse_centre and pulse_centre + pulse_width. Use average-acceleration Newmark integration with beta=1/4 and gamma=1/2, zero initial displacement, velocity and acceleration, and traction evaluated at each step's end time. Include the initial zero sample. Each residue maps its unit-inertia oscillator response into the three receptor components. Retain the finite-record phase of every cluster.

A finite traction pulse excites modes with distinct amplitudes and phases. The retained observation interval controls their subsequent spectral overlap.

Returns
-------
np.ndarray, receptor histories of shape (n_steps+1, n_clusters, 3), in metre.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def modal_impact_record(frequencies, residues, dt, n_steps, pulse_amplitude, pulse_width, pulse_centre) -> np.ndarray:
    r"""Compute each elastic eigenspace's receptor history under the difference of two Gaussian traction pulses of common amplitude and width, centred at pulse_centre and pulse_centre + pulse_width. Use average-acceleration Newmark integration with beta=1/4 and gamma=1/2, zero initial displacement, velocity and acceleration, and traction evaluated at each step's end time. Include the initial zero sample. Each residue maps its unit-inertia oscillator response into the three receptor components. Retain the finite-record phase of every cluster.

    Parameters
    ----------
    frequencies
        Finite positive sorted frequencies in hertz.
    residues
        Finite vector residues (n_clusters, 3) for unit scalar traction.
    dt
        Positive time increment in second.
    n_steps
        Positive integer number of time increments.
    pulse_amplitude
        Finite scalar traction amplitude in pascal.
    pulse_width
        Positive Gaussian width in second.
    pulse_centre
        Finite first-pulse centre in second.

    Returns
    -------
    np.ndarray, receptor histories of shape (n_steps+1, n_clusters, 3), in metre.

    Raises
    ------
    ValueError
        If input dimensions or stated physical constraints are violated.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import block_diag, eigh, null_space


def _oracle_modal_impact_record(frequencies, residues, dt, n_steps, pulse_amplitude, pulse_width, pulse_centre):
    frequencies = np.asarray(frequencies, dtype=float)
    residues = np.asarray(residues, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0 or residues.shape != (frequencies.size, 3):
        raise ValueError('frequencies and residues have incompatible shapes')
    if np.any(frequencies <= 0) or not np.isfinite(frequencies).all() or not np.isfinite(residues).all():
        raise ValueError('modal inputs must be finite with positive frequencies')
    if dt <= 0 or pulse_width <= 0 or not all(np.isfinite(x) for x in (dt, pulse_width, pulse_amplitude, pulse_centre)):
        raise ValueError('invalid pulse parameters')
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError('n_steps must be positive')
    omega2 = (2*np.pi*frequencies)**2
    q, v, a = [np.zeros_like(frequencies) for _ in range(3)]
    history = np.zeros((int(n_steps)+1, frequencies.size, 3))
    coefficient = 0.25*dt*dt
    for step in range(1, int(n_steps)+1):
        t = step*dt
        pulse = pulse_amplitude*(np.exp(-0.5*((t-pulse_centre)/pulse_width)**2)-np.exp(-0.5*((t-pulse_centre-pulse_width)/pulse_width)**2))
        predictor = q + dt*v + coefficient*a
        q_new = (predictor+coefficient*pulse)/(1+coefficient*omega2)
        a_new = pulse-omega2*q_new
        v += 0.5*dt*(a+a_new)
        q, a = q_new, a_new
        history[step] = q[:, None]*residues
    return history

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return numerical test specifications."""
    return [
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
F=np.array([1.1e6,3.2e6,7.8e6])
RES=np.array([[.2,-.4,.1],[-.3,.25,.4],[.7,.1,-.2]])
def summary(out):
    return (out.shape,tuple(np.round(out[[1,20,70,-1]].ravel()*1e9,6)),int(np.all(out[0]==0)))
""",
            "call": 'summary(modal_impact_record(F,RES,3e-9,200,1e6,2e-8,8e-8))',
            "gold_call": 'summary(_oracle_modal_impact_record(F,RES,3e-9,200,1e6,2e-8,8e-8))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
F=np.array([1.1e6,3.2e6,7.8e6])
RES=np.array([[.2,-.4,.1],[-.3,.25,.4],[.7,.1,-.2]])
def summary(out):
    return (out.shape,tuple(np.round(out[[1,20,70,-1]].ravel()*1e9,6)),int(np.all(out[0]==0)))
""",
            "call": 'summary(modal_impact_record(F,RES,3e-9,200,0.,2e-8,8e-8))',
            "gold_call": 'summary(_oracle_modal_impact_record(F,RES,3e-9,200,0.,2e-8,8e-8))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
F=np.array([1.1e6,3.2e6,7.8e6])
RES=np.array([[.2,-.4,.1],[-.3,.25,.4],[.7,.1,-.2]])
def summary(out):
    return (out.shape,tuple(np.round(out[[1,20,70,-1]].ravel()*1e9,6)),int(np.all(out[0]==0)))
""",
            "call": 'summary(modal_impact_record(F,RES,7e-9,95,-2e6,4e-8,0.))',
            "gold_call": 'summary(_oracle_modal_impact_record(F,RES,7e-9,95,-2e6,4e-8,0.))',
        },
    ]
