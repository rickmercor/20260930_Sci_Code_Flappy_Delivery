"""
Electrode self-energies of a biased junction in the wide-band limit

Each molecule is wired to a left and a right electrode. In the wide-band limit an electrode is

summarised by a single energy-independent hybridisation, which is the rate at which an electron

placed on the molecular level leaks into that electrode. The two electrodes attached to one

molecule need not be equivalent, and here they are not.



What the electrodes inject and remove is not fixed by the hybridisations alone. Each electrode

also carries its own chemical potential, set by how the applied bias divides across the junction,

and its own electronic temperature. The occupied and empty parts of the electrode spectrum enter

the lesser and greater components of the electrode self-energy with opposite roles: the lesser

component counts what the electrodes are able to put onto the level at a given frequency, the

greater component what they are able to take off it. Both components are pure imaginary, with the

lesser one having a non-negative imaginary part and the greater one a non-positive one, and their

difference reconstructs the total hybridisation.



A negative electronic temperature is admissible and is not an error. The Fermi-Dirac function

evaluated at a negative temperature stays inside the interval (0, 1) but rises with energy instead

of falling, which is the defining property of a population-inverted reservoir. The routine must

handle that case without special-casing it.

Returns
-------
#     np.ndarray of shape (2, len(w)), complex: lesser self-energy then greater self-energy, in eV  # ============================================================================
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def lead_selfenergies(w: np.ndarray, bias_v: float, temp_k: float, gamma_left: float,
                      gamma_right: float, frac_left: float, e_fermi: float) -> np.ndarray:
    '''Lesser and greater electrode self-energies of one junction on a frequency grid.

    Parameters
    ----------
    w : np.ndarray
        Real frequency grid in eV, strictly increasing.
    bias_v : float
        Applied bias across the junction in volts; the electron charge is one, so a bias of
        V volts moves a chemical potential by V eV. May be zero or negative.
    temp_k : float
        Electronic temperature of both electrodes in kelvin. May be negative, which describes a
        population-inverted reservoir. Must not be zero.
    gamma_left, gamma_right : float
        Wide-band hybridisation of the left and right electrode in eV. Both must be positive.
    frac_left : float
        Fraction of the applied bias dropped on the left side, in [0, 1]. The left chemical
        potential is e_fermi + frac_left * bias_v and the right one is
        e_fermi - (1 - frac_left) * bias_v.
    e_fermi : float
        Common equilibrium Fermi energy in eV.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (2, len(w)); row 0 is the lesser self-energy and row 1 the
        greater self-energy, both in eV.

    Raises
    ------
    ValueError
        If gamma_left or gamma_right is not positive, if temp_k is zero, or if frac_left lies
        outside [0, 1].
    '''
    return result  # placeholder

# ============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_lead_selfenergies(w, bias_v, temp_k, gamma_left, gamma_right, frac_left, e_fermi):
    _KB_EV = 8.617333262e-5
    if not (gamma_left > 0.0) or not (gamma_right > 0.0):
        raise ValueError("hybridisations must be positive")
    if temp_k == 0.0:
        raise ValueError("temp_k must be non-zero")
    if not (0.0 <= frac_left <= 1.0):
        raise ValueError("frac_left must lie in [0, 1]")
    w = np.asarray(w, dtype=float)
    mu_l = e_fermi + frac_left * bias_v
    mu_r = e_fermi - (1.0 - frac_left) * bias_v
    f_l = 0.5 * (1.0 - np.tanh(np.clip((w - mu_l) / (2.0 * _KB_EV * temp_k), -400.0, 400.0)))
    f_r = 0.5 * (1.0 - np.tanh(np.clip((w - mu_r) / (2.0 * _KB_EV * temp_k), -400.0, 400.0)))
    sig_lesser = 1j * (gamma_left * f_l + gamma_right * f_r)
    sig_greater = -1j * (gamma_left * (1.0 - f_l) + gamma_right * (1.0 - f_r))
    return np.vstack([sig_lesser, sig_greater])

# ============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

_SETUP = ("import numpy as np\n"
          "_w = (np.arange(512) - 256) * (20.0 / 512)")


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": _SETUP,
            "call": 'lead_selfenergies(_w, 2.0, 300.0, 0.050, 0.030, 0.70, 0.0)',
            "gold_call": '_oracle_lead_selfenergies(_w, 2.0, 300.0, 0.050, 0.030, 0.70, 0.0)',
            "note": 'normal, the biased junction of the task with unequal electrodes',
        },
        {
            "setup": _SETUP,
            "call": 'lead_selfenergies(_w, 0.0, 300.0, 0.050, 0.030, 0.70, 0.0)',
            "gold_call": '_oracle_lead_selfenergies(_w, 0.0, 300.0, 0.050, 0.030, 0.70, 0.0)',
            "note": 'boundary, zero bias collapses the two chemical potentials onto one another',
        },
        {
            "setup": _SETUP,
            "call": 'lead_selfenergies(_w, 2.0, -300.0, 0.050, 0.030, 0.70, 0.0)',
            "gold_call": '_oracle_lead_selfenergies(_w, 2.0, -300.0, 0.050, 0.030, 0.70, 0.0)',
            "note": 'edge, a population-inverted reservoir at negative electronic temperature',
        },
        {
            "setup": _SETUP,
            "call": 'lead_selfenergies(_w, -1.4, 150.0, 0.020, 0.065, 0.25, -0.30)',
            "gold_call": '_oracle_lead_selfenergies(_w, -1.4, 150.0, 0.020, 0.065, 0.25, -0.30)',
            "note": 'normal, reversed bias, shifted Fermi energy and the wider electrode on the right',
        },
        {
            "setup": _SETUP,
            "call": ('np.array([np.abs(lead_selfenergies(_w, 2.0, 300.0, 0.050, 0.030, 0.70, 0.0)[1]'
                     ' - lead_selfenergies(_w, 2.0, 300.0, 0.050, 0.030, 0.70, 0.0)[0]).max()])'),
            "gold_call": ('np.array([np.abs(_oracle_lead_selfenergies(_w, 2.0, 300.0, 0.050, 0.030, 0.70, 0.0)[1]'
                          ' - _oracle_lead_selfenergies(_w, 2.0, 300.0, 0.050, 0.030, 0.70, 0.0)[0]).max()])'),
            "note": 'contract, the difference of the two components must reconstruct the total width',
        },
        {
            "setup": _SETUP,
            "call": 'lead_selfenergies(_w, 3.0, 300.0, 0.040, 0.040, 1.0, 0.0)',
            "gold_call": '_oracle_lead_selfenergies(_w, 3.0, 300.0, 0.040, 0.040, 1.0, 0.0)',
            "note": 'boundary, the whole bias dropped on one side',
        },
        {
            "setup": _SETUP + '\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": ('np.array([_raises(lambda: lead_selfenergies(_w, 2.0, 300.0, -0.05, 0.03, 0.7, 0.0)),'
                     ' _raises(lambda: lead_selfenergies(_w, 2.0, 0.0, 0.05, 0.03, 0.7, 0.0)),'
                     ' _raises(lambda: lead_selfenergies(_w, 2.0, 300.0, 0.05, 0.03, 1.7, 0.0))])'),
            "gold_call": ('np.array([_raises(lambda: _oracle_lead_selfenergies(_w, 2.0, 300.0, -0.05, 0.03, 0.7, 0.0)),'
                          ' _raises(lambda: _oracle_lead_selfenergies(_w, 2.0, 0.0, 0.05, 0.03, 0.7, 0.0)),'
                          ' _raises(lambda: _oracle_lead_selfenergies(_w, 2.0, 300.0, 0.05, 0.03, 1.7, 0.0))])'),
            "note": 'contract, invalid hybridisation, zero temperature and out-of-range bias split must raise ValueError',
        },
    ]
