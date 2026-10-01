"""
Return the real part of the diagonal interband optical conductivity at each photon energy in omega, from the transition list, the linear grid size, and a broadening eta that replaces each sharp transition by a normalised Lorentzian of that half width. Return it per unit cell area, in units of the conductance quantum. The source fixes a convention here that the natural reading does not; follow the source.

An interband transition absorbs light at the energy separating the two states it connects, with a strength set by its squared velocity matrix element divided by an energy. Which energy stands in that denominator is what distinguishes the conductivity from the joint density of states, and the source states it outright.

Returns
-------
ndarray of shape (len(omega),), the real part of the conductivity in units of the conductance quantum per unit cell area.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optical_conductivity(omega: "np.ndarray", transitions: "np.ndarray", n_grid: int, eta: float) -> "np.ndarray":
    """Return the real part of the diagonal interband optical conductivity at each photon energy in omega, from the transition list, the linear grid size, and a broadening eta that replaces each sharp transition by a normalised Lorentzian of that half width. Return it per unit cell area, in units of the conductance quantum. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    ndarray of shape (len(omega),), the real part of the conductivity in units of the conductance quantum per unit cell area.

    Raises
    ------
    ValueError: if transitions is not a non-empty array of shape (N, 2), if n_grid is not positive, if the broadening is not positive, or if any transition energy is not positive.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def _oracle_optical_conductivity(omega: "np.ndarray", transitions: "np.ndarray", n_grid: int, eta: float) -> "np.ndarray":
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    t = np.asarray(transitions, dtype=float)
    if t.ndim != 2 or t.shape[1] != 2 or t.shape[0] == 0:
        raise ValueError("transitions must be a non-empty array of shape (N, 2)")
    if n_grid < 1:
        raise ValueError("n_grid must be a positive integer")
    if eta <= 0:
        raise ValueError("the broadening must be positive")
    if np.any(t[:, 0] <= 0.0):
        raise ValueError("every transition energy must be positive")
    de = t[:, 0][:, None]
    sq = t[:, 1][:, None]
    out = np.zeros(w.shape[0])
    # CONVENTION (source, eq. 13): |v|^2 divided by the transition's OWN energy, not by the
    # photon energy; the delta becomes a normalised LORENTZIAN of half width eta.
    for s in range(0, de.shape[0], 2048):
        d, q = de[s:s + 2048], sq[s:s + 2048]
        out += ((q / d) * (eta / np.pi) / ((w[None, :] - d) ** 2 + eta ** 2)).sum(0)
    return np.pi * out / (n_grid * n_grid)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nT=np.array([[1.5,0.83],[2.14,0.31],[3.77,1.92],[2.14,0.07],[5.6,0.44]])",
         "call": "optical_conductivity(np.linspace(1.0,6.0,11), T, 3, 0.1)",
         "gold_call": "_oracle_optical_conductivity(np.linspace(1.0,6.0,11), T, 3, 0.1)"},   # normal
        {"setup": "import numpy as np\nT=np.array([[1.5,0.83],[2.14,0.31],[3.77,1.92],[2.14,0.07],[5.6,0.44]])",
         "call": "optical_conductivity(np.array([2.14]), T, 1, 0.001)",
         "gold_call": "_oracle_optical_conductivity(np.array([2.14]), T, 1, 0.001)"},   # boundary
        {"setup": "import numpy as np\nT=np.array([[1.5,0.83],[2.14,0.31],[3.77,1.92],[2.14,0.07],[5.6,0.44]])",
         "call": "optical_conductivity(np.linspace(0.1,20.0,7), T, 12, 2.5)",
         "gold_call": "_oracle_optical_conductivity(np.linspace(0.1,20.0,7), T, 12, 2.5)"},   # edge
        {"setup": "import numpy as np\nT=np.array([[1.5,0.83],[2.14,0.31],[3.77,1.92],[2.14,0.07],[5.6,0.44]])\ndef _c():\n    try:\n        optical_conductivity(np.linspace(1.0,6.0,11), T, 3, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_optical_conductivity(np.linspace(1.0,6.0,11), T, 3, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
