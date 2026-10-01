"""
Return the orbital entanglement measure built from four retained configuration weights and the magnitude of the surviving coherence between them. Invalid input raises ValueError: weights must hold four finite nonnegative numbers and z_magnitude must be a finite nonnegative scalar.

For a structured state of two effective two-level systems the entanglement of formation is a monotonic function of a quantity assembled from populations and coherences. Here the state carries coherence between the two configurations holding one excitation each, while coherence between the two configurations holding both excitations on one side is absent once the state is restricted to measurements respecting local particle number. The surviving coherence has subtracted from it the geometric mean of the two populations that the absent coherence would have connected, namely the two double-occupancy weights. Multiplication by two and clipping from below at zero classify a state whose coherence is too small relative to that population product as unentangled. Three choices are load bearing and none is interchangeable: the geometric mean cannot be replaced by the arithmetic mean, the coherence must be paired with the double-occupancy weights rather than the single-occupancy ones, and the populations remain unnormalized values evaluated in the full state.

Returns
-------
float, the orbital entanglement measure from the spin-exchange branch
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def orbital_concurrence(weights, z_magnitude):
    """Compute the spin-exchange branch of the orbital entanglement measure.

    Parameters
    ----------
    weights : array_like
        Four unnormalized populations ordered as ``[u_plus, w_1, w_2,
        u_minus]``.
    z_magnitude : float
        Magnitude of the spin-exchange coherence.

    Returns
    -------
    value : float
        Orbital entanglement measure from the spin-exchange branch.
    
    Raises
    ------
    ValueError
        If ``weights`` is not a one-dimensional sequence of four finite
        nonnegative numbers, or if ``z_magnitude`` is not a finite
        nonnegative scalar.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_orbital_concurrence(weights, z_magnitude):
    try:
        w = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must contain numeric values") from exc
    if w.ndim != 1 or w.size != 4:
        raise ValueError("weights must be a one-dimensional array with four entries")
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and nonnegative")
    try:
        valid_z = np.asarray(z_magnitude).ndim == 0 and bool(np.isfinite(z_magnitude)) and z_magnitude >= 0.0
    except (TypeError, ValueError):
        valid_z = False
    if not valid_z:
        raise ValueError("z_magnitude must be a finite nonnegative scalar")
    u_plus = w[0]
    u_minus = w[3]
    return float(2.0 * max(0.0, z_magnitude - np.sqrt(u_plus * u_minus)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'weights = [0.005286215182, 0.091865686565, 0.091865686565, 0.155711059849]\n'
               'z_magnitude = 0.062182948281\n',
      'call': 'round(float(orbital_concurrence(weights, z_magnitude)), 9)',
      'gold_call': 'round(float(_oracle_orbital_concurrence(weights, z_magnitude)), 9)'},
     {'setup': 'weights = [0.0, 0.5, 0.5, 0.0]\nz_magnitude = 0.5\n',
      'call': 'orbital_concurrence(weights, z_magnitude)',
      'gold_call': '_oracle_orbital_concurrence(weights, z_magnitude)'},
     {'setup': 'weights = [0.25, 0.25, 0.25, 0.25]\nz_magnitude = 0.1\n',
      'call': 'orbital_concurrence(weights, z_magnitude)',
      'gold_call': '_oracle_orbital_concurrence(weights, z_magnitude)'},
     {'setup': 'weights = [0.04, 0.1, 0.1, 0.09]\nz_magnitude = 0.06\n',
      'call': 'orbital_concurrence(weights, z_magnitude)',
      'gold_call': '_oracle_orbital_concurrence(weights, z_magnitude)'},
     {'setup': '\n'
               '\n'
               'import numpy as np\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(orbital_concurrence, [0.1, 0.2, 0.3], 0.5)',
      'gold_call': '_exception_code(_oracle_orbital_concurrence, [0.1, 0.2, 0.3], 0.5)'}]
