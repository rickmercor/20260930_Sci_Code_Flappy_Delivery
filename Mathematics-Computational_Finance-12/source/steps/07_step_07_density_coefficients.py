"""
Compute the damped density coefficients at resolution level m for every integer translation index from lam_lo to lam_hi inclusive, from the packed transform values produced earlier.

The construction mirrors the payoff coefficients of the previous step: the same symmetric finite frequency interval determined by the resolution level, the same uniform composite trapezoidal rule with NQ counted the same way, and the same complex exponential in the translation index.

The transform values supplied must have been evaluated on the non-negative half of exactly that grid, in increasing order, so their number is one more than half of NQ minus one. The values on the negative half are not supplied and must be reconstructed: the damped density is a real function of the log-price, so its transform at a frequency and at the negative of that frequency are complex conjugates of one another. Reconstruct the negative half by that conjugation rather than by re-evaluating the transform.

Return the coefficients in increasing order of translation index.

Raises ValueError if transform_packed does not have exactly two rows; if NQ is not an odd integer of at least 3; if the number of columns of transform_packed is not one more than half of NQ minus one; or if lam_lo > lam_hi.

The density coefficients and the payoff coefficients of a wavelet pricing method are structurally the same object: a bounded frequency integral of a transform against a complex exponential in the translation index. What differs is where the transform comes from. The payoff transform is elementary, so its coefficients cost one quadrature. The density transform is the characteristic function of the model, which for anything beyond the simplest dynamics is expensive, and for an affine model with feedback requires solving a differential system at every frequency node.

That asymmetry shapes how the computation is organised. The expensive family should be evaluated once on a fixed grid and then reused, rather than recomputed whenever a coefficient is needed, and any structure that halves the number of transform evaluations is worth exploiting. The relevant structure here is conjugate symmetry. A damped density is a real function of the log-price, and the Fourier transform of any real function takes conjugate values at a frequency and its negative. The transform therefore only has to be computed on the non-negative half of the grid, with the negative half obtained by conjugation at no further cost.

Two conventions have to be fixed for this to work. The first is the sign convention relating the transform argument to the damping level, since the damped density and the damped payoff are displaced off the real axis in opposite directions and interchanging them silently produces a different and incorrect price. The second is the node convention for the quadrature, since a symmetric grid with a node at the origin requires an odd total node count, and reading that count as the number of intervals rather than the number of nodes shifts every node by half a spacing. Neither convention is detectable from the shape of the output, so both have to be treated as part of the interface rather than as implementation choices.

Returns
-------
np.ndarray of shape (lam_hi - lam_lo + 1,), the damped density coefficients in increasing translation index as native float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def density_coefficients(transform_packed: 'np.ndarray', alpha: float, m: int,
                         lam_lo: int, lam_hi: int, NQ: int) -> 'np.ndarray':
    '''Damped density coefficients from packed transform values.

    Parameters
    ----------
    transform_packed : np.ndarray
        Shape (2, (NQ + 1) // 2): real and imaginary parts of the transform on
        the non-negative half of the frequency grid, in increasing order.
    alpha : float
        Damping level used to produce transform_packed.
    m : int
        Resolution level.
    lam_lo, lam_hi : int
        Inclusive bounds of the translation range, lam_lo <= lam_hi.
    NQ : int
        Total trapezoidal nodes across the symmetric frequency interval,
        endpoints included; odd and at least 3.

    Returns
    -------
    result : np.ndarray
        Shape (lam_hi - lam_lo + 1,), in increasing translation index.

    Raises
    ------
    ValueError
        If transform_packed does not have exactly two rows; if NQ is not an
        odd integer of at least 3; if the number of columns of
        transform_packed is not one more than half of NQ minus one; or if
        lam_lo > lam_hi.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _h_frequency_grid(m, NQ):
    """Symmetric frequency grid and trapezoidal weights at resolution m."""
    half_width = 2.0 ** m * np.pi
    nodes = np.linspace(-half_width, half_width, int(NQ))
    step = nodes[1] - nodes[0]
    weights = np.full(int(NQ), step)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return nodes, weights


def _h_mirror(packed):
    """Rebuild the full-line transform from its non-negative half."""
    half = packed[0] + 1j * packed[1]
    return np.concatenate([np.conjugate(half[:0:-1]), half])


def _oracle_density_coefficients(transform_packed: 'np.ndarray', alpha: float,
                                 m: int, lam_lo: int, lam_hi: int,
                                 NQ: int) -> 'np.ndarray':
    """Reference implementation."""
    packed = np.asarray(transform_packed, dtype=float)
    if packed.ndim != 2 or packed.shape[0] != 2:
        raise ValueError("transform_packed must have exactly two rows")
    if int(NQ) < 3 or int(NQ) % 2 == 0:
        raise ValueError("NQ must be an odd integer of at least 3")
    if packed.shape[1] != (int(NQ) + 1) // 2:
        raise ValueError(
            "transform_packed column count inconsistent with NQ")
    if int(lam_lo) > int(lam_hi):
        raise ValueError("lam_lo must not exceed lam_hi")

    nodes, weights = _h_frequency_grid(int(m), int(NQ))
    transform = _h_mirror(packed)
    indices = np.arange(int(lam_lo), int(lam_hi) + 1)
    phase = np.exp(1j * np.outer(indices / 2.0 ** int(m), nodes))
    scale = 2.0 ** (-int(m) / 2.0) / (2.0 * np.pi)
    return scale * np.real(phase @ (weights * transform))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _setup = """import numpy as np
def _synthetic(nq, m, alpha, var, shift):
    half = np.linspace(0.0, 2.0 ** m * np.pi, (nq + 1) // 2)
    z = half + 1j * alpha
    val = np.exp(-0.5 * var * z ** 2 + 1j * z * shift)
    return np.vstack([val.real, val.imag])
"""
    return [
        # --- normal: the locked grid and translation range ---
        {
            "setup": _setup + "tp = _synthetic(513, 3, 4.3317046485, 0.04, 4.5)\n",
            "call": "density_coefficients(tp, 4.3317046485, 3, 20, 101, 513)",
            "gold_call": ("_oracle_density_coefficients(tp, 4.3317046485, 3, "
                          "20, 101, 513)"),
        },
        # --- boundary: coarse resolution, small node count, range straddling
        #     the origin ---
        {
            "setup": _setup + "tp = _synthetic(33, 1, 1.5, 0.25, 0.0)\n",
            "call": "density_coefficients(tp, 1.5, 1, -4, 4, 33)",
            "gold_call": "_oracle_density_coefficients(tp, 1.5, 1, -4, 4, 33)",
        },
        # --- boundary: the smallest admissible node count ---
        {
            "setup": _setup + "tp = _synthetic(3, 2, 2.0, 0.1, 1.0)\n",
            "call": "density_coefficients(tp, 2.0, 2, 0, 3, 3)",
            "gold_call": "_oracle_density_coefficients(tp, 2.0, 2, 0, 3, 3)",
        },
        # --- edge: a single translation index at fine resolution ---
        {
            "setup": _setup + "tp = _synthetic(129, 5, 2.0, 0.02, 4.55)\n",
            "call": "density_coefficients(tp, 2.0, 5, 60, 60, 129)",
            "gold_call": ("_oracle_density_coefficients(tp, 2.0, 5, 60, 60, "
                          "129)"),
        },
        # --- structural probe: an undamped, unshifted transform gives
        #     coefficients symmetric about the origin and peaking there;
        #     exact integers ---
        {
            "setup": _setup + """
def probe():
    tp = _synthetic(65, 2, 0.0, 0.30, 0.0)
    v = density_coefficients(tp, 0.0, 2, -6, 6, 65)
    sym = int(round(1e9 * float(np.max(np.abs(v - v[::-1])))))
    return [int(len(v)), sym, int(v[6] > 0.0),
            int(int(np.argmax(np.abs(v))) == 6)]
def probe_gold():
    tp = _synthetic(65, 2, 0.0, 0.30, 0.0)
    v = _oracle_density_coefficients(tp, 0.0, 2, -6, 6, 65)
    sym = int(round(1e9 * float(np.max(np.abs(v - v[::-1])))))
    return [int(len(v)), sym, int(v[6] > 0.0),
            int(int(np.argmax(np.abs(v))) == 6)]
""",
            "call": "probe()",
            "gold_call": "probe_gold()",
        },
        # --- invalid: transform supplied with a single row ---
        {
            "setup": _setup + """
tp = _synthetic(33, 1, 1.5, 0.25, 0.0)
def run_model():
    try:
        density_coefficients(tp[0:1], 1.5, 1, -4, 4, 33)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_density_coefficients(tp[0:1], 1.5, 1, -4, 4, 33)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: column count inconsistent with the node count ---
        {
            "setup": _setup + """
tp = _synthetic(33, 1, 1.5, 0.25, 0.0)
def run_model():
    try:
        density_coefficients(tp, 1.5, 1, -4, 4, 65)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_density_coefficients(tp, 1.5, 1, -4, 4, 65)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: even node count ---
        {
            "setup": _setup + """
tp = _synthetic(33, 1, 1.5, 0.25, 0.0)
def run_model():
    try:
        density_coefficients(tp, 1.5, 1, -4, 4, 32)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_density_coefficients(tp, 1.5, 1, -4, 4, 32)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: inverted translation range ---
        {
            "setup": _setup + """
tp = _synthetic(33, 1, 1.5, 0.25, 0.0)
def run_model():
    try:
        density_coefficients(tp, 1.5, 1, 10, 4, 33)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_density_coefficients(tp, 1.5, 1, 10, 4, 33)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
