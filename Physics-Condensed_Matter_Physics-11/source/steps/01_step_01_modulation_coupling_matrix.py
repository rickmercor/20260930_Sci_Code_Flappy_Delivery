"""
The modulated segment carries an elastic modulus that travels as a wave, E(x,t) = E0 [1 + alpha_m cos(omega_m t - kappa_m x)]. Because that law is periodic in space and in time together, it is represented exactly by a Fourier series in the single phase omega_m t - kappa_m x, and because the law is one cosine the series closes at order one: the coefficient at order zero is E0, the coefficients at orders plus and minus one are E0 alpha_m / 2, and every higher coefficient vanishes. The truncation order P of the modulus expansion is therefore not a numerical choice but a property of the modulation law, and the value P = 1 is exact rather than approximate.

What the later stages need is not the coefficient list but the operator it induces. A field inside the segment is expanded in harmonics indexed by n, and multiplying that field by the modulus mixes neighbouring harmonics: the product at harmonic q draws on the field at harmonic q - p weighted by the coefficient at order p. Collecting those weights into a matrix whose entry in row q and column n is the coefficient at order q - n turns the multiplication into a matrix product. The matrix is banded with half-bandwidth P, symmetric because the cosine makes the coefficients at plus and minus p equal, and it reduces to E0 times the identity when the modulation depth is zero, which is the check that the convention has the right orientation.

The band structure is the reason the later matching cannot use every retained harmonic. A product that spreads harmonic content by P in each direction cannot be balanced at the outermost P orders of a window truncated at n_order, and that shortfall propagates into the number of scattering orders the interfaces can carry. This step fixes the bandwidth that causes it.

Returns
-------
dict holding a float64 coefficients of shape (3,), the Fourier coefficients of the modulus at orders minus one, zero and plus one in that order; and a float64 coupling of shape (2 n_order + 1, 2 n_order + 1) whose entry in row q and column n is the coefficient at order q - n, zero where the order exceeds one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def modulation_coupling_matrix(alpha_m: float, E0: float, n_order: int) -> dict:
    """Build the Fourier coefficients of the travelling modulus and the harmonic coupling operator they induce.

    Parameters
    ----------
    alpha_m : float
        Normalised modulation depth, finite and of magnitude below one.
    E0 : float
        Elastic modulus of the unmodulated rod in pascal, above zero.
    n_order : int
        Truncation order of the harmonic expansion, one or more.

    Returns
    -------
    dict
        Under the keys coefficients and coupling. The coefficients entry is a float64
        array of shape (3,) holding the modulus Fourier coefficients at orders minus
        one, zero and plus one. The coupling entry is a float64 array of shape
        (2 n_order + 1, 2 n_order + 1) whose entry in row q and column n is the
        modulus coefficient at order q - n.

    Raises
    ------
    ValueError
        If alpha_m is not finite or is not of magnitude below one, if E0 is not
        finite or not above zero, or if n_order is below one.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_modulation_coupling_matrix(alpha_m: float, E0: float, n_order: int) -> dict:
    alpha_m = float(alpha_m)
    E0 = float(E0)
    if not np.isfinite(alpha_m) or abs(alpha_m) >= 1.0:
        raise ValueError("alpha_m must be finite and of magnitude below one")
    if not np.isfinite(E0) or E0 <= 0.0:
        raise ValueError("E0 must be finite and above zero")
    if int(n_order) != n_order or int(n_order) < 1:
        raise ValueError("n_order must be an integer of one or more")
    n_order = int(n_order)

    coefficients = np.array([E0 * alpha_m / 2.0, E0, E0 * alpha_m / 2.0], dtype=np.float64)
    orders = np.arange(-n_order, n_order + 1)
    offset = orders[:, None] - orders[None, :]
    coupling = np.zeros((2 * n_order + 1, 2 * n_order + 1), dtype=np.float64)
    for p_index, p in enumerate((-1, 0, 1)):
        coupling[offset == p] = coefficients[p_index]
    return {"coefficients": coefficients, "coupling": coupling}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ndef pack(d):\n    coefficients = np.asarray(d['coefficients'])\n    coupling = np.asarray(d['coupling'])\n    shape = np.asarray(coefficients.shape + coupling.shape, dtype=float)\n    return np.concatenate((shape, coefficients.real.ravel(), coefficients.imag.ravel(), coupling.real.ravel(), coupling.imag.ravel()))\n",
            "call": "pack(modulation_coupling_matrix(0.3, 1.0, 8))",
            "gold_call": "pack(_oracle_modulation_coupling_matrix(0.3, 1.0, 8))",
        },
        {
            "setup": "import numpy as np\ndef pack(d):\n    coefficients = np.asarray(d['coefficients'])\n    coupling = np.asarray(d['coupling'])\n    shape = np.asarray(coefficients.shape + coupling.shape, dtype=float)\n    return np.concatenate((shape, coefficients.real.ravel(), coefficients.imag.ravel(), coupling.real.ravel(), coupling.imag.ravel()))\n",
            "call": "pack(modulation_coupling_matrix(0.0, 2.5, 1))",
            "gold_call": "pack(_oracle_modulation_coupling_matrix(0.0, 2.5, 1))",
        },
        {
            "setup": "import numpy as np\ndef pack(d):\n    coefficients = np.asarray(d['coefficients'])\n    coupling = np.asarray(d['coupling'])\n    shape = np.asarray(coefficients.shape + coupling.shape, dtype=float)\n    return np.concatenate((shape, coefficients.real.ravel(), coefficients.imag.ravel(), coupling.real.ravel(), coupling.imag.ravel()))\n",
            "call": "pack(modulation_coupling_matrix(-0.6, 3.0, 4))",
            "gold_call": "pack(_oracle_modulation_coupling_matrix(-0.6, 3.0, 4))",
        },
        {
            "setup": "def verdict(fn, a=0.3, e=1.0, n=4):\n    try:\n        fn(a, e, n)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "(verdict(modulation_coupling_matrix, a=1.0), verdict(modulation_coupling_matrix, a=float('nan')), verdict(modulation_coupling_matrix, e=0.0), verdict(modulation_coupling_matrix, n=0), verdict(modulation_coupling_matrix))",
            "gold_call": "(verdict(_oracle_modulation_coupling_matrix, a=1.0), verdict(_oracle_modulation_coupling_matrix, a=float('nan')), verdict(_oracle_modulation_coupling_matrix, e=0.0), verdict(_oracle_modulation_coupling_matrix, n=0), verdict(_oracle_modulation_coupling_matrix))",
        },
    ]
