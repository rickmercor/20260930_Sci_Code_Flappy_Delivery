"""
Census size of a population at given mutant frequencies, and its integral over the frequency, for five forms of the size response.

When a mutation changes how many individuals the environment supports, the census

size of the population does not stay constant while the mutation spreads: it moves

from the size of the ancestral population towards the size that a population fixed

for the mutant would have, along a path set by how the two types share and modify

their environment. Frequency-weighted means of the two sizes give one family of

paths. The source also uses a symmetric S-shaped path of its own, which changes

slowly while either type is rare. The strength of genetic drift at each frequency,

and every fixation quantity later in the pipeline, is built from this census size

and from its accumulated integral over the mutant frequency.

Returns
-------
np.ndarray of shape (n, 2), the census size N(p) and its integral from 0 to p at each of the n mutant frequencies
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def size_response(frequency: "np.ndarray", ancestral_size: float, mutant_size: float, response: str) -> "np.ndarray":
    '''Census size of a population at given mutant frequencies, and its integral over the frequency, for five forms of the size response.

    A population has census size Na = ancestral_size when the mutant is absent
    and Nm = mutant_size when the mutant is fixed. At mutant frequency p its
    census size N(p) follows `response`:
    "arithmetic", (1 - p) Na + p Nm;
    "geometric", Na^(1 - p) Nm^p;
    "harmonic", ((1 - p)/Na + p/Nm)^(-1);
    "root_mean_square", ((1 - p) Na^2 + p Nm^2)^(1/2);
    "s_shaped", the symmetric S-shaped response of the source,
    Na + (Nm - Na) p^2 / (p^2 + (1 - p)^2), which moves from Na at p = 0 to
    Nm at p = 1 with zero slope at both ends and satisfies
    N(p) + N(1 - p) = Na + Nm.
    When Na equals Nm every response is the constant Na.

    Parameters
    ----------
    frequency : np.ndarray
        Shape (n,), n >= 1; mutant frequencies, each a finite number in [0, 1].
    ancestral_size : float
        Census size Na with the mutant absent, a finite number from 1 to 1e7.
    mutant_size : float
        Census size Nm with the mutant fixed, a finite number from 1 to 1e7.
    response : str
        One of "arithmetic", "geometric", "harmonic", "root_mean_square" and
        "s_shaped".

    Returns
    -------
    size_table : np.ndarray
        Shape (n, 2). size_table[j, 0] is the census size N at frequency[j];
        size_table[j, 1] is the integral of N(p) dp from p = 0 to
        p = frequency[j]. Every entry is accurate to a relative error below
        1e-12.

    Raises
    ------
    ValueError
        If frequency is not a one-dimensional array of at least one finite
        number in [0, 1], if ancestral_size or mutant_size is not a finite
        number from 1 to 1e7, or if response is not one of the five names.
    '''
    return size_table  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import special


def _response_names() -> tuple:
    """The five size responses, in the order used throughout the pipeline."""
    return ("arithmetic", "geometric", "harmonic", "root_mean_square", "s_shaped")


def _check_size(name: str, value: float, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number from {minimum:g} to 1e7")
    v = float(value)
    if not np.isfinite(v) or v < minimum or v > 1e7:
        raise ValueError(f"{name} must be a finite number from {minimum:g} to 1e7")
    return v


def _check_response(response: str) -> str:
    if not isinstance(response, str) or response not in _response_names():
        raise ValueError("response must be one of " + ", ".join(_response_names()))
    return response


def _size_table(p: "np.ndarray", na: float, nm: float, kind: str) -> "np.ndarray":
    """Unchecked census size and its integral from 0, for a float array p in [0, 1]."""
    if kind == "arithmetic":
        size = (1.0 - p) * na + p * nm
        integral = p * (na + 0.5 * (nm - na) * p)
    elif kind == "geometric":
        rate = np.log(nm / na)
        size = na * np.exp(p * rate)
        integral = na * p * special.exprel(p * rate)             # Na (e^{pL} - 1)/L without cancellation
    elif kind == "harmonic":
        x = (na / nm - 1.0) * p
        size = na / (1.0 + x)
        ratio = np.ones_like(x)
        large = np.abs(x) > 1e-8
        ratio[large] = np.log1p(x[large]) / x[large]
        ratio[~large] = 1.0 - x[~large] / 2.0 + x[~large] ** 2 / 3.0
        integral = na * p * ratio                                # log(1 + x)/(1/Nm - 1/Na)
    elif kind == "root_mean_square":
        a = (1.0 - p) * na * na + p * nm * nm
        b = na * na
        size = np.sqrt(a)
        integral = (2.0 / 3.0) * p * (a + np.sqrt(a * b) + b) / (np.sqrt(a) + np.sqrt(b))
    else:                                                        # the source's S-shaped response
        share = p * p / (p * p + (1.0 - p) ** 2)
        size = na + (nm - na) * share
        integral = na * p + (nm - na) * (0.5 * p + 0.25 * np.log1p(2.0 * p * (p - 1.0)))
    return np.column_stack([size, integral])


def _oracle_size_response(frequency: "np.ndarray", ancestral_size: float, mutant_size: float, response: str) -> "np.ndarray":
    try:
        p = np.asarray(frequency, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("frequency must be a one-dimensional array of at least one finite number in [0, 1]")
    if p.ndim != 1 or p.size < 1 or not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("frequency must be a one-dimensional array of at least one finite number in [0, 1]")
    na = _check_size("ancestral_size", ancestral_size, 1.0)
    nm = _check_size("mutant_size", mutant_size, 1.0)
    kind = _check_response(response)
    return _size_table(p, na, nm, kind)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the S-shaped response of a sixteen-fold boost in census size ---
        {
            "setup": "import numpy as np\nfrequencies = np.array([0.05, 0.3, 0.5, 0.9])\n",
            "call": "size_response(frequencies, 800.0, 12800.0, 's_shaped')",
            "gold_call": "_oracle_size_response(frequencies, 800.0, 12800.0, 's_shaped')",
        },
        # --- boundary: the harmonic response of a four-fold drop, at both ends and halfway ---
        {
            "setup": "import numpy as np\nfrequencies = np.array([0.0, 0.5, 1.0])\n",
            "call": "size_response(frequencies, 800.0, 200.0, 'harmonic')",
            "gold_call": "_oracle_size_response(frequencies, 800.0, 200.0, 'harmonic')",
        },
        # --- edge: the S-shaped response of a steep drop, below and above halfway ---
        {
            "setup": "import numpy as np\nfrequencies = np.array([0.2, 0.75])\n",
            "call": "size_response(frequencies, 2000.0, 150.0, 's_shaped')",
            "gold_call": "_oracle_size_response(frequencies, 2000.0, 150.0, 's_shaped')",
        },
    ]
