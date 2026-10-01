"""
Given the two matrices of the pencil and an interval on the positive imaginary frequency axis, sample the interval uniformly and return the smallest singular value of the shifted pencil at each sample, the imaginary parts at which interior local minima occur in increasing order, and the imaginary part of the deepest such minimum. The pencil is shifted by the eigenvalue equal to the imaginary unit times the frequency. When no interior local minimum exists the list is empty and the reported location is not a number.

An iterative eigenvalue solver has to be told roughly where to look, and for this pencil a dense generalised eigensolver is a poor way to find out. The second member of the pencil is singular, so the pencil carries an infinite eigenvalue and cannot be inverted into a standard problem, and the compactified equation has an irregular singular point at infinity whose second solution varies like a simple exponential of the reciprocal coordinate and is therefore invisible to a polynomial basis. A dense solve consequently returns a spectrum in which the physical modes are hidden among artefacts whose positions move with the resolution, and sorting that list by imaginary part finds nothing useful.




For the mode of interest the search is one-dimensional, because the mode sits on the imaginary frequency axis. That the axis is the right place to look is itself a result rather than an assumption: the scan over spin label and dimension that preceded this analysis found purely imaginary total transmission modes for gravitational vector perturbations and for no other sector, and for these there is exactly one such mode in each background. Sweeping a purely imaginary frequency along the axis and recording the smallest singular value of the shifted pencil therefore isolates it: the shifted matrix is singular exactly at an eigenvalue, so the smallest singular value dips at the mode and the dip is the only interior local minimum over a wide interval.




Two cautions. The global minimum over the sampled points is not the discriminator and must not be used on its own: on an interval that contains the mode it is the mode itself, since the smallest singular value rises away from the mode towards both ends, while on an interval that contains no mode it sits at an endpoint instead. What identifies the mode is the interior local minimum, meaning a sample smaller than both of its neighbours, and the absence of one is what reports that the interval carries no mode. And the sweep should be run at a coarse resolution, both because it costs a singular-value decomposition at every sample and because the location only has to be good to a digit or two for the refinement that follows to converge cubically. Resolving the mode is the job of the next step, not of this one.

Returns
-------
dict holding the array frequencies, the imaginary parts sampled; the array smallest_singular_values, one per sample; the array minima, the imaginary parts of the interior local minima in increasing order; and the float location, the imaginary part of the deepest interior local minimum, or not a number when there is none.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def imaginary_axis_scan(
    operator_a: np.ndarray,
    operator_b: np.ndarray,
    lower: float,
    upper: float,
    samples: int,
) -> dict:
    """Locate purely imaginary modes of the pencil by sweeping the smallest singular value.

    Parameters
    ----------
    operator_a : np.ndarray
        Second-order member of the pencil.
    operator_b : np.ndarray
        First-order member of the pencil.
    lower : float
        Lower end of the sweep in the imaginary part of the frequency.
    upper : float
        Upper end of the sweep.
    samples : int
        Number of uniformly spaced samples.

    Returns
    -------
    dict
        Under the keys frequencies, smallest_singular_values, minima and location.

    Raises
    ------
    ValueError
        When the matrices are not square, finite and of the same shape with side at least five, when
        the interval is not a finite strictly positive increasing pair, or when the sample count is
        not an integer of at least five.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validate_scan(operator_a, operator_b, lower, upper, samples):
    a = np.asarray(operator_a, dtype=complex)
    b = np.asarray(operator_b, dtype=complex)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 5:
        raise ValueError("operator_a must be a square array of side at least five")
    if b.shape != a.shape:
        raise ValueError("operator_b must have the same shape as operator_a")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("the pencil matrices must be finite")
    low, high = float(lower), float(upper)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high <= low:
        raise ValueError("the sweep interval must be finite, strictly positive and increasing")
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)):
        raise ValueError("samples must be an integer")
    if int(samples) < 5:
        raise ValueError("samples must be at least five")
    return a, b, low, high, int(samples)


def _oracle_imaginary_axis_scan(
    operator_a: np.ndarray,
    operator_b: np.ndarray,
    lower: float,
    upper: float,
    samples: int,
) -> dict:
    """Reference implementation."""
    a, b, low, high, count = _validate_scan(operator_a, operator_b, lower, upper, samples)
    frequencies = np.linspace(low, high, count)
    values = np.empty(count)
    for index, height in enumerate(frequencies):
        shifted = a - (1j * (1j * height)) * b
        values[index] = np.linalg.svd(shifted, compute_uv=False)[-1]
    interior = [
        i for i in range(1, count - 1)
        if values[i] < values[i - 1] and values[i] < values[i + 1]
    ]
    minima = frequencies[interior] if interior else np.empty(0)
    if interior:
        location = float(frequencies[min(interior, key=lambda i: values[i])])
    else:
        location = float("nan")
    return {
        "frequencies": frequencies,
        "smallest_singular_values": values,
        "minima": minima,
        "location": location,
    }

# =============================================================================
# TEST CASES
# =============================================================================

FLAT = """
def flat(x):
    if isinstance(x, dict):
        return flat([x[k] for k in sorted(x)])
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""

SETUP = """
import numpy as np
def tangherlini_pencil(n, d, ell, s):
    # Chebyshev-Lobatto collocation of the total transmission pencil, written out in closed form
    # so that no other step's function is needed to build this case's input
    k = np.arange(n + 1)
    x = np.cos(np.pi * k / n)
    c = np.where((k == 0) | (k == n), 2.0, 1.0) * (-1.0) ** k
    diff = x[:, None] - x[None, :]
    base = np.outer(c, 1.0 / c) / (diff + np.eye(n + 1))
    base = base - np.diag(base.sum(axis=1))
    nodes = (1.0 - x) / 2.0
    dmat = -2.0 * base
    m = d - 3
    p = nodes ** 2 - nodes ** (m + 2)
    dp = 2.0 * nodes - (m + 2) * nodes ** (m + 1)
    q = ell * (ell + d - 3) + (d - 2) * (d - 4) / 4.0 + (1.0 - s ** 2) * (d - 2) ** 2 * nodes ** m / 4.0
    a = p[:, None] * (dmat @ dmat) + dp[:, None] * dmat - np.diag(q)
    return {"operator_a": a, "operator_b": 2.0 * dmat, "nodes": nodes}
P14 = tangherlini_pencil(40, 14, 2, 2.0)
P20 = tangherlini_pencil(40, 20, 2, 2.0)
def digest(out):
    return (np.round(out["frequencies"], 12), np.round(out["smallest_singular_values"], 10),
            np.round(out["minima"], 12), round(float(out["location"]), 12))
"""


def test_cases():
    return [
        {
            # the graded sector: one interior minimum, near an imaginary part of eleven tenths
            "setup": SETUP + FLAT,
            "call": "flat(digest(imaginary_axis_scan(P14['operator_a'], P14['operator_b'], 0.3, 4.0, 75)))",
            "gold_call": "flat(digest(_oracle_imaginary_axis_scan(P14['operator_a'], P14['operator_b'], 0.3, 4.0, 75)))",
        },
        {
            # the sweep must find exactly one interior minimum in the vector sector, near an imaginary
            # part of eleven tenths; on an interval containing the mode the global minimum over the
            # samples is the mode itself; the twenty-dimensional background carries its mode lower down
            "setup": SETUP + """
def structure(fn):
    vec = fn(P14["operator_a"], P14["operator_b"], 0.3, 4.0, 75)
    big = fn(P20["operator_a"], P20["operator_b"], 0.3, 4.0, 75)
    return (int(vec["minima"].size == 1),
            int(abs(vec["location"] - 1.1) < 0.06),
            int(big["minima"].size == 1),
            int(big["location"] < vec["location"]),
            int(vec["frequencies"][int(np.argmin(vec["smallest_singular_values"]))] == vec["location"]))
""" + FLAT,
            "call": "flat(structure(imaginary_axis_scan))",
            "gold_call": "flat(structure(_oracle_imaginary_axis_scan))",
        },
        {
            # edge: a narrow interval containing no mode, which must report an empty list and a
            # location that is not a number, although the global minimum over the samples still
            # exists and sits at an endpoint, which is why it cannot be used to locate a mode
            "setup": SETUP + """
def empty(fn):
    out = fn(P14["operator_a"], P14["operator_b"], 2.5, 3.5, 11)
    return (int(out["minima"].size == 0), int(out["location"] != out["location"]),
            int(out["frequencies"].size == 11),
            int(int(np.argmin(out["smallest_singular_values"])) in (0, 10)))
""" + FLAT,
            "call": "flat(empty(imaginary_axis_scan))",
            "gold_call": "flat(empty(_oracle_imaginary_axis_scan))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(operator_a=P14["operator_a"], operator_b=P14["operator_b"],
                lower=0.3, upper=4.0, samples=75)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(imaginary_axis_scan, lower=0.0), "
                    "verdict(imaginary_axis_scan, lower=4.0, upper=0.3), "
                    "verdict(imaginary_axis_scan, samples=3), "
                    "verdict(imaginary_axis_scan, samples=75.0), "
                    "verdict(imaginary_axis_scan, operator_a=np.ones((3, 3)))))",
            "gold_call": "flat((verdict(_oracle_imaginary_axis_scan, lower=0.0), "
                         "verdict(_oracle_imaginary_axis_scan, lower=4.0, upper=0.3), "
                         "verdict(_oracle_imaginary_axis_scan, samples=3), "
                         "verdict(_oracle_imaginary_axis_scan, samples=75.0), "
                         "verdict(_oracle_imaginary_axis_scan, operator_a=np.ones((3, 3)))))",
        },
    ]
