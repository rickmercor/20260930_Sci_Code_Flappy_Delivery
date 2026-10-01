"""
Lay down the ordered set of reference configurations that defines the path collective variable, taking the guide curve of the configuration as the route: n_images points on the guide curve r(t) = (t, 0.15 sin(pi t)), equally spaced in arc length along the curve (not uniformly in the parameter t, and not by equal chord length), with the first at t = 0 and the last at t = 1, every configuration placed to within 1e-9 Angstrom of its exact equal-arc-length position (an arc-length table of at least 40 000 points over the curve, or exact quadrature of the speed with a root finder, achieves this; a 10 000-point table does not). Return the configurations in the order in which they run from the initial site to the final site, one row per configuration.

A path collective variable does not act on a curve, it acts on a finite list of configurations sampled from that curve. How they are spaced along the route is not cosmetic: the soft minimum that both coordinates are built from weighs neighbouring entries against each other, so an uneven list distorts the progress coordinate and leaves the orthogonal coordinate with a floor that varies along the path. The source asks for an equidistant list; on a curved route equal arc length, equal chord length and equal parameter steps are three different lists, and this calculation fixes the first.

Returns
-------
An (n_images, 2) float64 array of reference configurations in the collective-variable plane, first column the migration coordinate and second column the transverse distortion in Angstrom.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reference_path(n_images: int) -> "np.ndarray":
    """Lay down the ordered set of reference configurations that defines the path collective variable,
    taking the guide curve of the configuration as the route: n_images points on the guide curve
    r(t) = (t, 0.15 sin(pi t)), equally spaced in arc length along the curve (not uniformly in the
    parameter t, and not by equal chord length), with the first at t = 0 and the last at t = 1,
    every configuration placed to within 1e-9 Angstrom of its exact equal-arc-length position (an
    arc-length table of at least 40 000 points over the curve, or exact quadrature of the speed with
    a root finder, achieves this; a 10 000-point table does not). Return the configurations in the
    order in which they run from the initial site to the final site, one row per configuration.

    Args:
        n_images: int, the number of reference configurations (at least 2).

    Returns:
        An (n_images, 2) float64 array of reference configurations in the collective-variable plane,
        first column the migration coordinate and second column the transverse distortion in
        Angstrom.

    Raises:
        ValueError: if n_images is smaller than 2.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _PATH_AMP():
    return 0.15

def _guide(t):
    t = np.asarray(t, dtype=float)
    return np.stack([t, _PATH_AMP()*np.sin(np.pi*t)], -1)

def _oracle_reference_path(n_images: int) -> "np.ndarray":
    n_images = int(n_images)
    if n_images < 2:
        raise ValueError("n_images must be at least 2")
    t = np.linspace(0.0, 1.0, 40001)
    pc = _guide(t)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pc, axis=0), axis=1))])
    target = np.linspace(0.0, arc[-1], n_images)
    return np.stack([np.interp(target, arc, pc[:, 0]),
                     np.interp(target, arc, pc[:, 1])], 1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nn_images = 12\n',
         'call': 'reference_path(n_images)',
         'gold_call': '_oracle_reference_path(n_images)'},
        {'setup': 'import numpy as np\nn_images = 60\n',
         'call': 'reference_path(n_images)',
         'gold_call': '_oracle_reference_path(n_images)'},
        {'setup': 'import numpy as np\nn_images = 200\n',
         'call': 'reference_path(n_images)',
         'gold_call': '_oracle_reference_path(n_images)'},
    ]
