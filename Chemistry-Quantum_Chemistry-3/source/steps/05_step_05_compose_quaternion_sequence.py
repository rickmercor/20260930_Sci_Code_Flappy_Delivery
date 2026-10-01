"""
Implement compose_quaternion_sequence which accumulates the rotation of a whole pulse sequence element from its per-interval quaternions.

A piecewise-constant irradiation element is a product of rotations, one per interval. Composing
them in the quaternion algebra keeps the result an exact rotation at every stage, avoids the
drift that repeated matrix multiplication would accumulate, and costs sixteen multiplications
per interval instead of a full propagator product.

Rotations do not commute, so the accumulation order encodes the physical time order: the
interval that acts last on the spin stands leftmost in the product, and the interval that acts
first stands rightmost. The resulting quaternion is the single rotation equivalent to the whole
element and is the object whose overlap with a target rotation is optimized.

Returns
-------
np.ndarray, real array of shape leading_shape + (4,) holding the scalar-first rotation quaternion of the whole sequence
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compose_quaternion_sequence(quaternions: "np.ndarray") -> "np.ndarray":
    '''Compose an ordered set of interval quaternions into the element quaternion.

    The interval stored at index 0 acts first and the interval stored at the
    last index acts last, so the returned quaternion is the product in which
    the last interval stands leftmost.

    Parameters
    ----------
    quaternions : np.ndarray
        Real array of shape leading_shape + (n_steps, 4) holding scalar-first
        interval quaternions in time order. Leading axes are treated as
        independent sequences. n_steps must be at least 1.

    Returns
    -------
    q_total : np.ndarray
        Real array of shape leading_shape + (4,) holding the scalar-first
        quaternion of the whole sequence.

    Raises
    ------
    ValueError
        If quaternions has fewer than two axes, if its trailing axis does not
        have length 4, if it holds no intervals, or if it contains values
        that are not finite.
    '''
    return q_total  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _quaternion_multiply(left: "np.ndarray", right: "np.ndarray") -> "np.ndarray":
    """Return the scalar-first quaternion product left * right, broadcasting over leading axes."""
    a2, b2, c2, d2 = (left[..., 0], left[..., 1], left[..., 2], left[..., 3])
    a1, b1, c1, d1 = (right[..., 0], right[..., 1], right[..., 2], right[..., 3])
    return np.stack([a2 * a1 - b2 * b1 - c2 * c1 - d2 * d1,
                     a2 * b1 + b2 * a1 + c2 * d1 - d2 * c1,
                     a2 * c1 - b2 * d1 + c2 * a1 + d2 * b1,
                     a2 * d1 + b2 * c1 - c2 * b1 + d2 * a1], axis=-1)


def _oracle_compose_quaternion_sequence(quaternions: "np.ndarray") -> "np.ndarray":
    quaternions = np.asarray(quaternions, dtype=float)
    if quaternions.ndim < 2:
        raise ValueError("quaternions must have at least two axes")
    if quaternions.shape[-1] != 4:
        raise ValueError("quaternions must have a trailing axis of length 4")
    if quaternions.shape[-2] < 1:
        raise ValueError("quaternions must hold at least one interval")
    if not np.all(np.isfinite(quaternions)):
        raise ValueError("quaternions must be finite")

    # Copy so that a single-interval sequence does not alias the caller's array.
    q_total = quaternions[..., 0, :].copy()
    for index in range(1, quaternions.shape[-2]):
        q_total = _quaternion_multiply(quaternions[..., index, :], q_total)
    return q_total

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    common = "import numpy as np\n"
    element = (common
               + "amps = 2.0 * np.pi * np.array([99882.6002, 97593.2747, 99938.7784, 99788.5816, 98277.6433, 99577.6807])\n"
               + "phs = np.deg2rad(np.array([308.58, 91.90, 322.74, 235.95, 158.68, 107.58]))\n"
               + "off = 2.0 * np.pi * np.array([20.1e3, 43.5e3, 64.25e3, 77.21e3, 78.62e3, 67.35e3])\n"
               + "qs = _oracle_build_interval_quaternion(amps, phs, off, 2.0e-6)\n")
    return [
        # Normal: six consecutive intervals of the graded pulse element.
        {
            "setup": element,
            "call": "[round(float(v), 10) for v in compose_quaternion_sequence(qs)]",
            "gold_call": "[round(float(v), 10) for v in _oracle_compose_quaternion_sequence(qs)]",
        },
        # Boundary: a single interval composes to itself.
        {
            "setup": element,
            "call": "[round(float(v), 12) for v in compose_quaternion_sequence(qs[:1])]",
            "gold_call": "[round(float(v), 12) for v in _oracle_compose_quaternion_sequence(qs[:1])]",
        },
        # Boundary: identity intervals leave the accumulated rotation unchanged.
        {
            "setup": common + "ident = np.zeros((5, 4)); ident[:, 0] = 1.0\n",
            "call": "[round(float(v), 12) for v in compose_quaternion_sequence(ident)]",
            "gold_call": "[round(float(v), 12) for v in _oracle_compose_quaternion_sequence(ident)]",
        },
        # Edge: two non-commuting quarter turns about perpendicular axes, which
        # distinguishes the physical time order from its reverse.
        {
            "setup": common + "s = np.sqrt(0.5)\npair = np.array([[s, -s, 0.0, 0.0], [s, 0.0, -s, 0.0]])\n",
            "call": "[round(float(v), 12) for v in compose_quaternion_sequence(pair)]",
            "gold_call": "[round(float(v), 12) for v in _oracle_compose_quaternion_sequence(pair)]",
        },
        # Edge: composing a rotation with its inverse returns the identity.
        {
            "setup": common + "q = _oracle_build_interval_quaternion(2.0 * np.pi * 7.5e4, 1.1, 2.0 * np.pi * 3.0e4, 4.0e-6)\ninv = q * np.array([1.0, -1.0, -1.0, -1.0])\npairinv = np.stack([q, inv])\n",
            "call": "[round(float(v), 12) for v in compose_quaternion_sequence(pairinv)]",
            "gold_call": "[round(float(v), 12) for v in _oracle_compose_quaternion_sequence(pairinv)]",
        },
        # Edge: the composed rotation stays a unit quaternion.
        {
            "setup": element,
            "call": "round(float(np.linalg.norm(compose_quaternion_sequence(qs))), 12)",
            "gold_call": "round(float(np.linalg.norm(_oracle_compose_quaternion_sequence(qs))), 12)",
        },
        # Normal: several crystallites composed at once, which pins the batched
        # behaviour used inside the powder average.
        {
            "setup": common + "amps = 2.0 * np.pi * np.array([9.9e4, 9.5e4, 8.0e4, 7.0e4])\nphs = np.array([0.2, 1.4, 3.0, 4.7])\noff = 2.0 * np.pi * np.array([[1.0e4, 5.0e4, -3.0e4, 0.0], [-8.0e4, 2.0e4, 6.0e4, 4.0e4], [0.0, 0.0, 1.0e5, -1.0e5]])\nbatch = _oracle_build_interval_quaternion(amps[None, :], phs[None, :], off, 2.0e-6)\n",
            "call": "[round(float(v), 10) for v in compose_quaternion_sequence(batch).ravel()]",
            "gold_call": "[round(float(v), 10) for v in _oracle_compose_quaternion_sequence(batch).ravel()]",
        },
        # Invalid input must raise ValueError rather than return a quaternion.
        {
            "setup": "\n\nimport numpy as np\ndef _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "_exception_code(compose_quaternion_sequence, np.zeros((3, 3)))",
            "gold_call": "_exception_code(_oracle_compose_quaternion_sequence, np.zeros((3, 3)))",
        },
    ]
