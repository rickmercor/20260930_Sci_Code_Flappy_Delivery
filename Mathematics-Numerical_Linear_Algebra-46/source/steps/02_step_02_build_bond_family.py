"""
Build the reference lattice of material points, every ordered bond inside the horizon, and the normalized kernel weight carried by each bond.

A discretized nonlocal body replaces every neighbourhood integral by a sum over the lattice points inside the horizon, each bond weighted by a radial kernel normalized to unit integral over the full horizon ball.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: reference positions, bond start and end indices, and bond kernel weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_bond_family(
    grid_shape: tuple,
    spacing: float,
    horizon: float,
    profile_breaks: "np.ndarray",
    profile_coeffs: "np.ndarray",
) -> tuple:
    """Return lattice positions, ordered bonds and normalized kernel weights.

    The lattice has ``grid_shape = (N_x, N_y, N_z)`` nodes. Node
    ``(i, j, k)`` has index ``(i * N_y + j) * N_z + k`` and reference
    position ``spacing * (i, j, k)``. Every ordered pair of distinct nodes
    ``(a, b)`` with ``|X_b - X_a| <= horizon`` is a bond; bonds are sorted by
    start index and then by end index. The radial profile ``p(rho)``,
    ``rho = r / horizon``, equals ``sum_m profile_coeffs[j, m] * rho**m`` on
    ``[profile_breaks[j], profile_breaks[j + 1]]``; a radius on an interior
    break uses the interval to its right. The bond weight is
    ``omega = p(|X_b - X_a| / horizon) / (4 * pi * horizon**3 * M2)`` with
    ``M2 = int_0^1 p(rho) rho**2 drho``, so that the kernel integrates to one
    over the full ball of radius ``horizon``.

    Parameters
    ----------
    grid_shape : tuple
        Three positive integers ``(N_x, N_y, N_z)``.
    spacing : float
        Positive lattice spacing.
    horizon : float
        Horizon radius, at least ``spacing``.
    profile_breaks : np.ndarray
        Strictly increasing 1D array starting at 0 and ending at 1.
    profile_coeffs : np.ndarray
        2D array with one row of ascending-power coefficients per interval.

    Returns
    -------
    tuple
        ``(positions, bond_start, bond_end, bond_weight)``: an
        ``(N, 3)`` float array, two ``(N_b,)`` integer arrays and an
        ``(N_b,)`` float array.

    Raises
    ------
    ValueError
        If ``grid_shape`` is not three positive integers, if ``spacing`` is
        not a finite positive number, if ``horizon`` is not finite or is
        smaller than ``spacing``, if the breaks are not a strictly
        increasing 1D array of finite numbers from 0 to 1, if
        ``profile_coeffs`` is not a finite 2D array with one row per
        interval, or if ``M2`` is not positive.
    """
    return positions, bond_start, bond_end, bond_weight

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_bond_family(
    grid_shape: tuple,
    spacing: float,
    horizon: float,
    profile_breaks: "np.ndarray",
    profile_coeffs: "np.ndarray",
) -> tuple:
    """Reference implementation (integer offset stencil with a sorted bond list)."""
    import itertools
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    try:
        shape = tuple(grid_shape)
    except TypeError:
        raise ValueError("grid_shape must hold three integers") from None
    if len(shape) != 3 or not all(_is_integer(n) and n > 0 for n in shape):
        raise ValueError("grid_shape must hold three positive integers")
    if not (_is_number(spacing) and spacing > 0.0):
        raise ValueError("spacing must be a finite positive number")
    if not (_is_number(horizon) and horizon >= spacing):
        raise ValueError("horizon must be finite and at least the spacing")
    breaks = np.asarray(profile_breaks, dtype=float)
    coeffs = np.asarray(profile_coeffs, dtype=float)
    if breaks.ndim != 1 or breaks.size < 2 or not np.all(np.isfinite(breaks)):
        raise ValueError("profile_breaks must be a finite 1D array of length >= 2")
    if breaks[0] != 0.0 or breaks[-1] != 1.0 or np.any(np.diff(breaks) <= 0.0):
        raise ValueError("profile_breaks must increase strictly from 0 to 1")
    if coeffs.ndim != 2 or coeffs.shape[0] != breaks.size - 1 or not np.all(np.isfinite(coeffs)):
        raise ValueError("profile_coeffs needs one finite coefficient row per interval")
    powers = np.arange(coeffs.shape[1])
    m2 = float(np.sum(coeffs * (breaks[1:, None] ** (powers + 3)
                                - breaks[:-1, None] ** (powers + 3)) / (powers + 3)))
    if not m2 > 0.0:
        raise ValueError("the profile must have a positive second moment")

    nx, ny, nz = (int(n) for n in shape)
    grid = np.array(list(itertools.product(range(nx), range(ny), range(nz))), dtype=int)
    positions = float(spacing) * grid.astype(float)
    reach = int(np.floor(horizon / spacing)) + 1
    starts, ends, lengths = [], [], []
    for offset in itertools.product(range(-reach, reach + 1), repeat=3):
        length = float(spacing) * float(np.sqrt(np.dot(offset, offset)))
        if length == 0.0 or length > horizon:
            continue
        target = grid + np.array(offset)
        inside = np.all((target >= 0) & (target < np.array([nx, ny, nz])), axis=1)
        source = np.nonzero(inside)[0]
        starts.append(source)
        ends.append((target[inside, 0] * ny + target[inside, 1]) * nz + target[inside, 2])
        lengths.append(np.full(source.size, length))
    start = np.concatenate(starts).astype(np.int64)
    end = np.concatenate(ends).astype(np.int64)
    length = np.concatenate(lengths)
    order = np.lexsort((end, start))
    start, end, length = start[order], end[order], length[order]

    rho = length / float(horizon)
    piece = np.clip(np.searchsorted(breaks, rho, side="right") - 1, 0, coeffs.shape[0] - 1)
    profile = np.sum(coeffs[piece] * rho[:, None] ** powers, axis=1)
    weight = profile / (4.0 * np.pi * float(horizon) ** 3 * m2)
    return positions, start, end, weight

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return deterministic numerical comparisons with isolated inputs."""
    fixture_0 = (
        'import numpy as np\n'
        'def _fam(result):\n'
        '    pos, s, e, w = map(np.asarray, result)\n'
        '    assert pos.ndim == 2 and pos.shape[1] == 3\n'
        '    assert s.ndim == 1 and s.shape == e.shape == w.shape\n'
        '    return np.concatenate([pos.ravel(), s.astype(float), e.astype(float), w])\n'
        'def cubic():\n'
        '    return np.array([0.0, 0.5, 1.0]), np.array([[1.0, 0.0, -6.0, 6.0], [2.0, -6.0, 6.0, -2.0]])\n'
        'def linear():\n'
        '    return np.array([0.0, 1.0]), np.array([[1.0, -1.0]])\n'
        'def flat():\n'
        '    return np.array([0.0, 1.0]), np.array([[1.0]])\n'
    )
    fixture_1 = (
        'import numpy as np\n'
        'def _fam(result):\n'
        '    pos, s, e, w = map(np.asarray, result)\n'
        '    assert pos.ndim == 2 and pos.shape[1] == 3\n'
        '    assert s.ndim == 1 and s.shape == e.shape == w.shape\n'
        '    return np.concatenate([pos.ravel(), s.astype(float), e.astype(float), w])\n'
        'def cubic():\n'
        '    return np.array([0.0, 0.5, 1.0]), np.array([[1.0, 0.0, -6.0, 6.0], [2.0, -6.0, 6.0, -2.0]])\n'
        'def linear():\n'
        '    return np.array([0.0, 1.0]), np.array([[1.0, -1.0]])\n'
        'def flat():\n'
        '    return np.array([0.0, 1.0]), np.array([[1.0]])\n'
        'def _status(fn):\n'
        '    try:\n'
        '        fn()\n'
        '        return 0\n'
        '    except ValueError:\n'
        '        return 1\n'
        '    except Exception:\n'
        '        return 2\n'
    )
    return [
        {
            "setup": fixture_0,
            'call': '_fam(build_bond_family((3, 3, 3), 1.0, 1.5, *cubic()))',
            'gold_call': '_fam(_oracle_build_bond_family((3, 3, 3), 1.0, 1.5, *cubic()))',
        },
        {
            "setup": fixture_0,
            'call': '_fam(build_bond_family((4, 2, 3), 0.5, 1.1, *linear()))',
            'gold_call': '_fam(_oracle_build_bond_family((4, 2, 3), 0.5, 1.1, *linear()))',
        },
        {
            "setup": fixture_0,
            'call': '_fam(build_bond_family((1, 1, 2), 2.0, 2.5, *flat()))',
            'gold_call': '_fam(_oracle_build_bond_family((1, 1, 2), 2.0, 2.5, *flat()))',
        },
        {
            "setup": fixture_0,
            'call': '_fam(build_bond_family((5, 4, 3), 1.0, 3.015, *cubic())) / 100.0',
            'gold_call': '_fam(_oracle_build_bond_family((5, 4, 3), 1.0, 3.015, *cubic())) / 100.0',
        },
        {
            "setup": fixture_0,
            'call': 'float(np.sum(build_bond_family((6, 6, 6), 0.25, 0.5, *cubic())[3]) * 0.25 ** 3 / 216.0)',
            'gold_call': 'float(np.sum(_oracle_build_bond_family((6, 6, 6), 0.25, 0.5, *cubic())[3]) * 0.25 ** 3 / 216.0)',
        },
        {
            "setup": fixture_1,
            'call': '_status(lambda: build_bond_family((3, 3, 3), 1.0, 0.9, *cubic()))',
            'gold_call': '_status(lambda: _oracle_build_bond_family((3, 3, 3), 1.0, 0.9, *cubic()))',
        },
        {
            "setup": fixture_1,
            'call': '_status(lambda: build_bond_family((3, 0, 3), 1.0, 1.5, *cubic()))',
            'gold_call': '_status(lambda: _oracle_build_bond_family((3, 0, 3), 1.0, 1.5, *cubic()))',
        },
        {
            "setup": fixture_1,
            'call': '_status(lambda: build_bond_family((3, 3, 3), 1.0, 1.5, np.array([0.0, 1.0]), np.array([[0.0]])))',
            'gold_call': '_status(lambda: _oracle_build_bond_family((3, 3, 3), 1.0, 1.5, np.array([0.0, 1.0]), np.array([[0.0]])))',
        },
    ]
