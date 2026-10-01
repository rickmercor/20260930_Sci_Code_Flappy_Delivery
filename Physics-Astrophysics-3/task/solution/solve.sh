#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

def seed_field(N, A, sigma, kx):
    """Reference implementation of seed_field (deterministic)."""
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)):
        raise ValueError("N must be an integer")
    N = int(N)
    if N < 2:
        raise ValueError("N must be >= 2")

    if isinstance(kx, bool) or not isinstance(kx, (int, np.integer)):
        raise ValueError("kx must be an integer")
    kx = int(kx)
    if kx < 0:
        raise ValueError("kx must be non-negative")

    try:
        A = float(A)
    except (TypeError, ValueError):
        raise ValueError("A must be a real number")
    if not np.isfinite(A):
        raise ValueError("A must be finite")

    try:
        sigma = float(sigma)
    except (TypeError, ValueError):
        raise ValueError("sigma must be a real number")
    if not np.isfinite(sigma):
        raise ValueError("sigma must be finite")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    coord = np.arange(N, dtype=np.float64) / float(N)
    X, Y, Z = np.meshgrid(coord, coord, coord, indexing="ij")

    dr2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    env = A * np.exp(-dr2 / (2.0 * sigma ** 2))
    phi = 2.0 * np.pi * float(kx) * X

    F0 = np.empty((3, N, N, N), dtype=np.float64)
    F0[0] = 1.0
    F0[1] = np.cos(phi) * env
    F0[2] = np.sin(phi) * env
    return F0

import numpy as np

def wavevector_grid(N):
    """Reference implementation of :func:`wavevector_grid`.

    Parameters
    ----------
    N : int
        Number of grid points along each axis of the cube. Must be an integer
        greater than or equal to 2.

    Returns
    -------
    numpy.ndarray
        Array ``K`` of shape (3, N, N, N) and dtype float64 with
        ``K[0] = KX``, ``K[1] = KY`` and ``K[2] = KZ``.

    Raises
    ------
    ValueError
        If ``N`` is not an integer or if ``N`` is smaller than 2.
    """
    if isinstance(N, bool):
        raise ValueError("N must be an integer, got a bool")
    if isinstance(N, (int, np.integer)):
        n_int = int(N)
    elif isinstance(N, (float, np.floating)):
        if not float(N).is_integer():
            raise ValueError("N must be an integer, got %r" % (N,))
        n_int = int(N)
    else:
        raise ValueError("N must be an integer, got type %s" % type(N).__name__)
    if n_int < 2:
        raise ValueError("N must be >= 2, got %d" % n_int)

    k1 = 2.0 * np.pi * np.fft.fftfreq(n_int, d=1.0 / n_int)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    K = np.empty((3, n_int, n_int, n_int), dtype=np.float64)
    K[0] = KX
    K[1] = KY
    K[2] = KZ
    return K

import numpy as np

def solenoidal_projection(F: np.ndarray) -> np.ndarray:
    arr = np.asarray(F, dtype=float)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError("F must be an array of shape (3, N, N, N)")
    n = int(arr.shape[1])
    if arr.shape[2] != n or arr.shape[3] != n:
        raise ValueError("F must be defined on a cubic (N, N, N) grid")
    if n < 2:
        raise ValueError("N must be >= 2")

    kgrid = wavevector_grid(n)
    KX, KY, KZ = kgrid[0], kgrid[1], kgrid[2]

    Fh = np.fft.fftn(arr, axes=(1, 2, 3))

    nyq = n // 2
    if n % 2 == 0:
        Fh[:, nyq, :, :] = 0.0
        Fh[:, :, nyq, :] = 0.0
        Fh[:, :, :, nyq] = 0.0

    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K2[0, 0, 0] = 1.0

    dot = KX * Fh[0] + KY * Fh[1] + KZ * Fh[2]
    Gh = np.empty_like(Fh)
    Gh[0] = Fh[0] - KX * dot / K2
    Gh[1] = Fh[1] - KY * dot / K2
    Gh[2] = Fh[2] - KZ * dot / K2

    Gh[:, 0, 0, 0] = Fh[:, 0, 0, 0]

    if n % 2 == 0:
        Gh[:, nyq, :, :] = 0.0
        Gh[:, :, nyq, :] = 0.0
        Gh[:, :, :, nyq] = 0.0

    return np.real(np.fft.ifftn(Gh, axes=(1, 2, 3)))

import numpy as np

def unit_normalise(G):
    """Reference implementation of the pointwise unit-magnitude normalisation.

    Parameters
    ----------
    G : numpy.ndarray
        Vector field of shape (3, N, N, N), component-first, on the periodic unit
        cube.  Sites whose magnitude is exactly zero are left unchanged.

    Returns
    -------
    numpy.ndarray
        Normalised field of shape (3, N, N, N), float64.

    Raises
    ------
    ValueError
        If the input is not a (3, N, N, N) array with equal trailing dimensions.
    """
    arr = np.asarray(G)
    if arr.ndim != 4:
        raise ValueError("field must have 4 dimensions (3, N, N, N)")
    if arr.shape[0] != 3:
        raise ValueError("field must be component-first with shape (3, N, N, N)")
    if not (arr.shape[1] == arr.shape[2] == arr.shape[3]):
        raise ValueError("field must be defined on a cubic grid (3, N, N, N)")
    if arr.shape[1] < 1:
        raise ValueError("grid size N must be at least 1")

    field = np.asarray(arr, dtype=np.float64)
    mag = np.sqrt(np.sum(field ** 2, axis=0))
    safe = np.where(mag == 0.0, 1.0, mag)
    return field / safe

import numpy as np

def deflection_angles(B: np.ndarray) -> np.ndarray:
    F = np.asarray(B, dtype=float)
    if F.ndim != 4:
        raise ValueError("B must be a 4-dimensional array of shape (3, N, N, N)")
    if F.shape[0] != 3:
        raise ValueError("B must carry 3 components on the leading axis")
    n = int(F.shape[1])
    if F.shape[2] != n or F.shape[3] != n:
        raise ValueError("B must be cubic: the three trailing axes must be equal")
    if n < 1:
        raise ValueError("N must be at least 1")

    mag = np.sqrt(np.sum(F * F, axis=0))
    if np.any(mag == 0.0):
        raise ValueError(
            "B has a grid site of zero magnitude; the deflection angle is "
            "undefined there")

    # The clip guards arccos against ratios that float a few ulp past +/-1 at
    # exactly axial sites; it never moves a ratio interior to [-1, 1].
    return np.degrees(np.arccos(np.clip(F[0] / mag, -1.0, 1.0)))

import numpy as np

def max_divergence(B: np.ndarray) -> float:
    arr = np.asarray(B, dtype=float)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError("B must be an array of shape (3, N, N, N)")
    n = int(arr.shape[1])
    if arr.shape[2] != n or arr.shape[3] != n:
        raise ValueError("B must be defined on a cubic (N, N, N) grid")
    if n < 2:
        raise ValueError("N must be >= 2")

    kgrid = wavevector_grid(n)
    KX, KY, KZ = kgrid[0], kgrid[1], kgrid[2]

    Bh = np.fft.fftn(arr, axes=(1, 2, 3))
    div_hat = 1j * (KX * Bh[0] + KY * Bh[1] + KZ * Bh[2])
    div = np.real(np.fft.ifftn(div_hat))
    return float(np.max(np.abs(div)))

import numpy as np

def magnitude_defect(B: np.ndarray) -> float:
    arr = np.asarray(B, dtype=float)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError("B must be an array of shape (3, N, N, N)")
    n = int(arr.shape[1])
    if arr.shape[2] != n or arr.shape[3] != n:
        raise ValueError("B must be defined on a cubic (N, N, N) grid")
    if n < 2:
        raise ValueError("N must be >= 2")

    mag = np.sqrt(arr[0] ** 2 + arr[1] ** 2 + arr[2] ** 2)
    return float(np.std(mag))

import numpy as np

def packet_max_deflection(N, A, sigma, kx, n_iter):
    """Reference implementation of :func:`packet_max_deflection`.

    Parameters
    ----------
    N : int
        Number of grid points along each axis of the periodic unit cube. Must be
        an integer greater than or equal to 2.
    A : float
        Amplitude of the Gaussian envelope of the seed field.
    sigma : float
        Width of the Gaussian envelope of the seed field. Must be positive.
    kx : int
        Integer number of transverse rotations of the seed along x.
    n_iter : int
        Number of projection/normalisation cycles. Must be an integer greater
        than or equal to 1.

    Returns
    -------
    float
        Maximum deflection angle in degrees over all grid sites of the final
        projected field ``G``.

    Raises
    ------
    ValueError
        If ``n_iter`` is not an integer or is smaller than 1, or if any of
        ``N``, ``A``, ``sigma`` or ``kx`` fails the seed-field validation.
    """
    if isinstance(n_iter, bool):
        raise ValueError("n_iter must be an integer, got a bool")
    if isinstance(n_iter, (int, np.integer)):
        n_steps = int(n_iter)
    elif isinstance(n_iter, (float, np.floating)):
        if not float(n_iter).is_integer():
            raise ValueError("n_iter must be an integer, got %r" % (n_iter,))
        n_steps = int(n_iter)
    else:
        raise ValueError(
            "n_iter must be an integer, got type %s" % type(n_iter).__name__
        )
    if n_steps < 1:
        raise ValueError("n_iter must be >= 1, got %d" % n_steps)

    # The seed builder performs the step-01 validation of N, A, sigma and kx and
    # raises ValueError on bad input.  The golden composes the pipeline from the
    # oracle implementations of steps 01-07 directly, so the reference answer
    # never depends on a submitted public function.
    F = seed_field(N, A, sigma, kx)

    # n_steps >= 1, so the loop always runs and G is the final projected field.
    G = np.asarray(F, dtype=np.float64)
    for _ in range(n_steps):
        G = solenoidal_projection(F)
        F = unit_normalise(G)

    theta = deflection_angles(G)

    # Certify the delivered iterate against the two constraints before
    # reporting: the spectral grid must match the field, the hard constraint
    # must be measurable, and the soft-constraint residual must be finite.  A
    # non-finite diagnostic means the pipeline was corrupted upstream.
    K = wavevector_grid(G.shape[1])
    div_max = max_divergence(G)
    defect = magnitude_defect(G)
    if K.shape[1:] != G.shape[1:]:
        raise ValueError("wavevector grid does not match the field grid")
    if not (np.isfinite(div_max) and np.isfinite(defect)
            and np.all(np.isfinite(theta))):
        raise ValueError("non-finite diagnostic in the delivered field")

    # The hard constraint is imposed exactly, so the delivered field is
    # solenoidal to round-off at every cycle count: the largest value over
    # every shipped configuration is 1.4e-13.  Refuse to report an angle
    # for a field that fails it.  The soft constraint is not gated here -
    # the magnitude defect is still 0.44 after a single cycle and is only
    # approached asymptotically.
    if div_max > 1.0e-6:
        raise ValueError(
            "delivered field is not solenoidal: max|div| = %.3e" % div_max
        )

    return float(np.max(theta))
SCICODE_GOLD_EOF
