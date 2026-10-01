"""
Convert the centered Fourier field into radial-bin centers, peak-radial-PSD-normalized density per polar angle, and polar bin-center angles. Use uniform radial bins on [0,K], where K=min(max(abs(kx)),max(abs(ky)),k0), and discard samples above K before accumulating their powers. Bins are left-closed and right-open except that the final edge is included. Return three numeric arrays of length nbins, with zero density for a zero retained field.

Annular powers sum abs(U)**2*delta_kx*delta_ky over retained Cartesian samples. Dividing by the common radial width and then the largest radial PSD removes common scale factors. Polar edges are arcsin(radial_edges/k0), and division by their unequal differences gives density per radian; center angles use radial midpoints. These are polar angles from z, not transverse azimuths. Filtering annular centers after summation would admit above-cutoff power and is not the defined operation.

Returns
-------
return np.zeros(bin_count), np.zeros(bin_count), np.zeros(bin_count)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def radial_angular_power(U, kx, ky, k0, nbins=None):
    """Reduce a centered Fourier field to a polar angular density.

    U has shape (len(kx), len(ky)); both axes are finite, increasing,
    uniformly spaced one-dimensional arrays with at least two samples.
    k0 is finite and positive. nbins is a positive integer, defaulting
    to min(len(kx), len(ky)). Let K=min(max(abs(kx)), max(abs(ky)), k0).
    Use nbins uniform radial bins on [0, K], left-closed/right-open
    except that the final bin includes K. Retain only individual samples
    with hypot(kx, ky) <= K and <= k0 before summing abs(U)**2*dkx*dky.
    Divide these sums by the common radial width, normalize that radial
    PSD by its peak, then divide by each angular-bin width. Angular edges
    are arcsin(radial_edges/k0); angles use radial-bin midpoints.
    Return (k_perp, angular_power_density, Pangle), three float arrays of
    length nbins. Density is zero if all retained power is zero. Polar
    angles are measured from z in radians; density has units rad**(-1).
    """
    bin_count = nbins if nbins is not None else min(len(kx), len(ky))
    return np.zeros(bin_count), np.zeros(bin_count), np.zeros(bin_count)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_radial_angular_power(U, kx, ky, k0, nbins=None):
    """Return radial centers, angular density, and polar angles.

    Uniform radial bins cover [0, min(max(abs(kx)), max(abs(ky)), k0)].
    Bins are left-closed and right-open except for the closed final bin.
    Filtering is applied to individual Fourier samples before accumulation.
    Density is peak-normalized radial PSD divided by angular-bin width.
    """
    spectrum = np.asarray(U, dtype=complex)
    first_axis = np.asarray(kx, dtype=float)
    second_axis = np.asarray(ky, dtype=float)

    if first_axis.ndim != 1 or second_axis.ndim != 1:
        raise ValueError("kx and ky must be one-dimensional.")
    if spectrum.shape != (first_axis.size, second_axis.size):
        raise ValueError("U shape must be (len(kx), len(ky)).")
    if first_axis.size < 2 or second_axis.size < 2:
        raise ValueError("kx and ky must each contain at least two values.")
    if not np.isfinite(k0) or k0 <= 0:
        raise ValueError("k0 must be finite and positive.")
    if not all(
        np.all(np.isfinite(values))
        for values in (spectrum, first_axis, second_axis)
    ):
        raise ValueError("The field and wavenumbers must be finite.")

    if nbins is None:
        nbins = min(first_axis.size, second_axis.size)
    if (
        isinstance(nbins, (bool, np.bool_))
        or not isinstance(nbins, (int, np.integer))
        or nbins < 1
    ):
        raise ValueError("nbins must be a positive integer.")

    first_differences = np.diff(first_axis)
    second_differences = np.diff(second_axis)
    for differences in (first_differences, second_differences):
        if differences[0] <= 0 or not np.allclose(
            differences, differences[0], rtol=1e-12, atol=0
        ):
            raise ValueError(
                "Wavenumbers must be increasing and uniformly spaced."
            )

    radial_limit = min(
        np.max(np.abs(first_axis)), np.max(np.abs(second_axis)), k0
    )
    if radial_limit <= 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    edges = np.linspace(0.0, radial_limit, int(nbins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radii = np.hypot(first_axis[:, None], second_axis[None, :])
    retained = (radii <= radial_limit) & (radii <= k0)
    spectral_scale = np.max(np.abs(spectrum[retained]), initial=0.0)

    if spectral_scale > 0:
        weights = np.abs(spectrum[retained] / spectral_scale) ** 2
    else:
        weights = np.zeros(np.count_nonzero(retained), dtype=float)

    annular_power, _ = np.histogram(
        radii[retained], bins=edges, weights=weights
    )
    peak = np.max(annular_power, initial=0.0)
    normalized_radial_psd = (
        annular_power / peak if peak > 0 else np.zeros(int(nbins))
    )
    angle_edges = np.arcsin(np.clip(edges / k0, 0.0, 1.0))
    angles = np.arcsin(np.clip(centers / k0, 0.0, 1.0))
    density = normalized_radial_psd / np.diff(angle_edges)
    return centers, density, angles

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

ny = 16
kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=0.5))
ky = kx.copy()
U = np.ones((ny, ny), dtype=complex)


def run_model():
    return radial_angular_power(U, kx, ky, 20.0, nbins=ny)


def run_gold():
    return _oracle_radial_angular_power(U, kx, ky, 20.0, nbins=ny)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np

ny = 16
kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=0.5))
ky = kx.copy()
U = np.zeros((ny, ny), dtype=complex)
U[ny // 2, ny // 2] = 1.0


def run_model():
    return radial_angular_power(U, kx, ky, 20.0, nbins=ny)


def run_gold():
    return _oracle_radial_angular_power(U, kx, ky, 20.0, nbins=ny)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np

ny = 32
kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=0.5))
ky = kx.copy()
first_grid, second_grid = np.meshgrid(kx, ky, indexing="ij")
radial_grid = np.sqrt(first_grid**2 + second_grid**2)
target_radius = 3.0
U = np.exp(-((radial_grid - target_radius) / 0.25)**2)


def run_model():
    return radial_angular_power(U, kx, ky, 20.0, nbins=ny)


def run_gold():
    return _oracle_radial_angular_power(U, kx, ky, 20.0, nbins=ny)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np

axis = np.arange(-2.0, 3.0)
field = np.zeros((5, 5), dtype=complex)
field[2, 2] = 1
field[3, 2] = 2
field[4, 2] = 1e150
centers = np.array([0.25, 0.75])
expected = (
    centers,
    np.array([0.25, 1.0]) / np.diff(np.arcsin([0.0, 0.5, 1.0])),
    np.arcsin(centers),
)
""",
            "call": "radial_angular_power(field, axis, axis, 1.0, 2)",
            "gold_call": "expected",
        },
        {
            "setup": """
import numpy as np

axis = np.array([-1.25, -0.25, 0.75, 1.75])
field = np.zeros((4, 4), dtype=complex)
field[2, 1] = 1
field[0, 1] = 1e150
expected = (
    np.array([0.5]),
    np.array([2 / np.pi]),
    np.arcsin(np.array([0.5])),
)
""",
            "call": "radial_angular_power(field, axis, axis, 1.0, 1)",
            "gold_call": "expected",
        },
        {
            "setup": """
import numpy as np

axis = np.arange(-2.0, 3.0)
field = np.zeros((5, 5), dtype=complex)
centers = np.array([0.25, 0.75])
expected = (centers, np.zeros(2), np.arcsin(centers))
""",
            "call": "radial_angular_power(field, axis, axis, 1.0, 2)",
            "gold_call": "expected",
        },
    ]
