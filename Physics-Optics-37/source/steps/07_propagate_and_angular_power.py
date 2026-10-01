"""
Run the field-propagation and angular-density pipeline to exactly zlength. Form the centered initial transform, take floor(zlength/h) nominal increments, and apply any positive remainder using its own coefficients and fresh factors. Zero distance returns the initial Fourier field without marching. Return the seven outputs in the Signature: Fourier field, Cartesian peak-normalized power, both frequency axes, radial centers, angular density, and polar angles. Leave plotting disabled unless explicitly requested. This step is not the final orchestrator of the scalar task.

The material is invariant along z, so factors may be reused across equal increments, but a different increment changes the rational approximation. Discarding the remainder solves a different propagation distance. All collocation modes are retained during marching; the output cutoff belongs to the final angular diagnostic. Cartesian peak normalization is a separate returned diagnostic and must not replace annular normalization. The fixed [4/4] benchmark is neither an eighth-order-in-h method nor an exact interface-transmission calculation.

Returns
-------
( U,  normalized_power,  kx, ky,  k_perp, radial_power, Pangle)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def propagate_and_angular_power(
    U0, P, k0, h, zlength, dx, dy, mx, B, C, nbins=None, plot=False
):
    """Propagate the centered square physical field U0 to zlength.

    P is a positive integer; k0, h, dx, and dy are finite and positive.
    zlength is finite and nonnegative; mx is an integer Fourier shift.
    B and C are finite real modulation parameters. nbins defaults to
    the transverse size. plot=False suppresses plotting.
    Transform once using the convention of initial_fourier_transform.
    Take floor(zlength/h) full steps with t=k0*h, then any positive
    remainder with its own coefficients and fresh factors. Zero distance
    returns the initial FFT without propagation. Use X=M_tilde-I-D and
    the Pade function specified in Step 1.
    Return (U, normalized_power, kx, ky, k_perp, angular_density, Pangle).
    U is the final centered Fourier field. normalized_power is abs(U)**2
    divided by its Cartesian peak, or zero for a zero field. The last
    three arrays follow radial_angular_power, not Cartesian normalization.
    """
    size = np.shape(U0)[0]
    bin_count = nbins if nbins is not None else size
    spectrum = np.zeros_like(U0, dtype=complex)
    power = np.zeros_like(U0, dtype=float)
    kx = np.zeros(size)
    ky = np.zeros(size)
    centers = np.zeros(bin_count)
    density = np.zeros(bin_count)
    angles = np.zeros(bin_count)
    return spectrum, power, kx, ky, centers, density, angles

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_propagate_and_angular_power(
    U0, P, k0, h, zlength, dx, dy, mx, B, C, nbins=None, plot=False
):
    """Propagate to zlength, including a remainder with fresh factors.

    Return (U, normalized_power, kx, ky, k_perp, angular_density, Pangle).
    U is the centered unnormalized Fourier envelope, and normalized_power
    is its Cartesian power divided by its peak. The last three outputs
    follow radial_angular_power. A zero distance returns the initial FFT.
    """
    initial = np.asarray(U0, dtype=complex)
    if (
        initial.ndim != 2
        or initial.shape[0] != initial.shape[1]
        or initial.shape[0] < 2
    ):
        raise ValueError(
            "U0 must be a square 2D array with at least two samples per axis."
        )
    if not np.all(np.isfinite(initial)):
        raise ValueError("U0 must be finite.")
    if (
        isinstance(P, (bool, np.bool_))
        or not isinstance(P, (int, np.integer))
        or P < 1
    ):
        raise ValueError("P must be a positive integer.")
    if not all(np.isfinite(value) and value > 0 for value in (k0, h, dx, dy)):
        raise ValueError("k0, h, dx, and dy must be finite and positive.")
    if not np.isfinite(zlength) or zlength < 0:
        raise ValueError("zlength must be finite and nonnegative.")

    spectrum, first_axis, second_axis = _oracle_initial_fourier_transform(
        initial, dx, dy
    )
    step_count = int(np.floor(zlength / h))
    remainder = float(zlength - step_count * h)
    cache = None

    if step_count:
        poles, residues = _oracle_PadeCoeff(P, k0 * h)
        for _ in range(step_count):
            spectrum, cache = _oracle_apply_pade(
                first_axis, second_axis, mx, B, C, k0, spectrum,
                poles, residues, cache
            )

    if remainder > 0:
        poles, residues = _oracle_PadeCoeff(P, k0 * remainder)
        spectrum, _ = _oracle_apply_pade(
            first_axis, second_axis, mx, B, C, k0, spectrum,
            poles, residues, None
        )

    if not np.all(np.isfinite(spectrum)):
        raise ValueError("The propagated field is not finite.")
    peak_amplitude = np.max(np.abs(spectrum), initial=0.0)
    normalized_power = (
        np.abs(spectrum / peak_amplitude) ** 2
        if peak_amplitude > 0
        else np.zeros(spectrum.shape, dtype=float)
    )
    centers, density, angles = _oracle_radial_angular_power(
        spectrum, first_axis, second_axis, k0, nbins
    )

    if plot:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(1, 2)
        axes[0].pcolormesh(
            first_axis, second_axis, normalized_power.T, shading="auto"
        )
        axes[0].set(xlabel="kx (rad/m)", ylabel="ky (rad/m)")
        axes[1].plot(angles, density)
        axes[1].set(
            xlabel="Polar angle (rad)", ylabel="Angular density (rad^-1)"
        )
        plt.tight_layout()
        plt.show()

    return (
        spectrum, normalized_power, first_axis, second_axis,
        centers, density, angles
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

ny = 8
dx = dy = 0.5
rng = np.random.default_rng(4)
U0 = rng.random((ny, ny)) + 1j * rng.random((ny, ny))
parameters = dict(
    U0=U0, P=1, k0=20.0, h=0.001, zlength=0.002,
    dx=dx, dy=dy, mx=1, B=0.01, C=0.1, nbins=8, plot=False,
)
""",
            "call": "propagate_and_angular_power(**parameters)",
            "gold_call": "_oracle_propagate_and_angular_power(**parameters)",
        },
        {
            "setup": """
import numpy as np

ny = 8
dx = dy = 0.5
U0 = np.zeros((ny, ny), dtype=complex)
U0[ny // 2, ny // 2] = 1.0
parameters = dict(
    U0=U0, P=2, k0=20.0, h=0.001, zlength=0.003,
    dx=dx, dy=dy, mx=ny // 2, B=0.01, C=0.1, nbins=8, plot=False,
)
""",
            "call": "propagate_and_angular_power(**parameters)",
            "gold_call": "_oracle_propagate_and_angular_power(**parameters)",
        },
        {
            "setup": """
import numpy as np

ny = 8
dx = dy = 0.5
U0 = np.zeros((ny, ny), dtype=complex)
parameters = dict(
    U0=U0, P=1, k0=20.0, h=0.001, zlength=0.002,
    dx=dx, dy=dy, mx=ny, B=0.01, C=0.1, nbins=8, plot=False,
)
""",
            "call": "propagate_and_angular_power(**parameters)",
            "gold_call": "_oracle_propagate_and_angular_power(**parameters)",
        },
        {
            "setup": """
import numpy as np

U0 = np.zeros((4, 4), dtype=complex)
U0[2, 2] = 1.0 + 0.25j
parameters = dict(
    U0=U0, P=4, k0=2.0, h=0.1, zlength=0.0,
    dx=3.0, dy=2.5, mx=1, B=0.07, C=1.1, plot=False,
)
modes = np.array([-2.0, -1.0, 0.0, 1.0])
edges = np.linspace(0.0, np.pi / 3.0, 5)
centers = 0.5 * (edges[:-1] + edges[1:])
expected = (
    np.full((4, 4), 1.0 + 0.25j, dtype=complex),
    np.ones((4, 4)),
    np.pi * modes / 6.0,
    np.pi * modes / 5.0,
    centers,
    np.array([0.2, 0.0, 0.8, 1.0]) / np.diff(np.arcsin(edges / 2.0)),
    np.arcsin(centers / 2.0),
)
""",
            "call": "propagate_and_angular_power(**parameters)",
            "gold_call": "expected",
        },
        {
            "setup": """
import numpy as np

U0 = np.eye(4, dtype=complex)
U0 += 0.25j * np.roll(U0, 1, axis=0)
parameters = dict(
    U0=U0, P=4, k0=2.0, h=0.1, zlength=0.05,
    dx=3.0, dy=2.5, mx=1, B=0.07, C=1.1, nbins=4, plot=False,
)
expected_spectrum = np.array([
    [
        3.971902408361656 - 1.1059831897840957j,
        6.46787505429458e-05 + 0.01989640070496599j,
        -4.204731163132136e-05 - 5.958195207614314e-05j,
        3.880725032623378e-05 + 0.011937840422979705j,
    ],
    [
        0.0052169845021281525 + 0.019206804318854742j,
        -7.377079936521369e-05 - 0.00012526551626295193j,
        -0.00367082630312654 + 0.014176112578843193j,
        2.9999490221808998 + 0.006272438014681868j,
    ],
    [
        -0.0001997756003391249 - 0.0004073533132500917j,
        -6.906105337003976e-05 + 0.018559062788259882j,
        3.9897532699077054 + 1.0399230575747436j,
        -4.143663202257897e-05 + 0.0111354376729563j,
    ],
    [
        0.0052169845021285965 + 0.019206804318854964j,
        4.999915036968167 + 0.010454063357803349j,
        -0.003670826303126762 + 0.01417611257884354j,
        -4.426247961886176e-05 - 7.515930975784523e-05j,
    ],
], dtype=complex)
power = np.abs(expected_spectrum)**2
modes = np.array([-2.0, -1.0, 0.0, 1.0])
edges = np.linspace(0.0, np.pi / 3.0, 5)
centers = 0.5 * (edges[:-1] + edges[1:])
expected = (
    expected_spectrum,
    power / power.max(),
    np.pi * modes / 6.0,
    np.pi * modes / 5.0,
    centers,
    np.array([
        3.8087735956509317, 0.0, 0.0001903141543351773,
        6.779432300754655,
    ]),
    np.arcsin(centers / 2.0),
)
""",
            "call": "propagate_and_angular_power(**parameters)",
            "gold_call": "expected",
        },
        {
            "setup": """
import numpy as np

U0 = np.eye(4, dtype=complex)
U0 += 0.25j * np.roll(U0, 1, axis=0)
parameters = dict(
    U0=U0, P=4, k0=2.0, h=0.1, zlength=0.25,
    dx=3.0, dy=2.5, mx=1, B=0.07, C=1.1, nbins=4, plot=False,
)
expected_spectrum = np.array([
    [
        3.829863540995886 - 1.5206410995176158j,
        0.0016165272939834274 + 0.09943471188795014j,
        -0.0012742608987889037 - 0.0005946865398945428j,
        0.0009699163763903229 + 0.05966082713277026j,
    ],
    [
        0.03375087940112609 + 0.09356260279806772j,
        -0.0018438402504266094 - 0.0006323704814267389j,
        -0.020720491468912167 + 0.07020801904785544j,
        2.9987258138658954 + 0.03135799598573511j,
    ],
    [
        -0.0027895452892009454 - 0.00139901394289893j,
        -0.001726206913561401 + 0.09275675649974176j,
        3.943795773015232 + 1.198230325915516j,
        -0.0010357241481368795 + 0.055654053899845415j,
    ],
    [
        0.03375087940112609 + 0.09356260279806766j,
        4.997876356443158 + 0.05226332664289209j,
        -0.0207204914689125 + 0.07020801904785574j,
        -0.0011063041502556992 - 0.00037942228885612037j,
    ],
], dtype=complex)
power = np.abs(expected_spectrum)**2
modes = np.array([-2.0, -1.0, 0.0, 1.0])
edges = np.linspace(0.0, np.pi / 3.0, 5)
centers = 0.5 * (edges[:-1] + edges[1:])
expected = (
    expected_spectrum,
    power / power.max(),
    np.pi * modes / 6.0,
    np.pi * modes / 5.0,
    centers,
    np.array([
        3.8091734259953305, 0.0, 0.004758967146160304,
        6.779432300754655,
    ]),
    np.arcsin(centers / 2.0),
)
""",
            "call": "propagate_and_angular_power(**parameters)",
            "gold_call": "expected",
        },
        {
            "setup": """
import numpy as np

U0 = np.zeros((4, 4), dtype=complex)
parameters = dict(
    U0=U0, P=4, k0=2.0, h=0.1, zlength=0.25,
    dx=3.0, dy=2.5, mx=1, B=0.07, C=1.1, nbins=3, plot=False,
)
modes = np.array([-2.0, -1.0, 0.0, 1.0])
edges = np.linspace(0.0, np.pi / 3.0, 4)
centers = 0.5 * (edges[:-1] + edges[1:])
expected = (
    np.zeros((4, 4), dtype=complex),
    np.zeros((4, 4)),
    np.pi * modes / 6.0,
    np.pi * modes / 5.0,
    centers,
    np.zeros(3),
    np.arcsin(centers / 2.0),
)
""",
            "call": "propagate_and_angular_power(**parameters)",
            "gold_call": "expected",
        },
    ]
