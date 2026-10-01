"""
Propagate an unrefined spectral approximation and an exact reference through the same layers.

In the real propagation basis, form



$$

\widetilde K=R_{13}R_{12}\operatorname{diag}(0,m_{21},m_{31})R_{12}^TR_{13}^T+\operatorname{diag}(a,0,0),\qquad

a=\sigma(1.526493231029146\times10^{-4})qE.

$$



For the approximate branch, use zero polynomial corrections in the matter

spectrum and electron-row rank-one completion with $Q_{\tau\tau}=1-Q_{ee}-Q_{\mu\mu}$.

For the reference, use the exact real-symmetric eigensystem. In each branch remove

its own lowest-eigenvalue phase and build



$$

\widetilde S=I+Q_2\operatorname{expm1}(-i\kappa(\lambda_2-\lambda_1)L/E)

+Q_3\operatorname{expm1}(-i\kappa(\lambda_3-\lambda_1)L/E),\qquad

\kappa=10^{-9}10^3/[2(1.97327\times10^{-7})].

$$



Compose source-to-detector layers as $S_N\cdots S_1$.

For exactly palindromic input rows, symmetry gives $B^TCB$ with inbound product

$B$ and central layer $C$ (identity for even length).

The approximate matrices need not be exact projectors or exactly unitary;

do not orthogonalize, renormalize, or clip them. Independent branch-wide scalar

phases disappear from probabilities. Empty profiles return two identities.

Returns
-------
A complex array of shape (2, 3, 3) containing the unrefined and exact phase-reduced profile amplitudes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_spectral_pair(
    layers: "np.ndarray", energy_gev: float, mixing: "np.ndarray", charge: int = 1
) -> "np.ndarray":
    r"""Propagate an unrefined spectral approximation and an exact reference through the same layers.

    Parameters
    ----------
    layers : np.ndarray
        Finite real shape (N, 2) of nonnegative lengths in km and density
        products q in grams per cubic centimeter, in propagation order.
    energy_gev : float
        Finite positive energy in GeV.
    mixing : np.ndarray
        Finite real shape (4,) vector s12_sq, s13_sq, m21, m31; squared
        sines in (0, 1), mass-squared differences 0 < m21 < m31 in eV squared.
    charge : int, default 1
        +1 for neutrinos or -1 for antineutrinos; Boolean values are invalid.

    Returns
    -------
    amplitudes : np.ndarray
        Complex shape (2, 3, 3), approximate then exact, with the lowest
        layer eigenvalue phases removed separately in each branch.

    Raises
    ------
    ValueError
        If input shapes, finiteness, ranges or charge are invalid, or the
        approximate spectrum or electron-row pivot cannot be recovered.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _layer_amplitude(
    eigenvalues_ev2: "np.ndarray",
    projectors: "np.ndarray",
    length_km: float,
    energy_gev: float,
) -> "np.ndarray":
    roots = _real_array(eigenvalues_ev2, (3,), "eigenvalues_ev2")
    proj = _real_array(projectors, (2, 3, 3), "projectors")
    length = _finite_scalar(length_km, "length_km")
    energy = _finite_scalar(energy_gev, "energy_gev")
    if length < 0.0 or energy <= 0.0 or np.any(np.diff(roots) <= 0.0):
        raise ValueError("invalid length, energy, or eigenvalue ordering")
    if not np.allclose(proj, proj.transpose(0, 2, 1), rtol=0.0, atol=1e-10):
        raise ValueError("projectors must be symmetric")
    conversion = 1e-9 * 1e3 / (2.0 * 1.97327e-7)
    phases = conversion * (roots[1:] - roots[0]) * length / energy
    return np.eye(3, dtype=complex) + np.einsum(
        "i,ijk->jk", np.expm1(-1j * phases), proj
    )


def _compose_amplitude(
    layer_amplitudes: "np.ndarray", symmetric: bool = False
) -> "np.ndarray":
    layers = np.asarray(layer_amplitudes, dtype=complex)
    if (
        layers.ndim != 3
        or layers.shape[1:] != (3, 3)
        or not np.all(np.isfinite(layers))
    ):
        raise ValueError("layers must have finite shape (N, 3, 3)")
    if not isinstance(symmetric, (bool, np.bool_)):
        raise ValueError("symmetric must be Boolean")
    if symmetric:
        if not np.allclose(layers, layers.transpose(0, 2, 1), rtol=0.0, atol=1e-10):
            raise ValueError("individual layers must be symmetric")
        if not np.allclose(layers, layers[::-1], rtol=0.0, atol=1e-10):
            raise ValueError("the layer sequence must be palindromic")
        half = len(layers) // 2
        inbound = np.eye(3, dtype=complex)
        for layer in layers[:half]:
            inbound = layer @ inbound
        center = layers[half] if len(layers) % 2 else np.eye(3, dtype=complex)
        return inbound.T @ center @ inbound
    amplitude = np.eye(3, dtype=complex)
    for layer in layers:
        amplitude = layer @ amplitude
    return amplitude


def _vacuum_matrix(parameters):
    s12, s13, m21, m31 = parameters
    x, y = np.sqrt([s12, s13])
    u, v = np.sqrt([1.0 - s12, 1.0 - s13])
    r12 = np.array([[u, x, 0.0], [-x, u, 0.0], [0.0, 0.0, 1.0]])
    r13 = np.array([[v, 0.0, y], [0.0, 1.0, 0.0], [-y, 0.0, v]])
    rotation = r13 @ r12
    return rotation @ np.diag([0.0, m21, m31]) @ rotation.T


def _oracle_propagate_spectral_pair(
    layers: "np.ndarray", energy_gev: float, mixing: "np.ndarray", charge: int = 1
) -> "np.ndarray":
    raw = np.asarray(layers)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("layers must have shape (N, 2)")
    data = _real_array(layers, raw.shape, "layers")
    energy = _finite_scalar(energy_gev, "energy_gev")
    parameters = _real_array(mixing, (4,), "mixing")
    _validate_mixing(*parameters)
    if np.any(data < 0.0) or energy <= 0.0:
        raise ValueError("invalid layer values or energy")
    if (
        isinstance(charge, (bool, np.bool_))
        or not isinstance(charge, (int, np.integer))
        or charge not in (-1, 1)
    ):
        raise ValueError("charge must be +1 or -1")
    vacuum = _vacuum_matrix(parameters)
    amplitudes = np.empty((2, len(data), 3, 3), dtype=complex)
    for index, (length, density) in enumerate(data):
        if length == 0.0:
            amplitudes[:, index] = np.eye(3)
            continue
        potential = charge * 1.526493231029146e-4 * density * energy
        spectrum = _oracle_compute_matter_spectrum(potential, *parameters, 0)
        projectors = _oracle_compute_eigenprojectors(spectrum[0], spectrum[1:])
        amplitudes[0, index] = _layer_amplitude(spectrum[0], projectors, length, energy)
        matrix = vacuum + np.diag([potential, 0.0, 0.0])
        roots, vectors = np.linalg.eigh(matrix)
        exact_projectors = np.array(
            [np.outer(vectors[:, j], vectors[:, j]) for j in (1, 2)]
        )
        amplitudes[1, index] = _layer_amplitude(roots, exact_projectors, length, energy)
    symmetric = np.array_equal(data, data[::-1])
    return np.array([_compose_amplitude(branch, symmetric) for branch in amplitudes])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative numerical and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\np=np.array([.307,.02195,7.49e-5,.002534])\na=np.array([[2891.,2.2275],[6960.,5.137],[2891.,2.2275]])\n",
            "call": "propagate_spectral_pair(a.copy(), 6.0, p.copy())",
            "gold_call": "_oracle_propagate_spectral_pair(a.copy(), 6.0, p.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\np=np.array([.307,.02195,7.49e-5,.002534])\na=np.empty((0,2))\n",
            "call": "propagate_spectral_pair(a.copy(), 6.0, p.copy())",
            "gold_call": "_oracle_propagate_spectral_pair(a.copy(), 6.0, p.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\np=np.array([.3,.03,8e-5,.0024])\na=np.array([[250.,1.3],[1900.,4.2],[700.,2.1],[0.,0.]])\n",
            "call": "propagate_spectral_pair(a.copy(), 3.0, p.copy(), -1)",
            "gold_call": "_oracle_propagate_spectral_pair(a.copy(), 3.0, p.copy(), -1)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\np=np.array([.3,.03,8e-5,.0024])\na=np.array([[300.,0.],[200.,1.],[200.,1.],[300.,0.]])\n",
            "call": "propagate_spectral_pair(a.copy(), 1.7, p.copy())",
            "gold_call": "_oracle_propagate_spectral_pair(a.copy(), 1.7, p.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\np=np.array([.307,.02195,7.49e-5,.002534])\na=np.array([[-1.,2.]])\ndef _raises_value_error(function):\n    try:\n        function()\n    except ValueError:\n        return 1\n    return 0\n",
            "call": "_raises_value_error(lambda: propagate_spectral_pair(a.copy(), 6.0, p.copy()))",
            "gold_call": "_raises_value_error(lambda: _oracle_propagate_spectral_pair(a.copy(), 6.0, p.copy()))",
            "tol": 0.0,
        },
    ]
