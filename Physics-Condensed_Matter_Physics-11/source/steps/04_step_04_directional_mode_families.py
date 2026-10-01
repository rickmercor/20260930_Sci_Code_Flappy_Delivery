"""
The source method assigns the segment's modes to forward and backward group-velocity families to define its basic-mode indexing and retained modal subset. These are branch labels, not necessarily literal directions of physical energy flux in a driven medium. The assignment cannot be made by eigenvalue index, because the order returned by an eigensolver is a numerical convention rather than a physical label. It cannot be made by the sign of the wavenumber either, since the shifted wavenumber of a harmonic may have either sign on either family. It must be made using a branch property computed from the mode.

The source method uses group velocity. Differentiating the harmonic balance along a branch and projecting onto the mode gives it without finite differencing. Because the coefficient matrices assembled earlier are symmetric, the left eigenvector is the transpose of the right one, and the derivative of frequency with respect to wavenumber reduces to the ratio of the mode projected through the shifted-wavenumber diagonal and the coupling matrix to the mode projected through the shifted-frequency diagonal and scaled by density. Positive group velocity labels the forward family and negative group velocity labels the backward family. In the graded supersonic configuration the split is even, with 2 n_order + 1 modes in each family, which is the required source-method family count.

Time-averaged power flux remains the physical measure of energy-flow direction, but its sign does not reproduce these group-velocity family labels at the graded anticrossing. Two nearly degenerate modes can both carry positive flux while their group velocities have opposite signs. Flux signs therefore give an 18/16 split for the complete N=8 spectrum, whereas group-velocity labels give 17/17. This mismatch does not make the flux sign physically invalid and does not make unequal column groupings intrinsically impossible to match. It means that flux sign cannot substitute for the source method's equal-family group-velocity convention in this function.

Within each family the modes are labelled by basic-mode order. At vanishing modulation depth their physical wavenumbers are kappa_s^+ = (omega+s omega_m)/c0 - s kappa_m and kappa_s^- = -(omega+s omega_m)/c0 - s kappa_m. For omega_m > 0 and |omega_m/kappa_m| > c0, the forward physical wavenumber increases strictly with s and the backward physical wavenumber decreases strictly with s for either sign of kappa_m. Ordering the forward family by increasing real physical wavenumber and the backward family by decreasing real physical wavenumber therefore assigns s from minus n_order upwards in both. That correspondence survives the anticrossings, where a rule based on the dominant harmonic does not because the modes hybridise.

Returns
-------
dict holding complex128 forward_wavenumbers and backward_wavenumbers, each of shape (2 n_order + 1,) and indexed by basic-mode order from minus n_order to plus n_order; complex128 forward_modes and backward_modes, each of shape (2 n_order + 1, 2 n_order + 1) whose column s holds the mode shape of that basic mode; and a complex128 group_velocities of shape (4 n_order + 2,) in the order the wavenumbers were supplied.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def directional_mode_families(
    wavenumbers: np.ndarray,
    mode_shapes: np.ndarray,
    coupling: np.ndarray,
    omega: float,
    rho0: float,
    kappa_m: float,
    omega_m: float,
) -> dict:
    """Sort the Floquet modes into forward and backward families by group velocity and label them by basic-mode order.

    Parameters
    ----------
    wavenumbers : np.ndarray
        The wavenumber eigenvalues, shape (4 n_order + 2,).
    mode_shapes : np.ndarray
        Their harmonic mode shapes, shape (2 n_order + 1, 4 n_order + 2).
    coupling : np.ndarray
        Harmonic coupling operator of the modulus.
    omega : float
        Driving angular frequency in radians per second.
    rho0 : float
        Mass density in kilogram per cubic metre.
    kappa_m : float
        Modulation wavenumber in radians per metre.
    omega_m : float
        Modulation angular frequency in radians per second.

    Returns
    -------
    dict
        Under the keys forward_wavenumbers, backward_wavenumbers, forward_modes,
        backward_modes and group_velocities. The wavenumber entries are complex128
        arrays of shape (2 n_order + 1,) indexed by basic-mode order. The mode
        entries are complex128 arrays of shape (2 n_order + 1, 2 n_order + 1). The
        group_velocities entry is a complex128 array of shape (4 n_order + 2,).

    Raises
    ------
    ValueError
        If the shapes are inconsistent, if any input is not finite, if omega or rho0
        is not above zero, if kappa_m or omega_m is zero, or if the group velocity
        does not split the spectrum into two families of equal size.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_directional_mode_families(
    wavenumbers: np.ndarray,
    mode_shapes: np.ndarray,
    coupling: np.ndarray,
    omega: float,
    rho0: float,
    kappa_m: float,
    omega_m: float,
) -> dict:
    wavenumbers = np.asarray(wavenumbers, dtype=np.complex128)
    mode_shapes = np.asarray(mode_shapes, dtype=np.complex128)
    coupling = np.asarray(coupling, dtype=np.float64)
    omega = float(omega)
    rho0 = float(rho0)
    kappa_m = float(kappa_m)
    omega_m = float(omega_m)

    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1] or coupling.shape[0] % 2 != 1:
        raise ValueError("coupling must be square with odd side length")
    side = coupling.shape[0]
    n_order = (side - 1) // 2
    if wavenumbers.ndim != 1 or wavenumbers.size != 2 * side:
        raise ValueError("wavenumbers must hold twice the side length of coupling")
    if mode_shapes.shape != (side, 2 * side):
        raise ValueError("mode_shapes must have shape (side, twice side)")
    if not (np.all(np.isfinite(wavenumbers)) and np.all(np.isfinite(mode_shapes))
            and np.all(np.isfinite(coupling))):
        raise ValueError("wavenumbers, mode_shapes and coupling must be finite")
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and above zero")
    if not np.isfinite(rho0) or rho0 <= 0.0:
        raise ValueError("rho0 must be finite and above zero")
    if not np.isfinite(kappa_m) or kappa_m == 0.0:
        raise ValueError("kappa_m must be finite and not zero")
    if not np.isfinite(omega_m) or omega_m == 0.0:
        raise ValueError("omega_m must be finite and not zero")

    orders = np.arange(-n_order, n_order + 1, dtype=np.float64)
    frequencies = omega + orders * omega_m
    group = np.zeros(2 * side, dtype=np.complex128)
    for j in range(2 * side):
        mode = mode_shapes[:, j]
        shifted = wavenumbers[j] + orders * kappa_m
        numerator = mode @ (shifted * (coupling @ mode))
        denominator = rho0 * (mode @ (frequencies * mode))
        if denominator == 0.0:
            raise ValueError("a mode has vanishing inertia projection; the group velocity is undefined")
        group[j] = numerator / denominator

    forward = np.flatnonzero(group.real > 0.0)
    backward = np.flatnonzero(group.real < 0.0)
    if forward.size != side or backward.size != side:
        raise ValueError("the group velocity did not split the spectrum into two equal families")

    forward = forward[np.argsort(wavenumbers[forward].real, kind="stable")]
    backward = backward[np.argsort(-wavenumbers[backward].real, kind="stable")]
    return {
        "forward_wavenumbers": np.ascontiguousarray(wavenumbers[forward]),
        "backward_wavenumbers": np.ascontiguousarray(wavenumbers[backward]),
        "forward_modes": np.ascontiguousarray(mode_shapes[:, forward]),
        "backward_modes": np.ascontiguousarray(mode_shapes[:, backward]),
        "group_velocities": group,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nimport scipy.linalg as sla\ndef build(n, am, w, r, km, wm):\n    o = np.arange(-n, n + 1, dtype=float)\n    C = np.zeros((2 * n + 1, 2 * n + 1))\n    d = o[:, None] - o[None, :]\n    C[d == 0] = 1.0\n    C[np.abs(d) == 1] = am / 2.0\n    S = np.diag(o * km)\n    W = np.diag((w + o * wm) ** 2)\n    A2, A1, A0 = C, S @ C + C @ S, S @ C @ S - r * W\n    m = 2 * n + 1\n    L = np.block([[np.zeros((m, m)), np.eye(m)], [-A0, -A1]])\n    R = np.block([[np.eye(m), np.zeros((m, m))], [np.zeros((m, m)), A2]])\n    v, x = sla.eig(L, R)\n    x = x[:m, :] / np.linalg.norm(x[:m, :], axis=0)\n    k = np.lexsort((v.imag, v.real))\n    return np.asarray(v[k], dtype=complex), np.ascontiguousarray(x[:, k]), C\nV, X, C = build(3, 0.3, 15.0, 1.0, 10.0, 20.0)\ndef invariant_modes(modes):\n    modes = np.asarray(modes, dtype=complex)\n    pieces = [np.asarray([modes.ndim] + list(modes.shape), dtype=float)]\n    if modes.ndim == 2:\n        norms = np.linalg.norm(modes, axis=0)\n        pieces.append(norms)\n        for mode, norm in zip(modes.T, norms):\n            unit = mode / norm if np.isfinite(norm) and norm > 0.0 else np.full(mode.shape, np.nan + 0j)\n            projector = np.outer(unit, np.conjugate(unit))\n            pieces.extend((projector.real.ravel(), projector.imag.ravel()))\n    return np.concatenate(pieces)\ndef pack(d):\n    forward = np.asarray(d['forward_wavenumbers'], dtype=complex)\n    backward = np.asarray(d['backward_wavenumbers'], dtype=complex)\n    group = np.asarray(d['group_velocities'], dtype=complex)\n    header = np.asarray([forward.ndim, forward.size, backward.ndim, backward.size, group.ndim, group.size], dtype=float)\n    return np.concatenate((header, forward.real.ravel(), forward.imag.ravel(), backward.real.ravel(), backward.imag.ravel(), group.real.ravel(), group.imag.ravel(), invariant_modes(d['forward_modes']), invariant_modes(d['backward_modes'])))\n",
            "call": 'pack(directional_mode_families(np.array(V, copy=True),np.array(X, copy=True),np.array(C, copy=True), 15.0, 1.0, 10.0, 20.0))',
            "gold_call": 'pack(_oracle_directional_mode_families(np.array(V, copy=True),np.array(X, copy=True),np.array(C, copy=True), 15.0, 1.0, 10.0, 20.0))',
        },
        {
            "setup": "import numpy as np\nimport scipy.linalg as sla\ndef build(n, am, w, r, km, wm):\n    o = np.arange(-n, n + 1, dtype=float)\n    C = np.zeros((2 * n + 1, 2 * n + 1))\n    d = o[:, None] - o[None, :]\n    C[d == 0] = 1.0\n    C[np.abs(d) == 1] = am / 2.0\n    S = np.diag(o * km)\n    W = np.diag((w + o * wm) ** 2)\n    A2, A1, A0 = C, S @ C + C @ S, S @ C @ S - r * W\n    m = 2 * n + 1\n    L = np.block([[np.zeros((m, m)), np.eye(m)], [-A0, -A1]])\n    R = np.block([[np.eye(m), np.zeros((m, m))], [np.zeros((m, m)), A2]])\n    v, x = sla.eig(L, R)\n    x = x[:m, :] / np.linalg.norm(x[:m, :], axis=0)\n    k = np.lexsort((v.imag, v.real))\n    return np.asarray(v[k], dtype=complex), np.ascontiguousarray(x[:, k]), C\nV, X, C = build(2, 0.3, 15.0, 1.0, -10.0, 20.0)\ndef invariant_modes(modes):\n    modes = np.asarray(modes, dtype=complex)\n    pieces = [np.asarray([modes.ndim] + list(modes.shape), dtype=float)]\n    if modes.ndim == 2:\n        norms = np.linalg.norm(modes, axis=0)\n        pieces.append(norms)\n        for mode, norm in zip(modes.T, norms):\n            unit = mode / norm if np.isfinite(norm) and norm > 0.0 else np.full(mode.shape, np.nan + 0j)\n            projector = np.outer(unit, np.conjugate(unit))\n            pieces.extend((projector.real.ravel(), projector.imag.ravel()))\n    return np.concatenate(pieces)\ndef pack(d):\n    forward = np.asarray(d['forward_wavenumbers'], dtype=complex)\n    backward = np.asarray(d['backward_wavenumbers'], dtype=complex)\n    group = np.asarray(d['group_velocities'], dtype=complex)\n    header = np.asarray([forward.ndim, forward.size, backward.ndim, backward.size, group.ndim, group.size], dtype=float)\n    return np.concatenate((header, forward.real.ravel(), forward.imag.ravel(), backward.real.ravel(), backward.imag.ravel(), group.real.ravel(), group.imag.ravel(), invariant_modes(d['forward_modes']), invariant_modes(d['backward_modes'])))\n",
            "call": 'pack(directional_mode_families(np.array(V, copy=True),np.array(X, copy=True),np.array(C, copy=True), 15.0, 1.0, -10.0, 20.0))',
            "gold_call": 'pack(_oracle_directional_mode_families(np.array(V, copy=True),np.array(X, copy=True),np.array(C, copy=True), 15.0, 1.0, -10.0, 20.0))',
        },
        {
            "setup": "import numpy as np\nC = np.eye(1)\nV = np.array([4.0 + 0j, -4.0 + 0j])\nX = np.array([[1.0 + 0j, 1.0 + 0j]])\ndef invariant_modes(modes):\n    modes = np.asarray(modes, dtype=complex)\n    pieces = [np.asarray([modes.ndim] + list(modes.shape), dtype=float)]\n    if modes.ndim == 2:\n        norms = np.linalg.norm(modes, axis=0)\n        pieces.append(norms)\n        for mode, norm in zip(modes.T, norms):\n            unit = mode / norm if np.isfinite(norm) and norm > 0.0 else np.full(mode.shape, np.nan + 0j)\n            projector = np.outer(unit, np.conjugate(unit))\n            pieces.extend((projector.real.ravel(), projector.imag.ravel()))\n    return np.concatenate(pieces)\ndef pack(d):\n    forward = np.asarray(d['forward_wavenumbers'], dtype=complex)\n    backward = np.asarray(d['backward_wavenumbers'], dtype=complex)\n    group = np.asarray(d['group_velocities'], dtype=complex)\n    header = np.asarray([forward.ndim, forward.size, backward.ndim, backward.size, group.ndim, group.size], dtype=float)\n    return np.concatenate((header, forward.real.ravel(), forward.imag.ravel(), backward.real.ravel(), backward.imag.ravel(), group.real.ravel(), group.imag.ravel(), invariant_modes(d['forward_modes']), invariant_modes(d['backward_modes'])))\n",
            "call": 'pack(directional_mode_families(np.array(V, copy=True),np.array(X, copy=True),np.array(C, copy=True), 4.0, 1.0, 1.0, 3.0))',
            "gold_call": 'pack(_oracle_directional_mode_families(np.array(V, copy=True),np.array(X, copy=True),np.array(C, copy=True), 4.0, 1.0, 1.0, 3.0))',
        },
        {
            "setup": "import numpy as np\nC = np.eye(1)\nV = np.array([4.0 + 0j, -4.0 + 0j])\nX = np.array([[1.0 + 0j, 1.0 + 0j]])\nVBAD = np.array([4.0 + 0j, 5.0 + 0j, -4.0 + 0j])\nSAME = np.array([4.0 + 0j, 4.0 + 0j])\nNAN = np.array([4.0 + 0j, float('nan') + 0j])\ndef verdict(fn, v=V, x=X, c=C, w=4.0, r=1.0, km=1.0, wm=3.0):\n    try:\n        fn(np.array(v, copy=True), np.array(x, copy=True), np.array(c, copy=True), w, r, km, wm)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": '(verdict(directional_mode_families, v=VBAD), verdict(directional_mode_families, v=SAME), verdict(directional_mode_families, v=NAN), verdict(directional_mode_families, w=0.0), verdict(directional_mode_families, km=0.0), verdict(directional_mode_families))',
            "gold_call": '(verdict(_oracle_directional_mode_families, v=VBAD), verdict(_oracle_directional_mode_families, v=SAME), verdict(_oracle_directional_mode_families, v=NAN), verdict(_oracle_directional_mode_families, w=0.0), verdict(_oracle_directional_mode_families, km=0.0), verdict(_oracle_directional_mode_families))',
        },
    ]
