"""
Evaluate the full and Hodge-decomposed lattice actions at beta_g=1 using Eqs. (42)–(44) and (55)–(69).







Inputs are the state array with channels (B0,B1,B2,Re chi,Im chi), the Dirac source with plaquettes ordered as (01,02,12), and the matching zero-mean periodic massless Green function G.







Compute the charge density, Coulombic field, and regular Hodge potential using periodic differences and convolution. Include the finite-volume correction Fsole=curl(Breg)-2*pi*xi/V, where xi is the spatial sum of each source component and V is the number of lattice sites.







Evaluate Ssole using the corrected solenoidal gauge field and the original scalar kinetic and potential terms.







Return [S,Scoul,Ssole,max_field_reconstruction_error,energy_reconstruction_error], where the last entry is the signed difference S-Scoul-Ssole. Raise ValueError for incompatible shapes, nonfinite inputs, or nonpositive masses.



Returns

-------

A real NumPy array of length 5 containing [S,Scoul,Ssole,max_field_reconstruction_error,energy_reconstruction_error]. The energy error is the signed quantity S-Scoul-Ssole.

Evaluate the full and Hodge-decomposed lattice actions at beta_g=1 using Eqs. (42)–(44) and (55)–(69).



Inputs are the state array with channels (B0,B1,B2,Re chi,Im chi), the Dirac source with plaquettes ordered as (01,02,12), and the matching zero-mean periodic massless Green function G.



Compute the charge density, Coulombic field, and regular Hodge potential using periodic differences and convolution. Include the finite-volume correction Fsole=curl(Breg)-2*pi*xi/V, where xi is the spatial sum of each source component and V is the number of lattice sites.



Evaluate Ssole using the corrected solenoidal gauge field and the original scalar kinetic and potential terms.



Return [S,Scoul,Ssole,max_field_reconstruction_error,energy_reconstruction_error], where the last entry is the signed difference S-Scoul-Ssole. Raise ValueError for incompatible shapes, nonfinite inputs, or nonpositive masses.

Returns
-------
A real NumPy array of length 5 containing [S,Scoul,Ssole,max_field_reconstruction_error,energy_reconstruction_error]. The energy error is the signed quantity S-Scoul-Ssole.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def hodge_action(
    state: np.ndarray,
    sigma: np.ndarray,
    G: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
) -> np.ndarray:
    """Evaluate the full action, Hodge split and reconstruction errors.

    Parameters
    ----------
    state : numpy.ndarray
        Finite real array (5,Lx,Ly,Lz), channels B0,B1,B2,c1,c2.
        Each spatial side is at least 2.
    sigma : numpy.ndarray
        Finite real source (3,Lx,Ly,Lz), plaquettes (01,02,12),
        with spatial dimensions matching state.
    G : numpy.ndarray
        Finite real array (Lx,Ly,Lz). The caller supplies the
        matching zero-mean periodic Green function; this function
        validates shape and finiteness, not the Green equation.
    mB : float, default 0.5
        Positive finite gauge mass.
    mchi : float, default 0.5
        Positive finite scalar mass.

    Returns
    -------
    numpy.ndarray
        Real array of length 5: S,Scoul,Ssole,max_field_error,energy_error.
        max_field_error=max(abs(F-Fsole-Fcoul)); energy_error=S-Scoul-Ssole
        is signed. Ssole retains both original matter terms and the
        finite-volume field correction. Input arrays are not modified.

    Raises
    ------
    ValueError
        For invalid/incompatible numeric shapes, nonfinite arrays
        or masses, spatial sides below 2 or nonpositive masses.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_hodge_action(
    state: np.ndarray,
    sigma: np.ndarray,
    G: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
) -> np.ndarray:
    """Return the full action, its Hodge split, and reconstruction errors."""
    q = np.asarray(state, dtype=float)
    s = np.asarray(sigma, dtype=float)
    G = np.asarray(G, dtype=float)

    if (
        q.ndim != 4
        or q.shape[0] != 5
        or min(q.shape[1:]) < 2
    ):
        raise ValueError(
            "state must have shape (5,Lx,Ly,Lz), "
            "with each spatial side at least 2"
        )

    if s.shape != (3,) + q.shape[1:]:
        raise ValueError("sigma and state have incompatible shapes")

    if G.shape != q.shape[1:]:
        raise ValueError("G and state have incompatible shapes")

    if (
        not np.all(np.isfinite(q))
        or not np.all(np.isfinite(s))
        or not np.all(np.isfinite(G))
    ):
        raise ValueError("all arrays must contain finite values")

    if (
        not np.isfinite(mB)
        or not np.isfinite(mchi)
        or mB <= 0.0
        or mchi <= 0.0
    ):
        raise ValueError("masses must be positive and finite")

    B = q[:3]
    chi = q[3] + 1j * q[4]

    F = _oracle_lattice_curl(B) - 2.0 * np.pi * s

    charge_density = -(
        np.roll(s[2], -1, axis=0) - s[2]
        - np.roll(s[1], -1, axis=1) + s[1]
        + np.roll(s[0], -1, axis=2) - s[0]
    )

    green_fourier = np.fft.fftn(G)

    psi = np.fft.ifftn(
        green_fourier * np.fft.fftn(charge_density)
    ).real

    C = np.stack(
        (
            -(psi - np.roll(psi, 1, axis=2)),
            psi - np.roll(psi, 1, axis=1),
            -(psi - np.roll(psi, 1, axis=0)),
        ),
        axis=0,
    )

    Fcoul = 2.0 * np.pi * C

    divergence = np.zeros_like(B)

    for k, (u, v) in enumerate(((0, 1), (0, 2), (1, 2))):
        divergence[u] += (
            F[k] - np.roll(F[k], 1, axis=v)
        )
        divergence[v] -= (
            F[k] - np.roll(F[k], 1, axis=u)
        )

    Breg = np.stack(
        [
            np.fft.ifftn(
                green_fourier * np.fft.fftn(component)
            ).real
            for component in divergence
        ],
        axis=0,
    )

    correction = (
        2.0
        * np.pi
        * np.mean(s, axis=(1, 2, 3))
    )[:, None, None, None]

    Fsole = _oracle_lattice_curl(Breg) - correction

    kinetic_sum = 0.0

    for direction in range(3):
        covariant_difference = (
            chi
            - np.exp(1j * B[direction])
            * np.roll(chi, -1, axis=direction)
        )
        kinetic_sum += np.sum(
            np.abs(covariant_difference) ** 2
        )

    matter_action = (
        0.5 * mB ** 2 * kinetic_sum
        + (mB ** 2 * mchi ** 2 / 8.0)
        * np.sum((np.abs(chi) ** 2 - 1.0) ** 2)
    )

    S = 0.5 * np.sum(F ** 2) + matter_action
    Scoul = 0.5 * np.sum(Fcoul ** 2)
    Ssole = 0.5 * np.sum(Fsole ** 2) + matter_action

    field_error = np.max(
        np.abs(F - Fsole - Fcoul)
    )

    energy_error = S - Scoul - Ssole

    return np.array(
        [
            S,
            Scoul,
            Ssole,
            field_error,
            energy_error,
        ],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 's = np.zeros((3, 6, 6, 6), dtype=float)\n'
               's[0, 0, 0, 1:3] = -1.0\n'
               'q = np.zeros((5, 6, 6, 6), dtype=float)\n'
               'q[3] = 1.0\n'
               'q += 0.03 * np.sin(np.arange(q.size)).reshape(q.shape)\n'
               'G = _oracle_periodic_green(6)',
      'call': 'hodge_action(q, s, G, 0.5, 0.5)',
      'gold_call': '_oracle_hodge_action(q, s, G, 0.5, 0.5)'},
     {'setup': 's = np.zeros((3, 8, 8, 8), dtype=float)\n'
               's[0, 0, 0, 1:4] = -1.0\n'
               'q = np.zeros((5, 8, 8, 8), dtype=float)\n'
               'q[3] = 1.0\n'
               'q += 0.03 * np.sin(np.arange(q.size)).reshape(q.shape)\n'
               'G = _oracle_periodic_green(8)',
      'call': 'hodge_action(q, s, G, 0.4, 0.7)',
      'gold_call': '_oracle_hodge_action(q, s, G, 0.4, 0.7)'},
     {'setup': 's = np.zeros((3, 6, 6, 6), dtype=float)\n'
               's[0, 0, 0, 1:2] = -1.0\n'
               'q = np.zeros((5, 6, 6, 6), dtype=float)\n'
               'q[3] = 1.0\n'
               'G = _oracle_periodic_green(6)',
      'call': 'hodge_action(q, s, G, 0.6, 0.3)',
      'gold_call': '_oracle_hodge_action(q, s, G, 0.6, 0.3)'},
     {'setup': 'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n'
               'q=np.zeros((5,4,4,4), dtype=float)\n'
               'q[3]=1.0\n'
               's=np.zeros((3,4,4,4), dtype=float)\n',
      'call': 'raises_value_error(hodge_action, q, s, np.zeros((3,4,4), dtype=float))',
      'gold_call': '1.0',
      'tol': 0.0}]
