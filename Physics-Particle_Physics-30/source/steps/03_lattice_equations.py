"""
Evaluate the normalized field-equation residuals and local Newton curvatures for the three-dimensional dual-lattice DGL model using Eqs. (43)–(48) and (51)–(52).



The state array has shape (5,Lx,Ly,Lz), with channels (B0,B1,B2,Re chi,Im chi). The source array sigma has shape (3,Lx,Ly,Lz), with plaquettes ordered as (01,02,12). Use periodic boundaries and epsilon_12=+1.



Return eleven channels in this exact order:

(X0,X1,X2,Y1,Y2,dX0,dX1,dX2,H11,H12,H22).



Here X_mu=dS/dB_mu and Y_alpha=(dS/dc_alpha)/mB^2 for beta_g=1. The dX channels are diagonal gauge curvatures, and H is the full symmetric 2-by-2 scalar Newton block.



Require compatible finite arrays, spatial side lengths at least 2, and positive finite masses. Raise ValueError for invalid inputs.

The complex scalar is chi=c1+i*c2, and its covariant difference is chi(x)-exp(i*B_mu(x))*chi(x+mu). The physical noncompact field strength is F=curl(B)-2*pi*sigma.



Equations (47)–(48) give the normalized stationary residuals. The gauge residual includes the backward divergence of F and the scalar current. The scalar residual includes both forward and backward parallel transport and the quartic-potential derivative.



Equations (51)–(52) determine the local Newton curvatures. The scalar block includes its off-diagonal term mchi^2*c1*c2; treating the two scalar components as independent diagonal updates would implement a different method.

Returns
-------
A real NumPy array of shape (11,Lx,Ly,Lz), with channels ordered as (X0,X1,X2,Y1,Y2,dX0,dX1,dX2,H11,H12,H22).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def lattice_equations(
    state: np.ndarray,
    sigma: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
) -> np.ndarray:
    """Evaluate normalized residuals and local Newton curvatures.

    Parameters
    ----------
    state : numpy.ndarray
        Finite real array (5,Lx,Ly,Lz), channels B0,B1,B2,c1,c2.
        Each spatial side is at least 2. The input is not modified.
    sigma : numpy.ndarray
        Finite real source (3,Lx,Ly,Lz), plaquettes (01,02,12).
        Spatial dimensions must match state.
    mB : float, default 0.5
        Positive finite gauge mass.
    mchi : float, default 0.5
        Positive finite scalar mass; need not equal mB in this step.

    Returns
    -------
    numpy.ndarray
        Real array (11,Lx,Ly,Lz) with channels
        X0,X1,X2,Y1,Y2,dX0,dX1,dX2,H11,H12,H22.
        X=dS/dB and Y=(dS/dc)/mB**2; H is the normalized scalar block.

    Raises
    ------
    ValueError
        For invalid array shapes, incompatible dimensions, nonfinite
        numeric values or nonpositive masses.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_lattice_equations(
    state: np.ndarray,
    sigma: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
) -> np.ndarray:
    """Return five residual channels and six local curvature channels."""
    q = np.asarray(state, dtype=float)
    s = np.asarray(sigma, dtype=float)

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

    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(s)):
        raise ValueError("state and sigma must contain finite values")

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

    X = np.zeros_like(B)

    for k, (u, v) in enumerate(((0, 1), (0, 2), (1, 2))):
        X[u] += F[k] - np.roll(F[k], 1, axis=v)
        X[v] -= F[k] - np.roll(F[k], 1, axis=u)

    dX = np.empty_like(B)
    magnitude_squared = np.abs(chi) ** 2

    Y = (
        0.5
        * mchi ** 2
        * chi
        * (magnitude_squared - 1.0)
    )

    for u in range(3):
        link = np.exp(1j * B[u])

        forward_transport = (
            link * np.roll(chi, -1, axis=u)
        )

        backward_transport = np.roll(
            np.conjugate(link) * chi,
            1,
            axis=u,
        )

        overlap = np.conjugate(chi) * forward_transport

        X[u] += mB ** 2 * overlap.imag
        dX[u] = 4.0 + mB ** 2 * overlap.real

        Y += (
            2.0 * chi
            - forward_transport
            - backward_transport
        )

    base = 6.0 + 0.5 * mchi ** 2 * (
        magnitude_squared - 1.0
    )

    H11 = base + mchi ** 2 * q[3] ** 2
    H12 = mchi ** 2 * q[3] * q[4]
    H22 = base + mchi ** 2 * q[4] ** 2

    return np.concatenate(
        (
            X,
            np.stack((Y.real, Y.imag), axis=0),
            dX,
            np.stack((H11, H12, H22), axis=0),
        ),
        axis=0,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 's = np.zeros((3, 4, 4, 4), dtype=float)\n'
               's[0, 0, 0, 1:2] = -1.0\n'
               'q = np.zeros((5, 4, 4, 4), dtype=float)\n'
               'q[3] = 1.0\n'
               'q += 0.03 * np.sin(np.arange(q.size)).reshape(q.shape)',
      'call': 'lattice_equations(q, s, 0.5, 0.5)',
      'gold_call': '_oracle_lattice_equations(q, s, 0.5, 0.5)'},
     {'setup': 's = np.zeros((3, 6, 6, 6), dtype=float)\n'
               's[0, 0, 0, 1:3] = -1.0\n'
               'q = np.zeros((5, 6, 6, 6), dtype=float)\n'
               'q[3] = 1.0\n'
               'q += 0.03 * np.sin(np.arange(q.size)).reshape(q.shape)',
      'call': 'lattice_equations(q, s, 0.4, 0.7)',
      'gold_call': '_oracle_lattice_equations(q, s, 0.4, 0.7)'},
     {'setup': 's = np.zeros((3, 4, 4, 4), dtype=float)\n'
               's[0, 0, 0, 1:2] = -1.0\n'
               'q = np.zeros((5, 4, 4, 4), dtype=float)\n'
               'q[3] = 1.0',
      'call': 'lattice_equations(q, s, 0.6, 0.3)',
      'gold_call': '_oracle_lattice_equations(q, s, 0.6, 0.3)'},
     {'setup': 'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n'
               'q=np.zeros((5,4,4,4), dtype=float)\n'
               'q[3]=1.0\n'
               's=np.zeros((3,4,4,4), dtype=float)\n',
      'call': 'raises_value_error(lattice_equations, q, s, 0.0, 0.5)',
      'gold_call': '1.0',
      'tol': 0.0}]
