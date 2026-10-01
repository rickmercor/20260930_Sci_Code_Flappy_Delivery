"""
Differentiate an anchored retained-band frame through spectral projection and mass weighting.

The reference normal modes solve the mass-weighted problem

$$D(\lambda)=S(\lambda)H_*(\lambda)S(\lambda),\qquad S=M^{-1/2},$$

with $m_i(\lambda)=m_i\exp(\mu_i\lambda+\nu_i\lambda^2/2)$. Hence

$$S_i'= -\tfrac12\mu_iS_i,\qquad S_i''=(\tfrac14\mu_i^2-\tfrac12\nu_i)S_i.$$

Select the closed frequency band at $\lambda=0$ and continue its invariant subspace locally. Internal repeated eigenvalues are permitted; retained and excluded eigenvalues must be separated. The spectral projector obeys

$$P^2=P,\quad DP=PD,\quad P^T=P.$$

Its derivatives satisfy the differentiated commutator equations

$$[D_0,P_1]=[P_0,D_1],\qquad [D_0,P_2]=[P_0,D_2]+2[P_1,D_1],$$

and the differentiated idempotency equations, which determine the diagonal subspace blocks without divisions by internal eigenvalue gaps.



For a fixed full-rank anchor $F$, define the locally smooth frame

$$X=PF,\qquad W=X(X^TX)^{-1/2},\qquad B=SW,\qquad K=B^TH_*B.$$

The principal positive square root of $X^TX$ fixes the gauge. This anchored frame generally mixes the retained eigenmodes, so $K$ is dense even though it represents the same harmonic subspace. For a positive matrix square root $Z^2=G$, its response satisfies

$$ZZ'+Z'Z=G',\qquad ZZ''+Z''Z=G''-2(Z')^2.$$

These Sylvester equations remain well defined for repeated positive eigenvalues. The requested frame and stiffness responses include projector motion, gauge normalization, and mass variation; differentiating arbitrary signed eigenvectors independently is insufficient at internal degeneracy.

Returns
-------
np.ndarray, shape (3,k+s,k), anchored retained stiffness and Cartesian reconstruction jets.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def anchored_band_response(
    eq: "np.ndarray", mass: "np.ndarray", anchor: "np.ndarray", lo: float, hi: float
) -> "np.ndarray":
    r"""Differentiate an anchored retained-band frame through spectral projection and mass weighting.

    Parameters
    ----------
    eq : np.ndarray
        Shape (3,s,s+1), reference geometry jets in column 0 and Hessian jets in 1:.
    mass : np.ndarray
        Shape (3,s), rows m, mu, nu define m_i(lambda)=m_i*exp(mu_i*lambda+
        nu_i*lambda^2/2), with m_i>0; rows 1 and 2 are logarithmic derivatives.
    anchor : np.ndarray
        Shape (s,k), fixed anchor with full-rank projection on the selected band.
    lo, hi : float
        Positive ordered angular-frequency endpoints. Apply an absolute 1e-12
        padding to squared-frequency comparisons. Local band membership is fixed.

    Returns
    -------
    result, np.ndarray
        Shape (3,k+s,k), first k rows in each layer are dense retained stiffness K;
        remaining s rows are the Cartesian reconstruction B. The frame is
        W=P*anchor*(anchor.T*P*anchor)^(-1/2), with the principal positive root.
        Repeated eigenvalues within either subspace are allowed.

    Raises
    ------
    ValueError
        If band dimension differs from k, a retained/excluded squared-frequency
        gap is <1e-9, or the projected anchor Gram matrix is not positive definite.

    All derivative arrays have leading axis (value, first derivative, second derivative) at lambda=0, with no factorial scaling. All data are real, finite, and conforming; inputs must be preserved.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_sylvester


def _sqrt_jet(a):
    e, U = np.linalg.eigh(a[0])
    if e[0] <= 0:
        raise ValueError("positive definite square-root argument required")
    S = (U * np.sqrt(e)) @ U.T
    S1 = solve_sylvester(S, S, a[1])
    S2 = solve_sylvester(S, S, a[2] - 2 * S1 @ S1)
    return np.stack((S, S1, S2))


def _oracle_anchored_band_response(
    eq: "np.ndarray", mass: "np.ndarray", anchor: "np.ndarray", lo: float, hi: float
) -> "np.ndarray":
    s, k = anchor.shape
    inv = mass[0] ** -0.5
    sj = np.stack(
        [
            np.diag(inv),
            np.diag(-0.5 * mass[1] * inv),
            np.diag((0.25 * mass[1] ** 2 - 0.5 * mass[2]) * inv),
        ]
    )
    H = eq[:, :, 1:]
    D = _jmat(sj, _jmat(H, sj))
    ev, U = np.linalg.eigh(D[0])
    selected = (ev >= lo**2 - 1e-12) & (ev <= hi**2 + 1e-12)
    if selected.sum() != k:
        raise ValueError("anchor width must equal retained band dimension")
    p = selected.astype(float)
    D1 = U.T @ D[1] @ U
    D2 = U.T @ D[2] @ U
    cross = selected[:, None] != selected[None, :]
    gaps = ev[:, None] - ev[None, :]
    if np.any(np.abs(gaps[cross]) < 1e-9):
        raise ValueError("retained and excluded spectra must be separated")
    P0 = np.diag(p)
    P1 = np.zeros((s, s))
    P1[cross] = ((p[:, None] - p[None, :]) * D1)[cross] / gaps[cross]
    square = P1 @ P1
    P2 = np.zeros_like(P1)
    aa = selected[:, None] & selected[None, :]
    oo = ~selected[:, None] & ~selected[None, :]
    P2[aa] = -2 * square[aa]
    P2[oo] = 2 * square[oo]
    rhs = (p[:, None] - p[None, :]) * D2 + 2 * (P1 @ D1 - D1 @ P1)
    P2[cross] = rhs[cross] / gaps[cross]
    P = np.stack([U @ x @ U.T for x in (P0, P1, P2)])
    X = P @ anchor
    gram = _jmat(_jtranspose(X), X)
    W = _jmat(X, _jinverse(_sqrt_jet(gram)))
    B = _jmat(sj, W)
    K = _jmat(_jtranspose(B), _jmat(H, B))
    return np.concatenate((K, B), axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return coupled, boundary and response-stress cases."""
    return [
        {
            "setup": """import numpy as np
eq = np.array([[[0.06314992037127763, 1.6878355172209871, 2.4021050881974677, -1.184838870634317, 0.702157807945169, 0.29438960532202957], [-0.02075086114846333, 2.4021050881974677, 7.352403960607851, -2.6369942145676304, 1.347182935060397, 1.6859377406850597], [0.014284773049984536, -1.184838870634317, -2.636994214567631, 6.007531764165246, -0.32631635062594916, -2.1411539301750984], [-0.007784377219895726, 0.7021578079451689, 1.347182935060397, -0.32631635062594916, 12.227341438544611, 5.2355881172574215], [0.013004281330919812, 0.2943896053220296, 1.68593774068506, -2.1411539301750984, 5.2355881172574215, 8.40885198008918]], [[0.010433941074218422, 0.1001449946012693, 0.024181934391834803, -0.038213080804323236, -0.0035738383913535327, 0.04790076738445763], [-0.0029618404446127736, 0.024181934391834803, -0.02623357275932696, 0.039371259573977004, 0.041786919195676765, -0.0035738383913535327], [-0.0019528888274162578, -0.038213080804323236, 0.039371259573977004, 0.07935032370362603, -0.04006177267436061, 0.041786919195676765], [0.002372291726142509, -0.0035738383913535327, 0.041786919195676765, -0.04006177267436061, 0.03978283776446919, -0.042401443883623734], [-0.002991366652922518, 0.04790076738445763, -0.0035738383913535327, 0.041786919195676765, -0.042401443883623734, -0.07327162712327218]], [[-0.01736729591483845, -0.005595929586614149, -0.0037428704541024784, -0.0010852722515096438, 0.01717054450301929, -0.011509344543735917], [0.008095816727858862, -0.0037428704541024784, 0.017623253693740404, 0.01774752513762654, -0.0010852722515096438, -0.012829455496980712], [0.00222802721798731, -0.0010852722515096438, 0.01774752513762654, -0.016867100488597422, 0.014933529539603115, 0.013914727748490355], [-0.0009643308202315427, 0.01717054450301929, -0.0010852722515096438, 0.014933529539603115, 0.03006966009208716, 0.0016247187455741884], [0.0012534347520972871, -0.011509344543735917, -0.012829455496980712, 0.013914727748490355, 0.0016247187455741884, 0.004036481213709182]]], dtype=float)
mass = np.array([[1.0, 1.6, 2.3, 3.1, 4.2], [0.12, -0.08, 0.1, 0.04, -0.06], [0.03, -0.02, 0.025, -0.01, 0.015]], dtype=float)
anchor = np.array([[0.7727272727272727, -0.17272727272727273, 0.009090909090909094], [-0.35454545454545455, 0.2545454545454545, 0.4681818181818182], [0.1772727272727273, 0.5227272727272727, 0.8909090909090909], [-0.2272727272727273, -0.37272727272727274, 0.1090909090909091], [0.4545454545454546, 0.7454545454545455, -0.2181818181818182]], dtype=float)
lo = 0.6
hi = 1.8

def preserved(fn, *args):
    snapshots = [(x, x.copy()) for x in args if isinstance(x, np.ndarray)]
    result = fn(*args)
    for value, original in snapshots:
        assert np.array_equal(value, original), "inputs must be preserved"
    return result
""",
            "call": "preserved(anchored_band_response, eq.copy(), mass.copy(), anchor.copy(), lo, hi)",
            "gold_call": "preserved(_oracle_anchored_band_response, eq.copy(), mass.copy(), anchor.copy(), lo, hi)",
            "tol": 2e-09,
        },
        {
            "setup": """import numpy as np
eq = np.array([[[0.0, 1.8896000000000004, -3.053569923875988, 0.7648736366224166, -0.4415999999999999], [0.0, -3.053569923875988, 10.256800000000004, -2.1633893408261042, 1.2490334182879173], [0.0, 0.7648736366224166, -2.1633893408261042, 11.410800000000002, -5.1850672975381915], [0.0, -0.4415999999999998, 1.249033418287917, -5.1850672975381915, 6.233600000000001]], [[0.0, 0.04, 0.04, -0.04, 0.0], [0.0, 0.04, -0.04, 0.04, 0.04], [0.0, -0.04, 0.04, 0.08, -0.04], [0.0, 0.0, 0.04, -0.04, 0.04]], [[0.0, 0.04, -0.02, 0.0, 0.02], [0.0, -0.02, 0.02, 0.02, 0.0], [0.0, 0.0, 0.02, -0.02, 0.02], [0.0, 0.02, 0.0, 0.02, 0.04]]], dtype=float)
mass = np.array([[1.0, 2.0, 3.0, 4.0], [0.12, -0.08, 0.09, 0.04], [0.02, 0.03, -0.01, 0.025]], dtype=float)
anchor = np.array([[0.76, 0.56], [0.38, 0.27999999999999997], [-0.24000000000000002, 0.36], [-0.48000000000000004, 0.72]], dtype=float)
lo = 0.7
hi = 1.2
""",
            "call": "anchored_band_response(eq.copy(), mass.copy(), anchor.copy(), lo, hi)",
            "gold_call": "_oracle_anchored_band_response(eq.copy(), mass.copy(), anchor.copy(), lo, hi)",
            "tol": 2e-09,
        },
        {
            "setup": """import numpy as np
eq = np.array([[[0.0, 1.8896000000000004, -3.053569923875988, 0.7648736366224166, -0.4415999999999999], [0.0, -3.053569923875988, 10.256800000000004, -2.1633893408261042, 1.2490334182879173], [0.0, 0.7648736366224166, -2.1633893408261042, 11.410800000000002, -5.1850672975381915], [0.0, -0.4415999999999998, 1.249033418287917, -5.1850672975381915, 6.233600000000001]], [[0.0, 0.04, 0.04, -0.04, 0.0], [0.0, 0.04, -0.04, 0.04, 0.04], [0.0, -0.04, 0.04, 0.08, -0.04], [0.0, 0.0, 0.04, -0.04, 0.04]], [[0.0, 0.04, -0.02, 0.0, 0.02], [0.0, -0.02, 0.02, 0.02, 0.0], [0.0, 0.0, 0.02, -0.02, 0.02], [0.0, 0.02, 0.0, 0.02, 0.04]]], dtype=float)
mass = np.array([[1.0, 2.0, 3.0, 4.0], [0.12, -0.08, 0.09, 0.04], [0.02, 0.03, -0.01, 0.025]], dtype=float)
anchor = np.array([[0.76, 0.48000000000000004, -0.16, -0.42000000000000004], [0.48000000000000004, 0.23999999999999996, 0.42000000000000004, 0.8400000000000001], [-0.24000000000000002, 0.38, 0.8400000000000001, -0.32], [-0.38, 0.76, -0.32, 0.15999999999999995]], dtype=float)
lo = 0.5
hi = 3.0
""",
            "call": "anchored_band_response(eq.copy(), mass.copy(), anchor.copy(), lo, hi)",
            "gold_call": "_oracle_anchored_band_response(eq.copy(), mass.copy(), anchor.copy(), lo, hi)",
            "tol": 2e-09,
        },
    ]
