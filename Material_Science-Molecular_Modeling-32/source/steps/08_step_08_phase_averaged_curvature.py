"""
Compute phase-averaged stretching curvature using parameter-dependent monodromy conjugacy.

For a periodically driven orbit, finite-time singular-value stretching depends on the phase at which the period begins. Average over all discrete phases to obtain a diagnostic that does not privilege a starting node. Let $P_j(\lambda)$ be the tangent of the first $j$ steps along the parameter-dependent periodic orbit, with $P_0=I$, and let $M_0$ be its full-period monodromy. Periodicity gives the phase-shifted period map

$$M_j=P_jM_0P_j^{-1},\qquad j=0,\ldots,N-1.$$

This is an ordinary similarity, not generally an orthogonal one, so its singular values can depend on phase even though its eigenvalues do not. The first two derivatives of $P_j^{-1}$ and the raw-derivative product rule must be included when differentiating $M_j$. For example,

$$ (XYZ)''=X''YZ+XY''Z+XYZ''+2X'Y'Z+2X'YZ'+2XY'Z'.$$

Each prefix derivative follows the moving periodic initial point and the changing molecular reference; it is not a derivative at a fixed initial trajectory.



The specified observable is

$$\bar\chi(\lambda)=\frac1N\sum_{j=0}^{N-1}

\frac{\log\sigma_{\max}(D(\lambda)M_j(\lambda)D(\lambda)^{-1})}{Nh},\qquad

\mathcal C=\bar\chi''(0).$$

The endpoint phase $j=N$ duplicates the first phase and is excluded. Forcing is sampled as $g_j=g_c(\lambda)\cos(2\pi j/N)+g_s(\lambda)\sin(2\pi j/N)$ with an exactly repeated endpoint. This final orchestrator solves the relaxed reference, constructs its anchored band, obtains the nonlinear periodic branch, accumulates prefix jets, differentiates every phase conjugacy, and averages the moving-metric curvature. The periodic forcing and curvature diagnostic are task-defined extensions of the paper's band dynamics.

Returns
-------
float, second parameter derivative of the phase-averaged finite-period stretching rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def phase_averaged_curvature(
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    mass: "np.ndarray",
    anchor: "np.ndarray",
    lo: float,
    hi: float,
    cosdrive: "np.ndarray",
    sindrive: "np.ndarray",
    h: float,
    n: int,
) -> float:
    r"""Compute phase-averaged stretching curvature using parameter-dependent monodromy conjugacy.

    Parameters
    ----------
    A, C, cubic, beta, force : np.ndarray
        Potential coefficient arrays (3,s,s), (j,s), (j,), (3,j), (3,s).
        A, beta, force define quadratic parameter polynomials using raw derivatives.
    mass : np.ndarray
        Shape (3,s), positive masses and first/second log-mass derivatives as
        m_i(lambda)=mass[0,i]*exp(mass[1,i]*lambda+mass[2,i]*lambda^2/2).
    anchor : np.ndarray
        Shape (s,k), fixed full-rank anchor for the continued retained subspace.
    lo, hi : float
        Positive closed frequency bounds, with 1e-12 padding on squared endpoints.
    cosdrive, sindrive : np.ndarray
        Shape (3,s), Cartesian cosine/sine force-amplitude derivative layers.
    h : float
        Positive fixed timestep.
    n : int
        Number of steps per drive period, n>=2; period T=n*h is fixed in lambda.

    Returns
    -------
    result, float
        Second parameter derivative at zero of the mean over phases j=0,...,n-1
        of log(sigma_max(D*M_j*D^(-1)))/(n*h). Obtain the stationary reference
        from zero (tolerance 1e-13, max 40), the periodic branch using eight
        drive-amplitude continuations (tolerance 1e-12, max 30), and phase
        monodromies by differentiating P_j*M_0*P_j^(-1) along that branch.

    Raises
    ------
    ValueError
        If the relaxed/periodic solves, retained-band/anchor conditions, or
        metric-stretching positive/simple spectral conditions described above fail.

    All derivative arrays have leading axis (value, first derivative, second derivative) at lambda=0, with no factorial scaling. All data are real, finite, and conforming; inputs must be preserved.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_phase_averaged_curvature(
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    mass: "np.ndarray",
    anchor: "np.ndarray",
    lo: float,
    hi: float,
    cosdrive: "np.ndarray",
    sindrive: "np.ndarray",
    h: float,
    n: int,
) -> float:
    eq = _oracle_relaxed_reference_response(
        A, C, cubic, beta, force, np.zeros(A.shape[1]), 1e-13, 40
    )
    ref = _oracle_anchored_band_response(eq, mass, anchor, lo, hi)
    k = anchor.shape[1]
    stiffness = ref[:, :k]
    R = _oracle_harmonic_frechet_response(stiffness, h)
    phase = 2 * np.pi * np.arange(n + 1) / n
    drives = (
        np.cos(phase)[:, None, None] * cosdrive
        + np.sin(phase)[:, None, None] * sindrive
    )
    drives[-1] = drives[0]
    orbit = _oracle_periodic_orbit_response(
        eq, ref, R, A, C, cubic, beta, force, drives, h, 1e-12, 30
    )
    prefix = np.zeros_like(orbit)
    prefix[:, :, 0] = orbit[:, :, 0]
    prefix[0, :, 1:] = np.eye(2 * k)
    total = 0.0
    for j in range(n):
        P = prefix[:, :, 1:]
        shifted = _jmat(P, _jmat(orbit[:, :, 1:], _jinverse(P)))
        metrics = _oracle_metric_stretching_curvature(shifted, stiffness, n * h)
        total += metrics[3]
        prefix = _oracle_driven_band_jet_step(
            prefix, eq, ref, R, A, C, cubic, beta, force, drives[j], drives[j + 1], h
        )
    return float(total / n)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return coupled, boundary and response-stress cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[[1.6246487603305786, 2.4386649010382797, -1.1861873256612212, 0.7048547179989777, 0.2650657512186243], [2.4386649010382797, 7.305024793388431, -2.6241521776486656, 1.3458344800334927, 1.6886346507388683], [-1.1861873256612212, -2.624152177648666, 5.9906921487603295, -0.3213074308698191, -2.1425023852020026], [0.7048547179989776, 1.3458344800334927, -0.3213074308698191, 12.217138429752065, 5.242804888834193], [0.26506575121862436, 1.6886346507388685, -2.1425023852020026, 5.242804888834193, 8.375008264462812]], [[0.08, 0.04, -0.04, 0.0, 0.04], [0.04, -0.04, 0.04, 0.04, 0.0], [-0.04, 0.04, 0.08, -0.04, 0.04], [0.0, 0.04, -0.04, 0.04, -0.04], [0.04, 0.0, 0.04, -0.04, -0.08]], [[0.015, -0.015, 0.0, 0.015, 0.0], [-0.015, 0.03, 0.015, 0.0, -0.015], [0.0, 0.015, -0.015, 0.015, 0.015], [0.015, 0.0, 0.015, 0.03, 0.0], [0.0, -0.015, 0.015, 0.0, 0.015]]], dtype=float)
C = np.array([[1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0], [1.0, 0.0, 0.0, 0.0, 1.0], [1.0, -0.5, 0.25, -0.5, 1.0]], dtype=float)
cubic = np.array([0.12, -0.15, 0.08, -0.09, 0.1, -0.07], dtype=float)
beta = np.array([[0.65, 0.45, 0.55, 0.6, 0.5, 0.7], [0.22, -0.18, 0.12, 0.15, -0.14, 0.2], [0.08, 0.05, -0.04, 0.06, 0.03, -0.05]], dtype=float)
force = np.array([[0.035, -0.025, 0.04, -0.015, 0.02], [0.018, 0.01, -0.012, 0.015, -0.008], [-0.01, 0.012, 0.008, -0.006, 0.011]], dtype=float)
mass = np.array([[1.0, 1.6, 2.3, 3.1, 4.2], [0.12, -0.08, 0.1, 0.04, -0.06], [0.03, -0.02, 0.025, -0.01, 0.015]], dtype=float)
anchor = np.array([[0.7727272727272727, -0.17272727272727273, 0.009090909090909094], [-0.35454545454545455, 0.2545454545454545, 0.4681818181818182], [0.1772727272727273, 0.5227272727272727, 0.8909090909090909], [-0.2272727272727273, -0.37272727272727274, 0.1090909090909091], [0.4545454545454546, 0.7454545454545455, -0.2181818181818182]], dtype=float)
lo = 0.6
hi = 1.8
cosdrive = np.array([[0.018, -0.025, 0.022, 0.012, -0.016], [0.008, 0.006, -0.005, 0.007, -0.004], [-0.003, 0.004, 0.005, -0.002, 0.003]], dtype=float)
sindrive = np.array([[-0.014, 0.01, 0.016, -0.02, 0.012], [0.005, -0.004, 0.007, 0.003, -0.006], [0.002, 0.003, -0.004, 0.005, -0.001]], dtype=float)
h = 0.16
n = 24

def preserved(fn, *args):
    snapshots = [(x, x.copy()) for x in args if isinstance(x, np.ndarray)]
    result = fn(*args)
    for value, original in snapshots:
        assert np.array_equal(value, original), "inputs must be preserved"
    return result
""",
            "call": "preserved(phase_averaged_curvature, A.copy(), C.copy(), cubic.copy(), beta.copy(), force.copy(), mass.copy(), anchor.copy(), lo, hi, cosdrive.copy(), sindrive.copy(), h, n)",
            "gold_call": "preserved(_oracle_phase_averaged_curvature, A.copy(), C.copy(), cubic.copy(), beta.copy(), force.copy(), mass.copy(), anchor.copy(), lo, hi, cosdrive.copy(), sindrive.copy(), h, n)",
            "tol": 2e-08,
        },
        {
            "setup": """import numpy as np
A = np.array([[[1.6246487603305786, 2.4386649010382797, -1.1861873256612212, 0.7048547179989777, 0.2650657512186243], [2.4386649010382797, 7.305024793388431, -2.6241521776486656, 1.3458344800334927, 1.6886346507388683], [-1.1861873256612212, -2.624152177648666, 5.9906921487603295, -0.3213074308698191, -2.1425023852020026], [0.7048547179989776, 1.3458344800334927, -0.3213074308698191, 12.217138429752065, 5.242804888834193], [0.26506575121862436, 1.6886346507388685, -2.1425023852020026, 5.242804888834193, 8.375008264462812]], [[0.08, 0.04, -0.04, 0.0, 0.04], [0.04, -0.04, 0.04, 0.04, 0.0], [-0.04, 0.04, 0.08, -0.04, 0.04], [0.0, 0.04, -0.04, 0.04, -0.04], [0.04, 0.0, 0.04, -0.04, -0.08]], [[0.015, -0.015, 0.0, 0.015, 0.0], [-0.015, 0.03, 0.015, 0.0, -0.015], [0.0, 0.015, -0.015, 0.015, 0.015], [0.015, 0.0, 0.015, 0.03, 0.0], [0.0, -0.015, 0.015, 0.0, 0.015]]], dtype=float)
C = np.array([[1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0], [1.0, 0.0, 0.0, 0.0, 1.0], [1.0, -0.5, 0.25, -0.5, 1.0]], dtype=float)
cubic = np.array([0.204, -0.255, 0.136, -0.153, 0.17, -0.11900000000000001], dtype=float)
beta = np.array([[0.65, 0.45, 0.55, 0.6, 0.5, 0.7], [-0.154, 0.126, -0.08399999999999999, -0.105, 0.098, -0.13999999999999999], [-0.055999999999999994, -0.034999999999999996, 0.027999999999999997, -0.041999999999999996, -0.020999999999999998, 0.034999999999999996]], dtype=float)
force = np.array([[0.049, -0.034999999999999996, 0.055999999999999994, -0.020999999999999998, 0.027999999999999997], [0.025199999999999997, 0.013999999999999999, -0.0168, 0.020999999999999998, -0.0112], [-0.013999999999999999, 0.0168, 0.0112, -0.0084, 0.015399999999999999]], dtype=float)
mass = np.array([[1.0, 1.6, 2.3, 3.1, 4.2], [0.156, -0.10400000000000001, 0.13, 0.052000000000000005, -0.078], [0.039, -0.026000000000000002, 0.0325, -0.013000000000000001, 0.0195]], dtype=float)
anchor = np.array([[0.7727272727272727, -0.17272727272727273, 0.009090909090909094], [-0.35454545454545455, 0.2545454545454545, 0.4681818181818182], [0.1772727272727273, 0.5227272727272727, 0.8909090909090909], [-0.2272727272727273, -0.37272727272727274, 0.1090909090909091], [0.4545454545454546, 0.7454545454545455, -0.2181818181818182]], dtype=float)
lo = 0.6
hi = 1.8
cosdrive = np.array([[0.012599999999999998, -0.017499999999999998, 0.015399999999999999, 0.0084, -0.0112], [0.0056, 0.0042, -0.0034999999999999996, 0.0049, -0.0028], [-0.0021, 0.0028, 0.0034999999999999996, -0.0014, 0.0021]], dtype=float)
sindrive = np.array([[0.011200000000000002, -0.008, -0.0128, 0.016, -0.009600000000000001], [-0.004, 0.0032, -0.005600000000000001, -0.0024000000000000002, 0.0048000000000000004], [-0.0016, -0.0024000000000000002, 0.0032, -0.004, 0.0008]], dtype=float)
h = 0.21
n = 18
""",
            "call": "phase_averaged_curvature(A.copy(), C.copy(), cubic.copy(), beta.copy(), force.copy(), mass.copy(), anchor.copy(), lo, hi, cosdrive.copy(), sindrive.copy(), h, n)",
            "gold_call": "_oracle_phase_averaged_curvature(A.copy(), C.copy(), cubic.copy(), beta.copy(), force.copy(), mass.copy(), anchor.copy(), lo, hi, cosdrive.copy(), sindrive.copy(), h, n)",
            "tol": 2e-08,
        },
        {
            "setup": """import numpy as np
A = np.array([[[1.6246487603305786, 2.4386649010382797, -1.1861873256612212, 0.7048547179989777, 0.2650657512186243], [2.4386649010382797, 7.305024793388431, -2.6241521776486656, 1.3458344800334927, 1.6886346507388683], [-1.1861873256612212, -2.624152177648666, 5.9906921487603295, -0.3213074308698191, -2.1425023852020026], [0.7048547179989776, 1.3458344800334927, -0.3213074308698191, 12.217138429752065, 5.242804888834193], [0.26506575121862436, 1.6886346507388685, -2.1425023852020026, 5.242804888834193, 8.375008264462812]], [[0.08, 0.04, -0.04, 0.0, 0.04], [0.04, -0.04, 0.04, 0.04, 0.0], [-0.04, 0.04, 0.08, -0.04, 0.04], [0.0, 0.04, -0.04, 0.04, -0.04], [0.04, 0.0, 0.04, -0.04, -0.08]], [[0.015, -0.015, 0.0, 0.015, 0.0], [-0.015, 0.03, 0.015, 0.0, -0.015], [0.0, 0.015, -0.015, 0.015, 0.015], [0.015, 0.0, 0.015, 0.03, 0.0], [0.0, -0.015, 0.015, 0.0, 0.015]]], dtype=float)
C = np.array([[1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0], [1.0, 0.0, 0.0, 0.0, 1.0], [1.0, -0.5, 0.25, -0.5, 1.0]], dtype=float)
cubic = np.array([0.06, -0.075, 0.04, -0.045, 0.05, -0.035], dtype=float)
beta = np.array([[0.65, 0.45, 0.55, 0.6, 0.5, 0.7], [0.22, -0.18, 0.12, 0.15, -0.14, 0.2], [0.08, 0.05, -0.04, 0.06, 0.03, -0.05]], dtype=float)
force = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.018, 0.01, -0.012, 0.015, -0.008], [-0.01, 0.012, 0.008, -0.006, 0.011]], dtype=float)
mass = np.array([[1.0, 1.6, 2.3, 3.1, 4.2], [0.12, -0.08, 0.1, 0.04, -0.06], [0.03, -0.02, 0.025, -0.01, 0.015]], dtype=float)
anchor = np.array([[0.7727272727272727, -0.17272727272727273, 0.009090909090909094], [-0.35454545454545455, 0.2545454545454545, 0.4681818181818182], [0.1772727272727273, 0.5227272727272727, 0.8909090909090909], [-0.2272727272727273, -0.37272727272727274, 0.1090909090909091], [0.4545454545454546, 0.7454545454545455, -0.2181818181818182]], dtype=float)
lo = 0.6
hi = 1.8
cosdrive = np.array([[0.0198, -0.027500000000000004, 0.0242, 0.013200000000000002, -0.0176], [0.0088, 0.006600000000000001, -0.0055000000000000005, 0.007700000000000001, -0.0044], [-0.0033000000000000004, 0.0044, 0.0055000000000000005, -0.0022, 0.0033000000000000004]], dtype=float)
sindrive = np.array([[-0.005600000000000001, 0.004, 0.0064, -0.008, 0.0048000000000000004], [0.002, -0.0016, 0.0028000000000000004, 0.0012000000000000001, -0.0024000000000000002], [0.0008, 0.0012000000000000001, -0.0016, 0.002, -0.0004]], dtype=float)
h = 0.19
n = 20
""",
            "call": "phase_averaged_curvature(A.copy(), C.copy(), cubic.copy(), beta.copy(), force.copy(), mass.copy(), anchor.copy(), lo, hi, cosdrive.copy(), sindrive.copy(), h, n)",
            "gold_call": "_oracle_phase_averaged_curvature(A.copy(), C.copy(), cubic.copy(), beta.copy(), force.copy(), mass.copy(), anchor.copy(), lo, hi, cosdrive.copy(), sindrive.copy(), h, n)",
            "tol": 2e-08,
        },
    ]
