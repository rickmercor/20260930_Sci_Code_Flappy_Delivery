"""
Response of the inferred delay-mixture bound to physicality confidence

The same latent rate coefficient vector $$\boldsymbol u\in B=[-1,1]^3$$ acts on every supplied central-rate scenario, with the supplied Gaussian mixture globally conditioned on $$B$$. At shallow fraction $$f$$, the delay law is the mixture of separately normalized power laws on $$[0.01\,\mathrm{Gyr},t_0]$$, and physicality is simultaneous nonnegative reconstructed formation rate in every scenario over the inclusive window $$0.1\le z\le1.5$$. Write its probability as $$P(f)$$. Determine the greatest allowed fraction $$f_*(q)$$ for the supplied threshold $$q$$, together with the resulting conditional coefficient statistics and their first two derivatives with respect to $$q$$. These responses quantify how the delay-mixture inference and the surviving rate uncertainty depend on the strength of the physicality requirement. Compose the preceding public functions and consume their outputs, including the local fraction-response information. Before any inference, certify the regularized inverse at both mixture endpoints of every supplied scenario by reconvolving its reconstructed formation history against that scenario's merger history over the same window, and reject a sampling whose largest fractional reconvolution error exceeds the supplied bound. Cosmology, rate modes, continuous delay normalization and the inverse operator are defined by the preceding steps. This is an authored sensitivity extension of posterior physicality conditioning; its coefficients and response values are not published catalog estimates. The zero-amplitude case recovers the main deterministic benchmark.

Returns
-------
np.ndarray of shape (3,11), the critical-fraction diagnostics and their first two ordinary threshold derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_formation_response_pipeline(pivot_rates: np.ndarray, r_asym: float,
                                       n_samples: int, amplitudes: np.ndarray,
                                       weights: np.ndarray, means: np.ndarray,
                                       covariances: np.ndarray,
                                       required_probability: float = 0.5,
                                       alpha_steep: float = -1.73,
                                       alpha_shallow: float = -0.99,
                                       kappa: float = 3.2,
                                       max_reconvolution_error: float = 0.1) -> np.ndarray:
    r'''Find a physicality bound and its response to the required probability.

    Parameters
    ----------
    pivot_rates : np.ndarray
        Nonempty finite real positive one-dimensional array of central rates at z=0.2, in Gpc^-3 yr^-1. All scenarios share the same latent coefficient draw; the event is simultaneous physicality of every scenario.
    r_asym : float
        Finite nonnegative absolute high-redshift boundary, in Gpc^-3 yr^-1, shared by all scenarios.
    n_samples : int
        Integer >= 16 for the preceding inclusive cosmic-time grid; the inclusive 0.1 <= z <= 1.5 window must contain a sample.
    amplitudes : np.ndarray
        Finite nonnegative shape (3,), sum < 1, used by the affine merger-family step. Zero amplitudes recover the deterministic common physicality bound.
    weights : np.ndarray
        Finite nonnegative shape (G,), G >= 1, with positive total. Gaussian-mixture weights before global truncation on the common box [-1,1]^3.
    means : np.ndarray
        Finite real shape (G,3), each coordinate in [-1,1]. These synthetic coefficient-posterior parameters are supplied inputs, not catalog results.
    covariances : np.ndarray
        Finite real shape (G,3,3), each symmetric within absolute tolerance 1e-12 with eigenvalues in [0.1,2]. This resolved posterior domain keeps the box probability and conditional moments numerically well conditioned.
    required_probability : float
        Finite probability threshold in [0.05,0.95], default 0.5. The joint acceptance probability as a function of shallow fraction is nonincreasing on [0,1], and at zero fraction strictly exceeds this threshold. These are input preconditions, not requested numerical certifications. For nonzero amplitudes, either fraction one exceeds the threshold in a neighborhood of q, or there is a unique interior root with strictly negative first probability derivative and locally fixed full-dimensional polytope incidences, with simple moving vertices. This smoothness precondition applies at the critical fraction; the region may change topology elsewhere. Zero amplitudes are also supported: the deterministic cutoff is independent of q in (0,1).
    alpha_steep, alpha_shallow : float
        Finite real delay exponents with alpha_steep < alpha_shallow, continuously normalized separately on [0.01 Gyr,t0].
    kappa : float
        Finite real central redshift exponent, default 3.2.
    max_reconvolution_error : float
        Finite positive bound on the reconvolution validity diagnostic, default 0.1. The regularized inverse is certified at both mixture endpoints of every scenario before any inference, so a sampling too coarse to resolve the delay kernel is rejected rather than reported.

    Returns
    -------
    diagnostics : np.ndarray
        Finite shape (3,11). Row 0 contains the critical shallow fraction, its joint physicality probability conditional on the box, three conditional means, and centered covariance entries [11,12,13,22,23,33]. Rows 1 and 2 contain the first and second ordinary derivatives of all these quantities with respect to required_probability q, holding every other argument fixed. Derivatives refer to the mathematical critical-fraction curve, not to the discrete iterations of a numerical solver. Select fraction one when it remains feasible locally in q. Its response rows, and those in the zero-amplitude deterministic case, are zero because these outputs are then locally independent of q. The fraction has absolute accuracy 1e-8; each other reported quantity has error at most 2e-5 times one plus its magnitude. At a deterministic cutoff use its feasible-side limit so the conditional moments are defined. Any sufficiently accurate deterministic numerical method is acceptable.

    Raises
    ------
    ValueError
        If the finite real input, shape, physical parameter or resolved posterior contracts fail; if the grid window is empty; if any scenario fails the endpoint reconvolution certification; if the all-steep endpoint does not exceed the required probability; if the result is nonfinite; or if a preceding function rejects its inputs. The monotonicity and local-incidence preconditions need not be certified.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_run_formation_response_pipeline(pivot_rates: np.ndarray, r_asym: float,
                                               n_samples: int, amplitudes: np.ndarray,
                                               weights: np.ndarray, means: np.ndarray,
                                               covariances: np.ndarray,
                                               required_probability: float = 0.5,
                                               alpha_steep: float = -1.73,
                                               alpha_shallow: float = -0.99,
                                               kappa: float = 3.2,
                                               max_reconvolution_error: float = 0.1) -> np.ndarray:
    values=(pivot_rates,amplitudes,weights,means,covariances)
    if any(not np.isrealobj(v) for v in values):
        raise ValueError("all rate and posterior arrays must be real")
    try:
        rates,amp,weight,mu,cov=[np.asarray(v,dtype=float) for v in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("all arrays must contain finite real values") from exc
    if rates.ndim!=1 or rates.size==0 or not np.all(np.isfinite(rates)) or np.any(rates<=0):
        raise ValueError("nonempty positive finite pivot rates required")
    if weight.ndim!=1 or weight.size==0 or mu.shape!=(weight.size,3) or cov.shape!=(weight.size,3,3):
        raise ValueError("posterior weight, mean and covariance shapes are inconsistent")
    if not all(np.all(np.isfinite(v)) for v in (weight,mu,cov)) or np.any(weight<0) or not np.any(weight>0) or np.any(np.abs(mu)>1):
        raise ValueError("nonnegative nonzero weights and means inside [-1,1]^3 required")
    for matrix in cov:
        if not np.allclose(matrix,matrix.T,rtol=0,atol=1e-12):
            raise ValueError("posterior covariances must be symmetric")
        eigenvalues=np.linalg.eigvalsh((matrix+matrix.T)/2)
        if eigenvalues[0]<.1 or eigenvalues[-1]>2:
            raise ValueError("posterior covariance eigenvalues must lie in [0.1,2]")
    if any(not np.isscalar(v) or not np.isrealobj(v) for v in (required_probability,alpha_steep,alpha_shallow)):
        raise ValueError("threshold and delay indices must be real scalars")
    try:
        target,steep,shallow=map(float,(required_probability,alpha_steep,alpha_shallow))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("threshold and delay indices must be finite real scalars") from exc
    if not np.all(np.isfinite([target,steep,shallow])) or not .05<=target<=.95 or not steep<shallow:
        raise ValueError("threshold in [0.05,0.95] and ordered finite delay indices required")
    if not np.isscalar(max_reconvolution_error) or not np.isrealobj(max_reconvolution_error):
        raise ValueError("the reconvolution bound must be a real scalar")
    try:
        reconvolution_bound=float(max_reconvolution_error)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("the reconvolution bound must be a finite real scalar") from exc
    if not np.isfinite(reconvolution_bound) or reconvolution_bound<=0.0:
        raise ValueError("a finite positive reconvolution bound is required")
    t0 = _oracle_compute_cosmic_age(0.0)
    grid = _oracle_build_time_redshift_grid(n_samples)
    time, redshift = grid
    dt = float(time[1] - time[0])
    window = (redshift >= .1) & (redshift <= 1.5)
    if not np.any(window):
        raise ValueError("the physicality window must contain a sample")
    ps = _oracle_delay_time_distribution(time, steep, .01, t0)
    pl = _oracle_delay_time_distribution(time, shallow, .01, t0)
    families = [_oracle_affine_merger_family(redshift, float(rate), r_asym, amp, kappa)
                for rate in rates]
    for family in families:
        for endpoint, density in ((0.0, ps), (1.0, pl)):
            reconstruction = _oracle_wiener_formation_jet(family, ps, pl, endpoint, dt)[0, 0]
            if _oracle_reconvolution_error(reconstruction, density, family[0], dt,
                                           window) > reconvolution_bound:
                raise ValueError("the regularized inverse fails endpoint reconvolution certification")
    joint_window = np.tile(window, rates.size)
    lower_box, upper_box = -np.ones(3), np.ones(3)
    box = np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    box_jet = np.stack((box, np.zeros_like(box), np.zeros_like(box)))
    prior = np.array([_oracle_gaussian_polytope_moment_jet(box_jet, center, matrix)[0]
                      for center, matrix in zip(mu, cov)])

    def _posterior_response_at_fraction(fraction, response=False):
        formation = np.concatenate([_oracle_wiener_formation_jet(family, ps, pl, fraction, dt)
                                    for family in families], axis=2)
        if not response:
            formation = np.stack((formation[0], np.zeros_like(formation[0]), np.zeros_like(formation[0])))
        vertices = _oracle_physicality_vertex_jet(formation, joint_window, lower_box, upper_box)
        if vertices.shape[1] == 0:
            accepted = np.zeros((weight.size,3,10))
        elif vertices.shape[1] == 8 and np.all(vertices[1:] == 0.0) and np.allclose(
                vertices[0][np.lexsort((vertices[0,:,2],vertices[0,:,1],vertices[0,:,0]))],
                box, rtol=0.0, atol=1e-12):
            accepted = np.zeros((weight.size,3,10))
            accepted[:,0] = prior
        else:
            accepted = np.array([_oracle_gaussian_polytope_moment_jet(vertices, center, matrix)
                                 for center, matrix in zip(mu,cov)])
        return _oracle_condition_gaussian_mixture_jet(accepted, prior, weight)

    lower_summary = _posterior_response_at_fraction(0.0)
    if lower_summary[0,0] <= target:
        raise ValueError("the all-steep endpoint must exceed the required probability")
    upper_summary = _posterior_response_at_fraction(1.0)
    result = np.zeros((3,11))
    if upper_summary[0,0] >= target:
        result[0] = np.r_[1.0, upper_summary[0]]
        return result
    lower, upper = 0.0, 1.0
    while upper - lower > 1e-11:
        middle = (lower + upper) / 2.0
        summary = _posterior_response_at_fraction(middle)
        if summary[0,0] >= target:
            lower, lower_summary = middle, summary
        else:
            upper = middle
    result[0] = np.r_[lower, lower_summary[0]]
    if np.all(amp == 0.0):
        return result
    summary = _posterior_response_at_fraction(lower, response=True)
    slope, curvature = summary[1,0], summary[2,0]
    if not np.isfinite(slope) or slope >= 0.0:
        raise ValueError("the smooth interior critical fraction requires a negative probability slope")
    inverse_slope = 1.0 / slope
    inverse_curvature = -curvature / slope**3
    result[0,1:] = summary[0]
    result[1] = np.r_[inverse_slope, summary[1]*inverse_slope]
    result[2] = np.r_[inverse_curvature,
                      summary[2]*inverse_slope**2 + summary[1]*inverse_curvature]
    if not np.all(np.isfinite(result)):
        raise ValueError("the threshold response must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nfrom scipy.optimize import linprog\nfrom scipy.spatial import HalfspaceIntersection, ConvexHull\nfrom numpy.polynomial.legendre import leggauss\n'
               'rates=np.array([22.5, 29, 37.5],dtype=float)\n'
               'amp=np.array([0, 0, 0],dtype=float)\n'
               'w=np.array([1],dtype=float)\n'
               'mu=np.array([[0, 0, 0]],dtype=float)\n'
               'cov=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],dtype=float)\n'
               'asym=300;n=4096;q=0.5\n',
      'call': 'run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'gold_call': '_oracle_run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'tol': 2e-05},
     {'setup': 'import numpy as np\nfrom scipy.optimize import linprog\nfrom scipy.spatial import HalfspaceIntersection, ConvexHull\nfrom numpy.polynomial.legendre import leggauss\n'
               'rates=np.array([22.5, 29, 37.5],dtype=float)\n'
               'amp=np.array([0.24, 0.2, 0.2],dtype=float)\n'
               'w=np.array([0.6, 0.4],dtype=float)\n'
               'mu=np.array([[0.25, -0.2, 0.1], [-0.3, 0.35, -0.15]],dtype=float)\n'
               'cov=np.array([[[0.35, 0.1, -0.06], [0.1, 0.45, 0.08], [-0.06, 0.08, 0.3]], [[0.5, '
               '-0.12, 0.07], [-0.12, 0.3, -0.04], [0.07, -0.04, 0.4]]],dtype=float)\n'
               'asym=300;n=1024;q=0.5\n',
      'call': 'run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'gold_call': '_oracle_run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'tol': 2e-05},
     {'setup': 'import numpy as np\nfrom scipy.optimize import linprog\nfrom scipy.spatial import HalfspaceIntersection, ConvexHull\nfrom numpy.polynomial.legendre import leggauss\n'
               'rates=np.array([22.5, 29, 37.5],dtype=float)\n'
               'amp=np.array([0.24, 0.2, 0.2],dtype=float)\n'
               'w=np.array([0.6, 0.4],dtype=float)\n'
               'mu=np.array([[0.25, -0.2, 0.1], [-0.3, 0.35, -0.15]],dtype=float)\n'
               'cov=np.array([[[0.35, 0.1, -0.06], [0.1, 0.45, 0.08], [-0.06, 0.08, 0.3]], [[0.5, '
               '-0.12, 0.07], [-0.12, 0.3, -0.04], [0.07, -0.04, 0.4]]],dtype=float)\n'
               'asym=300;n=1024;q=0.8\n',
      'call': 'run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'gold_call': '_oracle_run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'tol': 2e-05},
     {'setup': 'import numpy as np\nfrom scipy.optimize import linprog\nfrom scipy.spatial import HalfspaceIntersection, ConvexHull\nfrom numpy.polynomial.legendre import leggauss\n'
               'rates=np.array([37.5, 22.5],dtype=float)\n'
               'amp=np.array([0.15, 0.25, 0.15],dtype=float)\n'
               'w=np.array([2, 3],dtype=float)\n'
               'mu=np.array([[0.25, -0.2, 0.1], [-0.3, 0.35, -0.15]],dtype=float)\n'
               'cov=np.array([[[0.35, 0.1, -0.06], [0.1, 0.45, 0.08], [-0.06, 0.08, 0.3]], [[0.5, '
               '-0.12, 0.07], [-0.12, 0.3, -0.04], [0.07, -0.04, 0.4]]],dtype=float)\n'
               'asym=100;n=512;q=0.5\n',
      'call': 'run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'gold_call': '_oracle_run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'tol': 2e-05},
     {'setup': 'import numpy as np\nfrom scipy.optimize import linprog\nfrom scipy.spatial import HalfspaceIntersection, ConvexHull\nfrom numpy.polynomial.legendre import leggauss\n'
               'rates=np.array([29],dtype=float)\n'
               'amp=np.array([0, 0, 0],dtype=float)\n'
               'w=np.array([0.6, 0.4],dtype=float)\n'
               'mu=np.array([[0.25, -0.2, 0.1], [-0.3, 0.35, -0.15]],dtype=float)\n'
               'cov=np.array([[[0.35, 0.1, -0.06], [0.1, 0.45, 0.08], [-0.06, 0.08, 0.3]], [[0.5, '
               '-0.12, 0.07], [-0.12, 0.3, -0.04], [0.07, -0.04, 0.4]]],dtype=float)\n'
               'asym=0;n=1024;q=0.8\n',
      'call': 'run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'gold_call': '_oracle_run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'tol': 2e-05},
     {'setup': 'import numpy as np\nfrom scipy.optimize import linprog\nfrom scipy.spatial import HalfspaceIntersection, ConvexHull\nfrom numpy.polynomial.legendre import leggauss\n'
               'rates=np.array([22.5, 29.0, 37.5],dtype=float)\n'
               'amp=np.array([0.24, 0.2, 0.2],dtype=float)\n'
               'w=np.array([0.65, 0.35],dtype=float)\n'
               'mu=np.array([[-0.75, 0.4, 0.3], [-0.5, -0.2, 0.6]],dtype=float)\n'
               'cov=np.array([[[0.15, 0.03, 0.02], [0.03, 0.23, -0.025], [0.02, -0.025, 0.21]], '
               '[[0.18, -0.025, 0.035], [-0.025, 0.21, 0.03], [0.035, 0.03, 0.19]]],dtype=float)\n'
               'asym=300.0;n=1024;q=0.8\n',
      'call': 'run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'gold_call': '_oracle_run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'tol': 2e-05},
     {'setup': 'import numpy as np\nfrom scipy.optimize import linprog\nfrom scipy.spatial import HalfspaceIntersection, ConvexHull\nfrom numpy.polynomial.legendre import leggauss\n'
               'rates=np.array([22.5, 29.0, 37.5],dtype=float)\n'
               'amp=np.array([0.06, 0.42, 0.42],dtype=float)\n'
               'w=np.array([0.6, 0.4],dtype=float)\n'
               'mu=np.array([[0.25, -0.2, 0.1], [-0.3, 0.35, -0.15]],dtype=float)\n'
               'cov=np.array([[[0.35, 0.1, -0.06], [0.1, 0.45, 0.08], [-0.06, 0.08, 0.3]], [[0.5, '
               '-0.12, 0.07], [-0.12, 0.3, -0.04], [0.07, -0.04, 0.4]]],dtype=float)\n'
               'asym=300.0;n=1024;q=0.5\n',
      'call': 'run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'gold_call': '_oracle_run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'tol': 2e-05},
     {'setup': 'import numpy as np\nfrom scipy.optimize import linprog\nfrom scipy.spatial import HalfspaceIntersection, ConvexHull\nfrom numpy.polynomial.legendre import leggauss\n'
               'rates=np.array([29.0, 37.5],dtype=float)\n'
               'amp=np.array([0.24, 0.2, 0.2],dtype=float)\n'
               'w=np.array([0.3, 0.45, 0.25],dtype=float)\n'
               'mu=np.array([[0.25, -0.2, 0.1], [-0.3, 0.35, -0.15], [0.1, 0.1, 0.35]],dtype=float)\n'
               'cov=np.array([[[0.35, 0.1, -0.06], [0.1, 0.45, 0.08], [-0.06, 0.08, 0.3]], [[0.5, '
               '-0.12, 0.07], [-0.12, 0.3, -0.04], [0.07, -0.04, 0.4]], [[0.25, 0.04, 0.03], [0.04, '
               '0.4, -0.07], [0.03, -0.07, 0.3]]],dtype=float)\n'
               'asym=100.0;n=2048;q=0.8\n',
      'call': 'run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'gold_call': '_oracle_run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'tol': 2e-05},
     {'setup': 'import numpy as np\nfrom scipy.optimize import linprog\nfrom scipy.spatial import HalfspaceIntersection, ConvexHull\nfrom numpy.polynomial.legendre import leggauss\n'
               'rates=np.array([22.5, 29.0, 37.5],dtype=float)\n'
               'amp=np.array([0.06, 0.42, 0.42],dtype=float)\n'
               'w=np.array([0.6, 0.4],dtype=float)\n'
               'mu=np.array([[0.25, -0.2, 0.1], [-0.3, 0.35, -0.15]],dtype=float)\n'
               'cov=np.array([[[0.35, 0.1, -0.06], [0.1, 0.45, 0.08], [-0.06, 0.08, 0.3]], [[0.5, '
               '-0.12, 0.07], [-0.12, 0.3, -0.04], [0.07, -0.04, 0.4]]],dtype=float)\n'
               'asym=300.0;n=1024;q=0.8\n',
      'call': 'run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'gold_call': '_oracle_run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)',
      'tol': 2e-05},
     {'setup': 'import numpy as np\nfrom scipy.optimize import linprog\nfrom scipy.spatial import HalfspaceIntersection, ConvexHull\nfrom numpy.polynomial.legendre import leggauss\n'
               'rates=np.array([22.5, 29.0, 37.5],dtype=float)\n'
               'amp=np.array([0.0, 0.0, 0.0],dtype=float)\n'
               'w=np.array([1.0],dtype=float)\n'
               'mu=np.array([[0.0, 0.0, 0.0]],dtype=float)\n'
               'cov=np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],dtype=float)\n'
               'asym=300.0;n=16;q=0.5\n'
               'def run_model():\n'
               '    try:\n'
               '        run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\nfrom scipy.optimize import linprog\nfrom scipy.spatial import HalfspaceIntersection, ConvexHull\nfrom numpy.polynomial.legendre import leggauss\n'
               'rates=np.array([22.5, 29.0, 37.5],dtype=float)\n'
               'amp=np.array([0.0, 0.0, 0.0],dtype=float)\n'
               'w=np.array([1.0],dtype=float)\n'
               'mu=np.array([[0.0, 0.0, 0.0]],dtype=float)\n'
               'cov=np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],dtype=float)\n'
               'asym=300.0;n=1024;q=0.5\n'
               'def run_model():\n'
               '    try:\n'
               '        run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q,max_reconvolution_error=0.005)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_run_formation_response_pipeline(rates.copy(),asym,n,amp.copy(),w.copy(),mu.copy(),cov.copy(),q,max_reconvolution_error=0.005)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
