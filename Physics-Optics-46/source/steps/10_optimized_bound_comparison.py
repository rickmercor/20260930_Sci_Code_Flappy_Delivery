"""
Compose the two-parameter throughput-limited optimization and the empirical in-band bound comparison into the complete benchmark.

This orchestrator selects the admissible coupling and detuning that minimize the exact in-band noise amplification at fixed detector noise, subject to the detector throughput ceiling, then extracts the throughput and normalized spectral correlation at the selected design. A constrained half-width fit defines the parameter used in the literature's underdetermined asymptotic prediction.



The in-band figure of merit gives unit weight to the $$K$$ frequency channels at the centre of the comb, zero-based indices $$(N-K)//2$$ through $$(N-K)//2+K-1$$, and zero weight elsewhere. The prediction for the weighted in-band sum is taken to be the full-comb prediction multiplied by $$K/N$$. The selected design is also characterised by the throughput multiplier and the constrained curvature returned by constrained_sensitivity.



The final scalar is the signed percentage discrepancy $$D=100(F_{\mathrm{pred}}/F_\star-1)$$, where $$F_\star$$ is the exact minimized in-band inverse-Gram variance and $$F_{\mathrm{pred}}$$ is the in-band empirical prediction evaluated from the fitted parameters at the selected design. Every earlier result enters the optimization, the fit or the reported comparison.

Returns
-------
np.ndarray: Real vector [D, q_star, s_star, F_star, T0, a_fit, fit_loss, F_pred, multiplier, tangential_curvature], shape (10,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimized_bound_comparison(n_modes: int, n_ports: int, n_channels: int, seed: int, bandwidth: float, bounds: "np.ndarray", throughput_limit: float, band_channels: int, max_lag: int, width_interval: "np.ndarray", scan_points: "np.ndarray") -> "np.ndarray":
    r"""Orchestrator for the two-parameter finite-cavity model comparison.
    
    Parameters
    ----------
    n_modes : int
        Mode count P >= M + 1.
    n_ports : int
        Measured output count M >= 2.
    n_channels : int
        Frequency count N > M.
    seed : int
        Nonnegative realization seed, as in cavity_realization.
    bandwidth : float
        Positive total frequency span B; sample N equally spaced frequencies
        from -B/2 through B/2, including both endpoints, before detuning.
    bounds : np.ndarray
        Real array [[q_lower,q_upper],[s_lower,s_upper]], the design rectangle,
        satisfying stationary_design's resolution and full-rank preconditions.
    throughput_limit : float
        Positive detector throughput ceiling, as in stationary_design.
    band_channels : int
        Count K, with 1 <= K <= N, of unit-weight central frequency channels
        placed as described in the background.
    max_lag : int
        Largest correlation lag L, with 1 <= L < N.
    width_interval : np.ndarray
        Positive [a_lower,a_upper] with a_upper <= 30. The optimized profile
        satisfies fit_lorentzian's uniqueness precondition.
    scan_points : np.ndarray
        Two integers >= 3, the scan node counts of stationary_design.
    
    Returns
    -------
    result : np.ndarray
        Real vector [D, q_star, s_star, F_star, T0, a_fit, fit_loss, F_pred,
        multiplier, tangential_curvature], shape (10,). D is in percent and
        F_star and F_pred are in-band sums.
    
    Notes
    -----
    Compose all earlier public functions and consume their returned values.
    Inputs satisfy the stated domain and the supplied arrays remain unchanged.
    No null-space bias, stochastic noise sampling, or realization averaging is added.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_optimized_bound_comparison(n_modes: int, n_ports: int, n_channels: int, seed: int, bandwidth: float, bounds: "np.ndarray", throughput_limit: float, band_channels: int, max_lag: int, width_interval: "np.ndarray", scan_points: "np.ndarray") -> "np.ndarray":
    cavity = _oracle_cavity_realization(n_modes, n_ports, seed)
    frequencies = np.linspace(-bandwidth/2, bandwidth/2, n_channels)
    weights = np.zeros(n_channels)
    start = (n_channels-band_channels)//2
    weights[start:start+band_channels] = 1.0
    design = _oracle_stationary_design(cavity, frequencies, weights, bounds, throughput_limit, scan_points)
    scattering = _oracle_scattering_jet(cavity, frequencies, design[0], design[1])
    transmission = _oracle_transmission_jet(scattering)
    profile = _oracle_spectral_correlation(transmission[0], max_lag)
    fit = _oracle_fit_lorentzian(profile, width_interval)
    predicted = _oracle_asymptotic_trace(fit[0], profile[0], n_ports, n_channels)*band_channels/n_channels
    sensitivity = _oracle_constrained_sensitivity(transmission, weights, bool(design[3]),
                                                  np.array([bool(design[4]), bool(design[5])]))
    discrepancy = 100*(predicted/design[2]-1)
    return np.array([discrepancy, design[0], design[1], design[2], profile[0], fit[0], fit[1],
                     predicted, sensitivity[0], sensitivity[2]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit differential cases for Studio contract validation."""
    return [
        {
            "setup": 'import numpy as np\np=32;m=6;n=48;seed=17;band=1.6;lag=8;limit=.76;kb=24\nbounds=np.array([[.3,1.],[-.06,.06]]);width=np.array([.05,20.]);scan=np.array([8,5])\ndef measure(fn):\n    b=bounds.copy(); w=width.copy(); sc=scan.copy()\n    out=np.asarray(fn(p,m,n,seed,band,b,limit,kb,lag,w,sc))\n    assert out.shape==(10,) and np.array_equal(b,bounds) and np.array_equal(w,width) and np.array_equal(sc,scan)\n    return out/np.array([1,1,1,1,1,1,1,1,1,1e3])\np=12;m=3;n=18;band=1.4;lag=5;limit=.72;kb=9;bounds=np.array([[-.2,.8],[-.12,.12]]);scan=np.array([6,5])\n',
            "call": "measure(optimized_bound_comparison)",
            "gold_call": "measure(_oracle_optimized_bound_comparison)",
            "tol": 2e-06
        },
        {
            "setup": 'import numpy as np\np=32;m=6;n=48;seed=17;band=1.6;lag=8;limit=.76;kb=24\nbounds=np.array([[.3,1.],[-.06,.06]]);width=np.array([.05,20.]);scan=np.array([8,5])\ndef measure(fn):\n    b=bounds.copy(); w=width.copy(); sc=scan.copy()\n    out=np.asarray(fn(p,m,n,seed,band,b,limit,kb,lag,w,sc))\n    assert out.shape==(10,) and np.array_equal(b,bounds) and np.array_equal(w,width) and np.array_equal(sc,scan)\n    return out/np.array([1,1,1,1,1,1,1,1,1,1e3])\np=12;m=3;n=18;band=1.4;lag=5;limit=.95;kb=9;bounds=np.array([[-.2,.8],[-.12,.12]]);scan=np.array([6,5])\n',
            "call": "measure(optimized_bound_comparison)",
            "gold_call": "measure(_oracle_optimized_bound_comparison)",
            "tol": 2e-06
        },
        {
            "setup": 'import numpy as np\np=32;m=6;n=48;seed=17;band=1.6;lag=8;limit=.76;kb=24\nbounds=np.array([[.3,1.],[-.06,.06]]);width=np.array([.05,20.]);scan=np.array([8,5])\ndef measure(fn):\n    b=bounds.copy(); w=width.copy(); sc=scan.copy()\n    out=np.asarray(fn(p,m,n,seed,band,b,limit,kb,lag,w,sc))\n    assert out.shape==(10,) and np.array_equal(b,bounds) and np.array_equal(w,width) and np.array_equal(sc,scan)\n    return out/np.array([1,1,1,1,1,1,1,1,1,1e3])\np=12;m=3;n=18;band=1.4;lag=4;limit=.72;kb=18;bounds=np.array([[-.2,.8],[-.12,.12]]);scan=np.array([6,5])\n',
            "call": "measure(optimized_bound_comparison)",
            "gold_call": "measure(_oracle_optimized_bound_comparison)",
            "tol": 2e-06
        },
        {
            "setup": 'import numpy as np\np=32;m=6;n=48;seed=17;band=1.6;lag=8;limit=.76;kb=24\nbounds=np.array([[.3,1.],[-.06,.06]]);width=np.array([.05,20.]);scan=np.array([8,5])\ndef measure(fn):\n    b=bounds.copy(); w=width.copy(); sc=scan.copy()\n    out=np.asarray(fn(p,m,n,seed,band,b,limit,kb,lag,w,sc))\n    assert out.shape==(10,) and np.array_equal(b,bounds) and np.array_equal(w,width) and np.array_equal(sc,scan)\n    return out/np.array([1,1,1,1,1,1,1,1,1,1e3])\np=10;m=3;n=16;seed=9;band=1.4;lag=5;limit=.60;kb=6;bounds=np.array([[-.1,.9],[-.15,.15]]);width=np.array([.1,15.]);scan=np.array([6,5])\n',
            "call": "measure(optimized_bound_comparison)",
            "gold_call": "measure(_oracle_optimized_bound_comparison)",
            "tol": 2e-06
        },
        {
            "setup": 'import numpy as np\np=32;m=6;n=48;seed=17;band=1.6;lag=8;limit=.76;kb=24\nbounds=np.array([[.3,1.],[-.06,.06]]);width=np.array([.05,20.]);scan=np.array([8,5])\ndef measure(fn):\n    b=bounds.copy(); w=width.copy(); sc=scan.copy()\n    out=np.asarray(fn(p,m,n,seed,band,b,limit,kb,lag,w,sc))\n    assert out.shape==(10,) and np.array_equal(b,bounds) and np.array_equal(w,width) and np.array_equal(sc,scan)\n    return out/np.array([1,1,1,1,1,1,1,1,1,1e3])\np=12;m=3;n=18;band=1.4;lag=5;limit=.72;kb=9;bounds=np.array([[-.2,.8],[-.12,.12]]);scan=np.array([6,5])\nwidth=np.array([.05,.3])\n',
            "call": "measure(optimized_bound_comparison)",
            "gold_call": "measure(_oracle_optimized_bound_comparison)",
            "tol": 2e-06
        },
        {
            "setup": 'import numpy as np\np=32;m=6;n=48;seed=17;band=1.6;lag=8;limit=.76;kb=24\nbounds=np.array([[.3,1.],[-.06,.06]]);width=np.array([.05,20.]);scan=np.array([8,5])\ndef measure(fn):\n    b=bounds.copy(); w=width.copy(); sc=scan.copy()\n    out=np.asarray(fn(p,m,n,seed,band,b,limit,kb,lag,w,sc))\n    assert out.shape==(10,) and np.array_equal(b,bounds) and np.array_equal(w,width) and np.array_equal(sc,scan)\n    return out/np.array([1,1,1,1,1,1,1,1,1,1e3])\n',
            "call": "measure(optimized_bound_comparison)",
            "gold_call": "measure(_oracle_optimized_bound_comparison)",
            "tol": 2e-06
        }
    ]
