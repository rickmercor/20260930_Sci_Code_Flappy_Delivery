"""
Evaluate the throughput multiplier, the limit response and the constrained curvature of the selected two-parameter design.

Where the throughput limit binds, the design problem's first-order optimality conditions balance the objective's gradient against the throughput constraint and against whichever bounds of the design rectangle are active at the design. The throughput constraint's multiplier in those conditions measures how the optimized in-band variance responds to the detector limit.

The third returned quantity is the objective's second derivative along the constraint curve $$T_0=T_{\max}$$ through the design, parameterized by arclength. By convention it is zero when a rectangle bound is active at the design, and all three returned quantities are zero when the limit does not bind. The caller supplies which regime holds. The throughput derivatives follow from the intensity jet, with each order the average over frequency channels of the summed measured intensities of the corresponding intensity derivative, matching the throughput convention of the correlation step.

Returns
-------
np.ndarray: Real vector [multiplier, dF_star/dT_max, tangential_curvature], shape (3,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def constrained_sensitivity(transmission: "np.ndarray", weights: "np.ndarray", throughput_active: bool, pinned_axes: "np.ndarray") -> "np.ndarray":
    r"""Return the constraint multiplier, the limit response and the tangential curvature.
    
    Parameters
    ----------
    transmission : np.ndarray
        Finite real array (6, M, N), the intensity jet at the selected design,
        as returned by transmission_jet. A is full rank min(M,N), M,N >= 1,
        and meets inverse_gram_jet's conditioning bound.
    weights : np.ndarray
        Finite nonnegative channel weights of length N, as in inverse_gram_jet.
    throughput_active : bool
        True when the throughput limit binds at the selected design; False
        otherwise. When True, not every coordinate sits on a bound and the
        throughput multiplier is uniquely determined.
    pinned_axes : np.ndarray
        Two booleans, True where the corresponding design coordinate sits on a
        bound of the design rectangle.
    
    Returns
    -------
    result : np.ndarray
        Real vector [multiplier, dF_star/dT_max, tangential_curvature],
        shape (3,): the throughput constraint's multiplier in the first-order
        optimality conditions, nonnegative at a binding minimum; the derivative
        of the optimized in-band variance with respect to the detector limit;
        and the curvature defined in the background.
    
    Notes
    -----
    Inputs satisfy the stated domain and must remain unchanged. The returned
    quantities are exact, not finite-difference estimates of a re-solved
    optimization.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_constrained_sensitivity(transmission: "np.ndarray", weights: "np.ndarray", throughput_active: bool, pinned_axes: "np.ndarray") -> "np.ndarray":
    if not throughput_active:
        return np.zeros(3)
    objective = _oracle_inverse_gram_jet(transmission, weights)
    throughput = transmission.sum(axis=1).mean(axis=1)
    grad_f, hess_f = _split(objective)
    grad_t, hess_t = _split(throughput)
    free = ~np.asarray(pinned_axes, dtype=bool)
    # Only the directions the rectangle leaves free can respond to the limit.
    multiplier = -float(grad_f[free]@grad_t[free]/(grad_t[free]@grad_t[free]))
    if free.sum() < 2:
        return np.array([multiplier, -multiplier, 0.0])
    tangent = np.array([-grad_t[1], grad_t[0]])
    tangent = tangent/np.linalg.norm(tangent)
    curvature = float(tangent@(hess_f+multiplier*hess_t)@tangent)
    return np.array([multiplier, -multiplier, curvature])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit differential cases for Studio contract validation."""
    return [
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(5)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.array([0.,1,1,1,1,1,1,1,1,1,0,0.])\n# Design returned by stationary_design for this instance. Both coordinates free, gradients parallel.\ndesign=(0.33273357942297443,-0.11183505227894337)\nw=weights; active=True; pinned=np.array([False, False])\ndef measure(fn, amplitude, intensity):\n    # Each side builds its own jet through its own dependency chain.\n    jet=np.asarray(intensity(amplitude(cavity.copy(),freq.copy(),design[0],design[1])))\n    reference=jet.copy()\n    v2=w.copy(); pn=pinned.copy(); out=np.asarray(fn(jet,v2,active,pn))\n    assert out.shape==(3,) and np.array_equal(jet,reference)\n    assert np.array_equal(v2,w) and np.array_equal(pn,pinned)\n    return out/np.array([1,1,1e2])\n',
            "call": "measure(constrained_sensitivity, scattering_jet, transmission_jet)",
            "gold_call": "measure(_oracle_constrained_sensitivity, _oracle_scattering_jet, _oracle_transmission_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(5)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.array([0.,1,1,1,1,1,1,1,1,1,0,0.])\n# Design returned by stationary_design for this instance. Same design with the limit not binding.\ndesign=(0.33273357942297443,-0.11183505227894337)\nw=weights; active=False; pinned=np.array([False, False])\ndef measure(fn, amplitude, intensity):\n    # Each side builds its own jet through its own dependency chain.\n    jet=np.asarray(intensity(amplitude(cavity.copy(),freq.copy(),design[0],design[1])))\n    reference=jet.copy()\n    v2=w.copy(); pn=pinned.copy(); out=np.asarray(fn(jet,v2,active,pn))\n    assert out.shape==(3,) and np.array_equal(jet,reference)\n    assert np.array_equal(v2,w) and np.array_equal(pn,pinned)\n    return out/np.array([1,1,1e2])\n',
            "call": "measure(constrained_sensitivity, scattering_jet, transmission_jet)",
            "gold_call": "measure(_oracle_constrained_sensitivity, _oracle_scattering_jet, _oracle_transmission_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(7)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.array([0.,1,1,1,1,1,1,1,1,1,0,0.])\n# Design returned by stationary_design for this instance. Second instance with both coordinates free.\ndesign=(0.579551512192315,0.13876728920594686)\nw=weights; active=True; pinned=np.array([False, False])\ndef measure(fn, amplitude, intensity):\n    # Each side builds its own jet through its own dependency chain.\n    jet=np.asarray(intensity(amplitude(cavity.copy(),freq.copy(),design[0],design[1])))\n    reference=jet.copy()\n    v2=w.copy(); pn=pinned.copy(); out=np.asarray(fn(jet,v2,active,pn))\n    assert out.shape==(3,) and np.array_equal(jet,reference)\n    assert np.array_equal(v2,w) and np.array_equal(pn,pinned)\n    return out/np.array([1,1,1e2])\n',
            "call": "measure(constrained_sensitivity, scattering_jet, transmission_jet)",
            "gold_call": "measure(_oracle_constrained_sensitivity, _oracle_scattering_jet, _oracle_transmission_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(3)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.array([0.,1,1,1,1,1,1,1,1,1,0,0.])\n# Design returned by stationary_design for this instance. Detuning pinned by the rectangle.\ndesign=(0.2252689067770365,-0.1)\nw=weights; active=True; pinned=np.array([False, True])\ndef measure(fn, amplitude, intensity):\n    # Each side builds its own jet through its own dependency chain.\n    jet=np.asarray(intensity(amplitude(cavity.copy(),freq.copy(),design[0],design[1])))\n    reference=jet.copy()\n    v2=w.copy(); pn=pinned.copy(); out=np.asarray(fn(jet,v2,active,pn))\n    assert out.shape==(3,) and np.array_equal(jet,reference)\n    assert np.array_equal(v2,w) and np.array_equal(pn,pinned)\n    return out/np.array([1,1,1e2])\n',
            "call": "measure(constrained_sensitivity, scattering_jet, transmission_jet)",
            "gold_call": "measure(_oracle_constrained_sensitivity, _oracle_scattering_jet, _oracle_transmission_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(7)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.ones(12)\n# Design returned by stationary_design for this instance. Uniform weights, detuning pinned.\ndesign=(0.6112542061484577,0.1)\nw=weights; active=True; pinned=np.array([False, True])\ndef measure(fn, amplitude, intensity):\n    # Each side builds its own jet through its own dependency chain.\n    jet=np.asarray(intensity(amplitude(cavity.copy(),freq.copy(),design[0],design[1])))\n    reference=jet.copy()\n    v2=w.copy(); pn=pinned.copy(); out=np.asarray(fn(jet,v2,active,pn))\n    assert out.shape==(3,) and np.array_equal(jet,reference)\n    assert np.array_equal(v2,w) and np.array_equal(pn,pinned)\n    return out/np.array([1,1,1e2])\n',
            "call": "measure(constrained_sensitivity, scattering_jet, transmission_jet)",
            "gold_call": "measure(_oracle_constrained_sensitivity, _oracle_scattering_jet, _oracle_transmission_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(3)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.array([0.,1,1,1,1,1,1,1,1,1,0,0.])\n# Design returned by stationary_design for this instance. Interior stationary design, limit inactive.\ndesign=(0.31034519387489634,-0.07387120356753268)\nw=weights; active=False; pinned=np.array([False, False])\ndef measure(fn, amplitude, intensity):\n    # Each side builds its own jet through its own dependency chain.\n    jet=np.asarray(intensity(amplitude(cavity.copy(),freq.copy(),design[0],design[1])))\n    reference=jet.copy()\n    v2=w.copy(); pn=pinned.copy(); out=np.asarray(fn(jet,v2,active,pn))\n    assert out.shape==(3,) and np.array_equal(jet,reference)\n    assert np.array_equal(v2,w) and np.array_equal(pn,pinned)\n    return out/np.array([1,1,1e2])\n',
            "call": "measure(constrained_sensitivity, scattering_jet, transmission_jet)",
            "gold_call": "measure(_oracle_constrained_sensitivity, _oracle_scattering_jet, _oracle_transmission_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(5)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.array([0.,1,1,1,1,1,1,1,1,1,0,0.])\n# Design returned by stationary_design for this instance. Edge design, limit inactive.\ndesign=(0.4723252853847898,-0.05)\nw=weights; active=False; pinned=np.array([False, True])\ndef measure(fn, amplitude, intensity):\n    # Each side builds its own jet through its own dependency chain.\n    jet=np.asarray(intensity(amplitude(cavity.copy(),freq.copy(),design[0],design[1])))\n    reference=jet.copy()\n    v2=w.copy(); pn=pinned.copy(); out=np.asarray(fn(jet,v2,active,pn))\n    assert out.shape==(3,) and np.array_equal(jet,reference)\n    assert np.array_equal(v2,w) and np.array_equal(pn,pinned)\n    return out/np.array([1,1,1e2])\n',
            "call": "measure(constrained_sensitivity, scattering_jet, transmission_jet)",
            "gold_call": "measure(_oracle_constrained_sensitivity, _oracle_scattering_jet, _oracle_transmission_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(5)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.zeros(12); weights[4:8]=1.\n# Design returned by stationary_design for this instance. Narrow band; this optimum pins the detuning.\ndesign=(0.40491564947447795,-0.25)\nw=weights; active=True; pinned=np.array([False, True])\ndef measure(fn, amplitude, intensity):\n    # Each side builds its own jet through its own dependency chain.\n    jet=np.asarray(intensity(amplitude(cavity.copy(),freq.copy(),design[0],design[1])))\n    reference=jet.copy()\n    v2=w.copy(); pn=pinned.copy(); out=np.asarray(fn(jet,v2,active,pn))\n    assert out.shape==(3,) and np.array_equal(jet,reference)\n    assert np.array_equal(v2,w) and np.array_equal(pn,pinned)\n    return out/np.array([1,1,1e2])\n',
            "call": "measure(constrained_sensitivity, scattering_jet, transmission_jet)",
            "gold_call": "measure(_oracle_constrained_sensitivity, _oracle_scattering_jet, _oracle_transmission_jet)",
            "tol": 1e-10
        }
    ]
