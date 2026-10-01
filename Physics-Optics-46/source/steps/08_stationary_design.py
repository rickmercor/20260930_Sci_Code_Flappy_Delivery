"""
Select the least-noise admissible design over the two-parameter coupling and detuning rectangle.

With fixed detector-noise variance, the exact objective is the channel-weighted inverse-Gram variance $$F(q,s)=\sum_nw_n[(A^{\mathsf T}A)^+]_{nn}$$ of the measured intensity matrix, now a function of both the logarithmic coupling $$q$$ and the comb detuning $$s$$. Changes of either parameter alter resonant linewidths, transmission patterns and which resonances the comb samples, so the objective need not be unimodal on the design rectangle.

Detector linearity caps the frequency-averaged total measured transmittance, so the admissible designs are those with $$T_0(q,s)\le T_{\max}$$, where $$T_0$$ is the average over frequency channels of the summed measured intensities. Designs on the constraint surface, at which $$T_0=T_{\max}$$ exactly, are admissible by definition; they are not re-tested against the limit in floating-point arithmetic.

The returned design is labelled by whether the throughput limit binds at it and by which of its coordinates, if any, sit on a bound of the rectangle. Each supported instance comes with its scan, and the guarantee holds for that pairing: the admissible minimum over the whole rectangle is unique, lies at least one percent in objective value below every other admissible candidate, and is found by a search started from the supplied scan nodes. A coarser scan than the one supplied carries no such guarantee.

Returns
-------
np.ndarray: Real vector [q_star, s_star, F_star, active, q_pinned, s_pinned], shape (6,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stationary_design(cavity: "np.ndarray", frequencies: "np.ndarray", weights: "np.ndarray", bounds: "np.ndarray", throughput_limit: float, scan_points: "np.ndarray") -> "np.ndarray":
    r"""Find the minimum admissible noise amplification over the design rectangle.
    
    Parameters
    ----------
    cavity : np.ndarray
        Cavity array (P,P+M+1), as in scattering_jet.
    frequencies : np.ndarray
        Finite real vector of N > M channels, before detuning. At every design
        in the rectangle the intensity matrix is full row rank, meets
        inverse_gram_jet's conditioning bound, and its shifted resolvents are
        nonsingular.
    weights : np.ndarray
        Finite nonnegative channel weights of length N, as in inverse_gram_jet.
    bounds : np.ndarray
        Real array [[q_lower,q_upper],[s_lower,s_upper]], each lower < upper.
    throughput_limit : float
        Positive detector throughput ceiling T_max. At least one admissible
        design exists in the rectangle.
    scan_points : np.ndarray
        Two integers >= 3, the numbers of uniformly spaced endpoint-inclusive
        nodes along the coupling and the detuning axis. Each supported
        instance is supplied with a scan for which the outcome conditions in
        the background hold; a coarser scan carries no such guarantee.
    
    Returns
    -------
    result : np.ndarray
        Real vector [q_star, s_star, F_star, active, q_pinned, s_pinned],
        shape (6,). The fourth entry is 1.0 when the throughput at the selected
        design equals the limit and 0.0 otherwise; the last two are 1.0 when the
        corresponding design coordinate sits on a bound of the rectangle and 0.0
        otherwise.
    
    Notes
    -----
    Compose scattering_jet, transmission_jet and inverse_gram_jet using their
    returned values. The throughput and its design derivatives come from the
    same intensity jet, each order being the average over frequency channels of
    the summed measured intensities of the corresponding intensity derivative,
    which matches the throughput convention of spectral_correlation. Inputs
    satisfy the stated domain and must remain unchanged.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _split(jet):
    """Gradient and Hessian from a six-component jet."""
    return np.array([jet[1], jet[2]]), np.array([[jet[3], jet[4]], [jet[4], jet[5]]])

def _inside(point, bounds):
    return (bounds[0][0] <= point[0] <= bounds[0][1]) and (bounds[1][0] <= point[1] <= bounds[1][1])

def _newton(residual, start, bounds, tolerance):
    """Damped-free Newton confined to the rectangle; None if it leaves or stalls."""
    point = np.array(start, dtype=float)
    for _ in range(60):
        values, jacobian = residual(point)
        try:
            step = np.linalg.solve(jacobian, values)
        except np.linalg.LinAlgError:
            return None
        point = point-step
        if not _inside(point, bounds):
            return None
        if np.max(np.abs(step)) < 1e-14:
            break
    if np.max(np.abs(residual(point)[0])) > tolerance:
        return None
    return point

def _distinct(points, tolerance=1e-6):
    unique = []
    for point in points:
        if not any(np.max(np.abs(point-other)) < tolerance for other in unique):
            unique.append(point)
    return unique

def _oracle_stationary_design(cavity: "np.ndarray", frequencies: "np.ndarray", weights: "np.ndarray", bounds: "np.ndarray", throughput_limit: float, scan_points: "np.ndarray") -> "np.ndarray":
    def _jets(point):
        scattering = _oracle_scattering_jet(cavity, frequencies, float(point[0]), float(point[1]))
        transmission = _oracle_transmission_jet(scattering)
        return _oracle_inverse_gram_jet(transmission, weights), transmission.sum(axis=1).mean(axis=1)

    def _stationary(point):
        gradient, hessian = _split(_jets(point)[0])
        return gradient, hessian

    def _constrained(point):
        objective, throughput = _jets(point)
        grad_f, hess_f = _split(objective)
        grad_t, hess_t = _split(throughput)
        parallel = grad_f[0]*grad_t[1]-grad_f[1]*grad_t[0]
        row = np.array([hess_f[0, 0]*grad_t[1]+grad_f[0]*hess_t[1, 0]-hess_f[1, 0]*grad_t[0]-grad_f[1]*hess_t[0, 0],
                        hess_f[0, 1]*grad_t[1]+grad_f[0]*hess_t[1, 1]-hess_f[1, 1]*grad_t[0]-grad_f[1]*hess_t[0, 1]])
        return np.array([throughput[0]-throughput_limit, parallel]), np.array([grad_t, row])

    nodes = [np.linspace(bounds[i][0], bounds[i][1], int(scan_points[i])) for i in (0, 1)]
    interior, constrained, edge = [], [], []
    for first in nodes[0]:
        for second in nodes[1]:
            start = (float(first), float(second))
            found = _newton(_stationary, start, bounds, 1e-8)
            if found is not None:
                interior.append(found)
            found = _newton(_constrained, start, bounds, 1e-7)
            if found is not None:
                constrained.append(found)
    for axis in (0, 1):
        free = 1-axis
        for fixed in bounds[axis]:
            def _at(value):
                point = np.empty(2)
                point[axis] = fixed
                point[free] = value
                return point
            # Edges are one-dimensional and cheap, so subdivide each scan cell before bracketing.
            line = np.linspace(bounds[free][0], bounds[free][1], 8*(len(nodes[free])-1)+1)
            slopes = [_split(_jets(_at(value))[0])[0][free] for value in line]
            levels = [_jets(_at(value))[1][0]-throughput_limit for value in line]
            for i in range(len(line)-1):
                if slopes[i]*slopes[i+1] < 0:
                    edge.append(_at(brentq(lambda v: _split(_jets(_at(v))[0])[0][free], line[i], line[i+1], xtol=1e-13)))
                if levels[i]*levels[i+1] < 0:
                    edge.append(_at(brentq(lambda v: _jets(_at(v))[1][0]-throughput_limit, line[i], line[i+1], xtol=1e-13)))
    corners = [np.array([a, b]) for a in bounds[0] for b in bounds[1]]
    bounds = np.asarray(bounds, dtype=float)
    best = None
    for kind, points in (("interior", _distinct(interior)), ("constrained", _distinct(constrained)),
                         ("edge", _distinct(edge)), ("corner", corners)):
        for point in points:
            objective, throughput = _jets(point)
            on_limit = kind == "constrained" or abs(throughput[0]-throughput_limit) <= 1e-12
            if not (on_limit or throughput[0] <= throughput_limit):
                continue
            if best is None or objective[0] < best[0]:
                best = (objective[0], point, 1.0 if on_limit else 0.0)
    pinned = [1.0 if min(abs(best[1][i]-bounds[i][0]), abs(best[1][i]-bounds[i][1])) <= 1e-12 else 0.0
              for i in (0, 1)]
    return np.array([best[1][0], best[1][1], best[0], best[2], pinned[0], pinned[1]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit differential cases for Studio contract validation."""
    return [
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(5)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.array([0.,1,1,1,1,1,1,1,1,1,0,0.])\nbounds=np.array([[-0.3,0.9],[-0.25,0.25]]); limit=0.7; scan=np.array([7,7])\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy(); w=weights.copy(); b=bounds.copy(); sc=scan.copy()\n    out=np.asarray(fn(c,f,w,b,limit,sc))\n    assert out.shape==(6,) and np.array_equal(c,cavity) and np.array_equal(f,freq)\n    assert np.array_equal(w,weights) and np.array_equal(b,bounds) and np.array_equal(sc,scan)\n    return out\n',
            "call": "measure(stationary_design)",
            "gold_call": "measure(_oracle_stationary_design)",
            "tol": 2e-06
        },
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(3)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.array([0.,1,1,1,1,1,1,1,1,1,0,0.])\nbounds=np.array([[-0.3,0.9],[-0.1,0.1]]); limit=0.95; scan=np.array([7,5])\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy(); w=weights.copy(); b=bounds.copy(); sc=scan.copy()\n    out=np.asarray(fn(c,f,w,b,limit,sc))\n    assert out.shape==(6,) and np.array_equal(c,cavity) and np.array_equal(f,freq)\n    assert np.array_equal(w,weights) and np.array_equal(b,bounds) and np.array_equal(sc,scan)\n    return out\n',
            "call": "measure(stationary_design)",
            "gold_call": "measure(_oracle_stationary_design)",
            "tol": 2e-06
        },
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(3)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.array([0.,1,1,1,1,1,1,1,1,1,0,0.])\nbounds=np.array([[-0.3,0.9],[-0.1,0.1]]); limit=0.75; scan=np.array([7,5])\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy(); w=weights.copy(); b=bounds.copy(); sc=scan.copy()\n    out=np.asarray(fn(c,f,w,b,limit,sc))\n    assert out.shape==(6,) and np.array_equal(c,cavity) and np.array_equal(f,freq)\n    assert np.array_equal(w,weights) and np.array_equal(b,bounds) and np.array_equal(sc,scan)\n    return out\n',
            "call": "measure(stationary_design)",
            "gold_call": "measure(_oracle_stationary_design)",
            "tol": 2e-06
        },
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(7)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.ones(n)\nbounds=np.array([[-0.3,0.9],[-0.1,0.1]]); limit=0.72; scan=np.array([7,5])\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy(); w=weights.copy(); b=bounds.copy(); sc=scan.copy()\n    out=np.asarray(fn(c,f,w,b,limit,sc))\n    assert out.shape==(6,) and np.array_equal(c,cavity) and np.array_equal(f,freq)\n    assert np.array_equal(w,weights) and np.array_equal(b,bounds) and np.array_equal(sc,scan)\n    return out\n',
            "call": "measure(stationary_design)",
            "gold_call": "measure(_oracle_stationary_design)",
            "tol": 2e-06
        },
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(5)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.array([0.,1,1,1,1,1,1,1,1,1,0,0.])\nbounds=np.array([[0.2,0.6],[-0.05,0.05]]); limit=0.95; scan=np.array([5,4])\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy(); w=weights.copy(); b=bounds.copy(); sc=scan.copy()\n    out=np.asarray(fn(c,f,w,b,limit,sc))\n    assert out.shape==(6,) and np.array_equal(c,cavity) and np.array_equal(f,freq)\n    assert np.array_equal(w,weights) and np.array_equal(b,bounds) and np.array_equal(sc,scan)\n    return out\n',
            "call": "measure(stationary_design)",
            "gold_call": "measure(_oracle_stationary_design)",
            "tol": 2e-06
        },
        {
            "setup": 'import numpy as np\np=8;m=3;n=12\nr=np.random.default_rng(3)\nx=r.standard_normal((p,p));y=r.standard_normal((p,p));v=r.standard_normal((p,m+1))\nh=(x+x.T+1j*(y-y.T))/(2*np.sqrt(p))\ncavity=np.concatenate((h,v/np.sqrt(p)),axis=1)\nfreq=np.linspace(-.7,.7,n)\nweights=np.zeros(12); weights[4:6]=1.\nbounds=np.array([[-0.3,0.9],[-0.1,0.1]]); limit=0.75; scan=np.array([7,5])\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy(); w=weights.copy(); b=bounds.copy(); sc=scan.copy()\n    out=np.asarray(fn(c,f,w,b,limit,sc))\n    assert out.shape==(6,) and np.array_equal(c,cavity) and np.array_equal(f,freq)\n    assert np.array_equal(w,weights) and np.array_equal(b,bounds) and np.array_equal(sc,scan)\n    return out\n',
            "call": "measure(stationary_design)",
            "gold_call": "measure(_oracle_stationary_design)",
            "tol": 2e-06
        }
    ]
