#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
def _joint_has_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_joint_has_bool(v) for v in value)
    return isinstance(value, np.ndarray) and value.dtype.kind == 'b'
def _joint_array(value, shape, complex_ok=False):
    try:
        raw = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid numeric array') from exc
    if (_joint_has_bool(value) or raw.dtype.kind not in ('iufc' if complex_ok else 'iuf') or raw.shape != shape):
        raise ValueError('invalid numeric type or shape')
    out = np.asarray(raw, dtype=complex if complex_ok else float)
    if not np.all(np.isfinite(out)):
        raise ValueError('nonfinite input')
    return out
def _joint_scalar(value, lo=None, hi=None):
    out = float(_joint_array(value, ()))
    if (lo is not None and out < lo) or (hi is not None and out > hi):
        raise ValueError('scalar outside public domain')
    return out
def gluon_beam_density(x: float, azimuth: float) -> 'np.ndarray':
    try:
        x = _joint_scalar(x, .1, .9)
        azimuth = _joint_scalar(azimuth, -np.pi, np.pi)
    except ValueError as exc:
        raise ValueError('invalid beam-density input') from exc
    u = np.array([np.cos(azimuth), np.sin(azimuth)])
    p = 6*((x/(1-x)+(1-x)/x)*np.eye(2) + 2*x*(1-x)*np.outer(u, u))
    return p/np.trace(p)

import numpy as np
from numpy.typing import ArrayLike
def _joint_has_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_joint_has_bool(v) for v in value)
    return isinstance(value, np.ndarray) and value.dtype.kind == 'b'
def _joint_array(value, shape, complex_ok=False):
    try:
        raw = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid numeric array') from exc
    if (_joint_has_bool(value) or raw.dtype.kind not in ('iufc' if complex_ok else 'iuf') or raw.shape != shape):
        raise ValueError('invalid numeric type or shape')
    out = np.asarray(raw, dtype=complex if complex_ok else float)
    if not np.all(np.isfinite(out)):
        raise ValueError('nonfinite input')
    return out
def hard_spin_color_density(coefficients: 'ArrayLike',
                                    beam_density: 'ArrayLike') -> 'np.ndarray':
    from fractions import Fraction as F
    c = _joint_array(coefficients, (2, 2, 3), True)
    beam = _joint_array(beam_density, (2, 2))
    if not np.array_equal(beam, beam.T):
        raise ValueError('beam density must be real symmetric')
    s = [[F(float(v)) for v in row] for row in beam]
    if s[0][0] <= 0 or s[0][0]*s[1][1]-s[0][1]**2 <= 0:
        raise ValueError('beam density must be positive definite')
    # Exact products of the supplied binary floats avoid intermediate overflow
    # and scale-dependent loss in this small, normalized positive quadratic form.
    h = {}
    for p, q, a, b in np.ndindex(2, 2, 3, 3):
        real = imag = F(0)
        for r, t in np.ndindex(2, 2):
            u, v = c[p, r, a], c[q, t, b]
            ur, ui, vr, vi = map(lambda x: F(float(x)), (u.real, u.imag, v.real, v.imag))
            real += s[r][t]*(ur*vr+ui*vi)
            imag += s[r][t]*(ui*vr-ur*vi)
        h[p, q, a, b] = real, imag
    gram = [[F(64, 9), F(-8, 9), F(-8, 9)],
            [F(-8, 9), F(64, 9), F(1, 9)],
            [F(-8, 9), F(1, 9), F(64, 9)]]
    norm = sum(h[p, p, a, b][0]*gram[b][a]
               for p, a, b in np.ndindex(2, 3, 3))
    if norm <= 0:
        raise ValueError('zero hard amplitude')
    out = np.empty((2, 2, 3, 3), complex)
    for index, (real, imag) in h.items():
        out[index] = complex(float(real/norm), float(imag/norm))
    return out

import numpy as np
from numpy.typing import ArrayLike
def _joint_has_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_joint_has_bool(v) for v in value)
    return isinstance(value, np.ndarray) and value.dtype.kind == 'b'
def _joint_array(value, shape, complex_ok=False):
    try:
        raw = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid numeric array') from exc
    if (_joint_has_bool(value) or raw.dtype.kind not in ('iufc' if complex_ok else 'iuf') or raw.shape != shape):
        raise ValueError('invalid numeric type or shape')
    out = np.asarray(raw, dtype=complex if complex_ok else float)
    if not np.all(np.isfinite(out)):
        raise ValueError('nonfinite input')
    return out
def _joint_finite_fraction(value):
    try:
        answer = float(value)
    except OverflowError as exc:
        raise ValueError('result not representable as finite float64') from exc
    if not np.isfinite(answer):
        raise ValueError('result not representable as finite float64')
    return answer
def color_spin_response(hard_density: 'ArrayLike') -> 'np.ndarray':
    from fractions import Fraction
    try:
        h = _joint_array(hard_density, (2, 2, 3, 3), True)
    except ValueError as exc:
        raise ValueError('invalid hard-density input') from exc
    real_l = [
        [[0, -1, 0], [1, 0, Fraction(-1, 8)], [0, Fraction(1, 8), 0]],
        [[0, Fraction(-1, 8), Fraction(7, 4)], [Fraction(1, 8), 0, Fraction(-3, 8)],
         [Fraction(-7, 4), Fraction(3, 8), 0]],
        [[0, Fraction(9, 8), Fraction(-7, 4)], [Fraction(-9, 8), 0, Fraction(1, 2)],
         [Fraction(7, 4), Fraction(-1, 2), 0]],
    ]
    out = np.empty((3, 2, 2), complex)
    for j, p, q in np.ndindex(3, 2, 2):
        real = imag = Fraction(0)
        for a, b in np.ndindex(3, 3):
            coefficient = real_l[j][a][b]
            real -= coefficient*Fraction(float(h[p, q, a, b].imag))
            imag += coefficient*Fraction(float(h[p, q, a, b].real))
        out[j, p, q] = complex(_joint_finite_fraction(real), _joint_finite_fraction(imag))
    return out

import numpy as np
from numpy.typing import ArrayLike
def _r11_has_bool(value):
    if isinstance(value,(bool,np.bool_)):
        return True
    if isinstance(value,(tuple,list)):
        return any(_r11_has_bool(x) for x in value)
    return isinstance(value,np.ndarray) and value.dtype.kind=='b'
def _r11_array(value,shape=None,complex_ok=False):
    try:
        a=np.asarray(value)
    except (TypeError,ValueError) as exc:
        raise ValueError('invalid numeric input') from exc
    if _r11_has_bool(value) or a.dtype.kind not in ('iufc' if complex_ok else 'iuf'):
        raise ValueError('invalid numeric type')
    if shape is not None and a.shape!=shape:
        raise ValueError('invalid shape')
    a=a.astype(complex if complex_ok else float)
    if not np.all(np.isfinite(a)):
        raise ValueError('nonfinite input')
    return a
def _r11_scalar(value,lo,hi):
    x=float(_r11_array(value,()))
    if not lo<=x<=hi:
        raise ValueError('outside public interval')
    return x
def _r11_angles(eta,phi,jet_eta,jet_phi):
    e=_r11_array(eta); p=_r11_array(phi)
    if e.ndim>2 or p.ndim>2 or not e.size or not p.size:
        raise ValueError('radiation input must be nonempty of dimension at most two')
    try:
        e,p=np.broadcast_arrays(e,p)
    except ValueError as exc:
        raise ValueError('radiation inputs must broadcast') from exc
    if np.any(abs(e)>.7) or np.any(abs(p)>2*np.pi):
        raise ValueError('radiation input outside domain')
    y=_r11_scalar(jet_eta,-2.5,2.5)
    if abs(y)<1:
        raise ValueError('hard direction outside separated domain')
    a=_r11_scalar(jet_phi,-2*np.pi,2*np.pi)
    return e,p,y,a
def _r11_st(t):
    return t-np.trace(t,axis1=-2,axis2=-1)[...,None,None]*np.eye(2)/2
def _r11_transverse(phi,jet_phi):
    u=np.stack([np.cos(phi),np.sin(phi)],axis=-1)
    v=np.array([np.cos(jet_phi),np.sin(jet_phi)])
    uu=np.einsum('...a,...b->...ab',u,u)
    cross=np.einsum('...a,b->...ab',u,v)+np.einsum('a,...b->...ab',v,u)
    return uu,cross
def _r11_fraction_float(value):
    try:
        x=float(value)
    except OverflowError as exc:
        raise ValueError('output not finite float64') from exc
    if not np.isfinite(x):
        raise ValueError('output not finite float64')
    return x
def dimensional_spin_kernel(eta: 'float | ArrayLike',
                                    phi: 'float | ArrayLike',
                                    jet_eta: float,
                                    jet_phi: float,
                                    radius: float) -> 'np.ndarray':
    try:
        e,p,y,a=_r11_angles(eta,phi,jet_eta,jet_phi)
        r=_r11_scalar(radius,0.,1.)
    except ValueError as exc:
        raise ValueError('invalid dimensional-kernel input') from exc
    c=np.cosh(e-y); q=np.cos(p-a)
    uu,cross=_r11_transverse(p,a)
    factor=(np.cosh(e)**2/(c-q*r))[...,None,None]
    b0=factor*(-2*c[...,None,None]*r*r*uu+r*cross)
    b1=factor*4*np.sinh(e-y)[...,None,None]*r*r*uu
    return np.stack([_r11_st(b0),_r11_st(b1)],axis=-3)

import numpy as np
def _r11_has_bool(value):
    if isinstance(value,(bool,np.bool_)):
        return True
    if isinstance(value,(tuple,list)):
        return any(_r11_has_bool(x) for x in value)
    return isinstance(value,np.ndarray) and value.dtype.kind=='b'
def _r11_array(value,shape=None,complex_ok=False):
    try:
        a=np.asarray(value)
    except (TypeError,ValueError) as exc:
        raise ValueError('invalid numeric input') from exc
    if _r11_has_bool(value) or a.dtype.kind not in ('iufc' if complex_ok else 'iuf'):
        raise ValueError('invalid numeric type')
    if shape is not None and a.shape!=shape:
        raise ValueError('invalid shape')
    a=a.astype(complex if complex_ok else float)
    if not np.all(np.isfinite(a)):
        raise ValueError('nonfinite input')
    return a
def _r11_scalar(value,lo,hi):
    x=float(_r11_array(value,()))
    if not lo<=x<=hi:
        raise ValueError('outside public interval')
    return x
def _r11_angles(eta,phi,jet_eta,jet_phi):
    e=_r11_array(eta); p=_r11_array(phi)
    if e.ndim>2 or p.ndim>2 or not e.size or not p.size:
        raise ValueError('radiation input must be nonempty of dimension at most two')
    try:
        e,p=np.broadcast_arrays(e,p)
    except ValueError as exc:
        raise ValueError('radiation inputs must broadcast') from exc
    if np.any(abs(e)>.7) or np.any(abs(p)>2*np.pi):
        raise ValueError('radiation input outside domain')
    y=_r11_scalar(jet_eta,-2.5,2.5)
    if abs(y)<1:
        raise ValueError('hard direction outside separated domain')
    a=_r11_scalar(jet_phi,-2*np.pi,2*np.pi)
    return e,p,y,a
def _r11_st(t):
    return t-np.trace(t,axis1=-2,axis2=-1)[...,None,None]*np.eye(2)/2
def _r11_transverse(phi,jet_phi):
    u=np.stack([np.cos(phi),np.sin(phi)],axis=-1)
    v=np.array([np.cos(jet_phi),np.sin(jet_phi)])
    uu=np.einsum('...a,...b->...ab',u,u)
    cross=np.einsum('...a,b->...ab',u,v)+np.einsum('a,...b->...ab',v,u)
    return uu,cross
def _r11_fraction_float(value):
    try:
        x=float(value)
    except OverflowError as exc:
        raise ValueError('output not finite float64') from exc
    if not np.isfinite(x):
        raise ValueError('output not finite float64')
    return x
def finite_glauber_prefactors(z: float,
                                      scale_ratio: float) -> 'np.ndarray':
    try:
        z=_r11_scalar(z,.1,.9)
        ratio=_r11_scalar(scale_ratio,.2,5.)
    except ValueError as exc:
        raise ValueError('invalid prefactor input') from exc
    pole=(1-z)/(18*np.pi**2*z)
    return np.array([pole,pole*(4+6*np.log(ratio))])

import numpy as np
from numpy.typing import ArrayLike
def _r11_has_bool(value):
    if isinstance(value,(bool,np.bool_)):
        return True
    if isinstance(value,(tuple,list)):
        return any(_r11_has_bool(x) for x in value)
    return isinstance(value,np.ndarray) and value.dtype.kind=='b'
def _r11_array(value,shape=None,complex_ok=False):
    try:
        a=np.asarray(value)
    except (TypeError,ValueError) as exc:
        raise ValueError('invalid numeric input') from exc
    if _r11_has_bool(value) or a.dtype.kind not in ('iufc' if complex_ok else 'iuf'):
        raise ValueError('invalid numeric type')
    if shape is not None and a.shape!=shape:
        raise ValueError('invalid shape')
    a=a.astype(complex if complex_ok else float)
    if not np.all(np.isfinite(a)):
        raise ValueError('nonfinite input')
    return a
def _r11_scalar(value,lo,hi):
    x=float(_r11_array(value,()))
    if not lo<=x<=hi:
        raise ValueError('outside public interval')
    return x
def _r11_angles(eta,phi,jet_eta,jet_phi):
    e=_r11_array(eta); p=_r11_array(phi)
    if e.ndim>2 or p.ndim>2 or not e.size or not p.size:
        raise ValueError('radiation input must be nonempty of dimension at most two')
    try:
        e,p=np.broadcast_arrays(e,p)
    except ValueError as exc:
        raise ValueError('radiation inputs must broadcast') from exc
    if np.any(abs(e)>.7) or np.any(abs(p)>2*np.pi):
        raise ValueError('radiation input outside domain')
    y=_r11_scalar(jet_eta,-2.5,2.5)
    if abs(y)<1:
        raise ValueError('hard direction outside separated domain')
    a=_r11_scalar(jet_phi,-2*np.pi,2*np.pi)
    return e,p,y,a
def finite_veto_tensors(jet_eta: 'ArrayLike',
                                jet_phi: 'ArrayLike',
                                gap: 'ArrayLike',
                                azimuth_window: 'ArrayLike') -> 'np.ndarray':
    from scipy.special import roots_legendre

    jets_eta = _r11_array(jet_eta, (3,))
    jets_phi = _r11_array(jet_phi, (3,))
    gap = _r11_array(gap, (2,))
    azimuth_window = _r11_array(azimuth_window, (2,))
    if np.any(abs(gap) > 0.7) or gap[0] >= gap[1]:
        raise ValueError("gap must be increasing in [-0.7,0.7]")
    if np.any(abs(azimuth_window) > 2 * np.pi) or not 0 < azimuth_window[1] - azimuth_window[0] <= 2 * np.pi:
        raise ValueError("azimuth endpoints must lie in [-2pi,2pi], width in (0,2pi]")
    for one_eta, one_phi in zip(jets_eta, jets_phi):
        _r11_angles(0.0, 0.0, one_eta, one_phi)
    nodes, weights = roots_legendre(72)
    eta = gap[0] + (nodes + 1) * (gap[1] - gap[0]) / 2
    phi = azimuth_window[0] + (nodes + 1) * (azimuth_window[1] - azimuth_window[0]) / 2
    eta, phi = np.meshgrid(eta, phi, indexing="ij")
    measure = np.outer(weights, weights) * (gap[1] - gap[0]) * (azimuth_window[1] - azimuth_window[0]) / (16 * np.pi * np.cosh(eta) ** 2)
    first = []
    second = []
    third = []
    radial_nodes, radial_weights = roots_legendre(48)
    radial_nodes = (radial_nodes + 1) / 2
    radial_weights = radial_weights / 2
    for one_eta, one_phi in zip(jets_eta, jets_phi):
        kernel = dimensional_spin_kernel(eta, phi, one_eta, one_phi, 1.0)
        first.append(np.einsum("ij,ijab->ab", measure, kernel[:, :, 0]))
        second.append(np.einsum("ij,ijab->ab", measure, kernel[:, :, 1]))
        endpoint = kernel[:, :, 0]
        evanescent = np.zeros_like(endpoint)
        for radius, weight in zip(radial_nodes, radial_weights):
            radial = dimensional_spin_kernel(eta, phi, one_eta, one_phi, radius)[:, :, 0]
            evanescent += weight * (-2 * radius) * (radial - endpoint) / (1 - radius * radius)
        third.append(np.einsum("ij,ijab->ab", measure, evanescent))
    return np.stack((np.array(first), np.array(second), np.array(third)), axis=1)

import numpy as np
from numpy.typing import ArrayLike
def _r11_has_bool(value):
    if isinstance(value,(bool,np.bool_)):
        return True
    if isinstance(value,(tuple,list)):
        return any(_r11_has_bool(x) for x in value)
    return isinstance(value,np.ndarray) and value.dtype.kind=='b'
def _r11_array(value,shape=None,complex_ok=False):
    try:
        a=np.asarray(value)
    except (TypeError,ValueError) as exc:
        raise ValueError('invalid numeric input') from exc
    if _r11_has_bool(value) or a.dtype.kind not in ('iufc' if complex_ok else 'iuf'):
        raise ValueError('invalid numeric type')
    if shape is not None and a.shape!=shape:
        raise ValueError('invalid shape')
    a=a.astype(complex if complex_ok else float)
    if not np.all(np.isfinite(a)):
        raise ValueError('nonfinite input')
    return a
def _r11_scalar(value,lo,hi):
    x=float(_r11_array(value,()))
    if not lo<=x<=hi:
        raise ValueError('outside public interval')
    return x
def _r11_angles(eta,phi,jet_eta,jet_phi):
    e=_r11_array(eta); p=_r11_array(phi)
    if e.ndim>2 or p.ndim>2 or not e.size or not p.size:
        raise ValueError('radiation input must be nonempty of dimension at most two')
    try:
        e,p=np.broadcast_arrays(e,p)
    except ValueError as exc:
        raise ValueError('radiation inputs must broadcast') from exc
    if np.any(abs(e)>.7) or np.any(abs(p)>2*np.pi):
        raise ValueError('radiation input outside domain')
    y=_r11_scalar(jet_eta,-2.5,2.5)
    if abs(y)<1:
        raise ValueError('hard direction outside separated domain')
    a=_r11_scalar(jet_phi,-2*np.pi,2*np.pi)
    return e,p,y,a
def _r11_st(t):
    return t-np.trace(t,axis1=-2,axis2=-1)[...,None,None]*np.eye(2)/2
def _r11_transverse(phi,jet_phi):
    u=np.stack([np.cos(phi),np.sin(phi)],axis=-1)
    v=np.array([np.cos(jet_phi),np.sin(jet_phi)])
    uu=np.einsum('...a,...b->...ab',u,u)
    cross=np.einsum('...a,b->...ab',u,v)+np.einsum('a,...b->...ab',v,u)
    return uu,cross
def _r11_fraction_float(value):
    try:
        x=float(value)
    except OverflowError as exc:
        raise ValueError('output not finite float64') from exc
    if not np.isfinite(x):
        raise ValueError('output not finite float64')
    return x
def finite_attachment_channels(response: 'ArrayLike',
                                       angular: 'ArrayLike',
                                       prefactors: 'ArrayLike') -> 'np.ndarray':
    from fractions import Fraction as F
    try:
        k=_r11_array(response,(3,2,2),True)
        t=_r11_array(angular,(3,3,2,2))
        pref=_r11_array(prefactors,(2,))
    except ValueError as exc:
        raise ValueError('invalid attachment-channel input') from exc
    f0,f1=map(lambda x:F(float(x)),pref)
    out=np.empty((3,3))
    for j in range(3):
        kk=[[F(float(k[j,p,q].real)) for q in range(2)] for p in range(2)]
        trace=kk[0][0]+kk[1][1]
        kk[0][0]-=trace/2;kk[1][1]-=trace/2
        tt=[[[F(float(t[j,b,p,q])) for q in range(2)] for p in range(2)] for b in range(3)]
        u=[[f1*tt[0][p][q]+f0*(tt[1][p][q]+tt[2][p][q]) for q in range(2)] for p in range(2)]
        diag=200*sum(kk[p][p]*u[p][p] for p in range(2))
        coherence=200*(kk[0][1]*u[1][0]+kk[1][0]*u[0][1])
        ev=200*f0*sum(kk[p][q]*tt[2][q][p] for p,q in np.ndindex(2,2))
        out[j]=[_r11_fraction_float(x) for x in [diag,coherence,ev]]
    return out

import numpy as np
from numpy.typing import ArrayLike
def _fit_has_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (tuple, list)):
        return any(_fit_has_bool(item) for item in value)
    return isinstance(value, np.ndarray) and value.dtype.kind == "b"
def _fit_array(value, shape=None, complex_ok=False):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid numeric input") from exc
    if _fit_has_bool(value) or array.dtype.kind not in ("iufc" if complex_ok else "iuf"):
        raise ValueError("invalid numeric type")
    if shape is not None and array.shape != shape:
        raise ValueError("invalid shape")
    array = array.astype(complex if complex_ok else float)
    if not np.all(np.isfinite(array)):
        raise ValueError("nonfinite input")
    return array
def _fit_residual(state, coefficients, angular, prefactors, observations,
                  uncertainties):
    beam = gluon_beam_density(state[0], state[1])
    hard = hard_spin_color_density(coefficients, beam)
    response = color_spin_response(hard)
    predicted = np.array([
        np.sum(finite_attachment_channels(response, one, prefactors)[:, :2])
        for one in angular])
    return (state[2] * predicted - observations) / uncertainties
def infer_beam_state(coefficients: 'ArrayLike',
                             jet_eta: 'ArrayLike',
                             jet_phi: 'ArrayLike',
                             calibration_gaps: 'ArrayLike',
                             calibration_windows: 'ArrayLike',
                             observations: 'ArrayLike',
                             uncertainties: 'ArrayLike',
                             z: float,
                             scale_ratio: float,
                             x_interval: 'ArrayLike',
                             azimuth_interval: 'ArrayLike',
                             gain_interval: 'ArrayLike',
                             initial_guess: 'ArrayLike') -> 'np.ndarray':
    from scipy.optimize import least_squares
    coefficients = _fit_array(coefficients, (2, 2, 3), True)
    jet_eta = _fit_array(jet_eta, (3,))
    jet_phi = _fit_array(jet_phi, (3,))
    gaps = _fit_array(calibration_gaps)
    windows = _fit_array(calibration_windows)
    if gaps.ndim != 2 or gaps.shape[1:] != (2,) or not 4 <= gaps.shape[0] <= 12:
        raise ValueError("invalid calibration-gap shape")
    if windows.shape != gaps.shape:
        raise ValueError("calibration windows must match gaps")
    count = gaps.shape[0]
    observations = _fit_array(observations, (count,))
    uncertainties = _fit_array(uncertainties, (count,))
    if np.any(uncertainties <= 0):
        raise ValueError("uncertainties must be positive")
    x_interval = _fit_array(x_interval, (2,))
    azimuth_interval = _fit_array(azimuth_interval, (2,))
    gain_interval = _fit_array(gain_interval, (2,))
    initial_guess = _fit_array(initial_guess, (3,))
    if not (0.1 <= x_interval[0] < x_interval[1] <= 0.49):
        raise ValueError("invalid x interval")
    if not (-np.pi <= azimuth_interval[0] < azimuth_interval[1] <= np.pi):
        raise ValueError("invalid azimuth interval")
    if azimuth_interval[1] - azimuth_interval[0] > np.pi:
        raise ValueError("azimuth interval too wide")
    if not (0 < gain_interval[0] < gain_interval[1]):
        raise ValueError("invalid gain interval")
    lower = np.array([x_interval[0], azimuth_interval[0], gain_interval[0]])
    upper = np.array([x_interval[1], azimuth_interval[1], gain_interval[1]])
    if np.any(initial_guess <= lower) or np.any(initial_guess >= upper):
        raise ValueError("initial guess must be strictly interior")
    try:
        angular = [finite_veto_tensors(jet_eta, jet_phi, gaps[row], windows[row])
                   for row in range(count)]
        prefactors = finite_glauber_prefactors(z, scale_ratio)
    except ValueError as exc:
        raise ValueError("invalid calibration geometry") from exc

    fractions = np.array([[.5, .5, .5], [.25, .25, .25], [.75, .75, .75],
                          [.25, .75, .5], [.75, .25, .5]])
    starts = [initial_guess] + [lower + f * (upper - lower) for f in fractions]
    args = (coefficients, angular, prefactors, observations, uncertainties)
    fits = [least_squares(_fit_residual, start, args=args,
                          bounds=(lower, upper), xtol=1e-13,
                          ftol=1e-13, gtol=1e-13, max_nfev=1000)
            for start in starts]
    best = min(fits, key=lambda fit: float(np.dot(fit.fun, fit.fun)))
    state = np.asarray(best.x, dtype=float)
    weighted_rms = float(np.sqrt(np.mean(np.asarray(best.fun, dtype=float) ** 2)))
    singular = np.linalg.svd(np.asarray(best.jac, dtype=float), compute_uv=False)
    if singular.shape != (3,) or singular[-1] <= 1e-7 or not np.isfinite(weighted_rms):
        raise ValueError("calibration is not locally identifiable")
    near = [fit for fit in fits
            if np.sqrt(np.mean(np.asarray(fit.fun, dtype=float) ** 2)) <= weighted_rms + 1e-7]
    if any(np.linalg.norm(np.asarray(fit.x) - state, ord=np.inf) > 1e-5 for fit in near):
        raise ValueError("calibration is not unique in the box")
    return np.array([state[0], state[1], state[2], weighted_rms], dtype=float)

import numpy as np
from numpy.typing import ArrayLike
def heldout_glauber_prediction(coefficients: 'ArrayLike',
                                       jet_eta: 'ArrayLike',
                                       jet_phi: 'ArrayLike',
                                       calibration_gaps: 'ArrayLike',
                                       calibration_windows: 'ArrayLike',
                                       observations: 'ArrayLike',
                                       uncertainties: 'ArrayLike',
                                       prediction_gap: 'ArrayLike',
                                       prediction_window: 'ArrayLike',
                                       z: float,
                                       scale_ratio: float,
                                       x_interval: 'ArrayLike',
                                       azimuth_interval: 'ArrayLike',
                                       gain_interval: 'ArrayLike',
                                       initial_guess: 'ArrayLike') -> float:
    try:
        inferred = infer_beam_state(
            coefficients, jet_eta, jet_phi, calibration_gaps,
            calibration_windows, observations, uncertainties, z, scale_ratio,
            x_interval, azimuth_interval, gain_interval, initial_guess)
        beam = gluon_beam_density(inferred[0], inferred[1])
        hard = hard_spin_color_density(coefficients, beam)
        response = color_spin_response(hard)
        angular = finite_veto_tensors(
            jet_eta, jet_phi, prediction_gap, prediction_window)
        prefactors = finite_glauber_prefactors(z, scale_ratio)
        channels = finite_attachment_channels(response, angular, prefactors)
    except ValueError as exc:
        raise ValueError("invalid held-out prediction input") from exc
    answer = float(inferred[2] * np.sum(channels[:, :2]))
    if not np.isfinite(answer):
        raise ValueError("nonfinite held-out prediction")
    return answer
SCICODE_GOLD_EOF
