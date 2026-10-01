#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Finite-Fock Gaussian optics: projected generators define the retained model.

These conventional gates support, but do not constitute, the adaptive source.
"""
import numpy as np
from functools import lru_cache
from scipy.linalg import expm
def _r6_real(value, shape=None):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf" or (shape is not None and raw.shape != shape):
            raise ValueError("real numeric kind and declared shape required")
        out = raw.astype(float)
        if not np.all(np.isfinite(out)):
            raise ValueError("finite data required")
        return out
    except (TypeError, OverflowError) as exc:
        raise ValueError("real numeric data required") from exc
def _r6_integer(value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("integer required, excluding bool")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError("integer outside domain")
    return value
def _r6_scalar(value, low=None, high=None):
    value = float(_r6_real(value, ()))
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError("scalar outside domain")
    return value
@lru_cache(maxsize=39)
def _r6_bs_blocks(d):
    blocks = []
    for total in range(2*d-1):
        pairs = [(i, total-i) for i in range(d) if 0 <= total-i < d]
        indices = np.array([i*d+j for i, j in pairs])
        generator = np.zeros((len(pairs), len(pairs)), complex)
        for col, (i, j) in enumerate(pairs):
            if i+1 < d and j > 0:
                generator[col+1, col] = np.sqrt((i+1)*j)
                generator[col, col+1] = -np.sqrt((i+1)*j)
        values, vectors = np.linalg.eigh(1j*generator)
        blocks.append((indices, np.array([i for i, _ in pairs]), values, vectors))
    return blocks
def gaussian_gate(d: int, kind: str, parameters: object, columns: object = None) -> object:
    d = _r6_integer(d, 2, 40)
    parameters = _r6_real(parameters, (2,))
    if not isinstance(kind, str) or kind not in ("squeeze", "mix"):
        raise ValueError("unknown Gaussian gate")
    if columns is not None:
        raw = np.asarray(columns)
        dimension = d if kind == "squeeze" else d*d
        if raw.dtype.kind not in "iufc" or raw.ndim != 2 or raw.shape[0] != dimension or raw.shape[1] < 1:
            raise ValueError("numeric amplitude columns with gate dimension required")
        columns = raw.astype(complex)
        if not np.all(np.isfinite(columns)) or np.vdot(columns, columns).real > 1+1e-10:
            raise ValueError("finite amplitude columns with total weight at most one required")
    magnitude, phase = parameters
    if kind == "squeeze":
        _r6_scalar(magnitude, 0, 1)
        a = np.diag(np.sqrt(np.arange(1, d)), 1)
        z = magnitude*np.exp(1j*phase)
        unitary = expm((z.conjugate()*a@a-z*a.T@a.T)/2)
        return unitary if columns is None else unitary@columns
    _r6_scalar(magnitude, 0, np.pi/2)
    if magnitude == 0:
        return np.eye(d*d, dtype=complex) if columns is None else columns.copy()
    if columns is not None:
        # Apply the generator directly. Spectral reconstruction can introduce
        # a spurious component even when the input is exactly in its kernel;
        # subsequent rare-event conditioning would magnify that component.
        steps = max(1, int(np.ceil(2*d*magnitude)))
        h = magnitude/steps
        roots = np.sqrt(np.arange(1, d))
        weights = roots[:, None, None]*roots[None, :, None]
        forward = np.exp(1j*phase)*weights
        backward = np.exp(-1j*phase)*weights
        result = columns.reshape(d, d, -1).copy()
        for _ in range(steps):
            term = result.copy()
            moved = result.copy()
            for order in range(1, 25):
                derivative = np.zeros_like(term)
                derivative[1:, :-1] += forward*term[:-1, 1:]
                derivative[:-1, 1:] -= backward*term[1:, :-1]
                term = (h/order)*derivative
                moved += term
                if not np.any(term):
                    break
            result = moved
        return result.reshape(columns.shape)
    result = np.zeros((d*d, d*d), complex)
    for ids, ns, values, vectors in _r6_bs_blocks(d):
        phases = np.exp(1j*phase)**ns
        block = (vectors*np.exp(-1j*magnitude*values))@vectors.conj().T
        result[np.ix_(ids, ids)] = phases[:, None]*block*phases.conj()[None, :]
    return result

"""Coherent vacuum attenuation on one mode of a retained subsystem."""
import math
import numpy as np
def _r7_numeric(value, shape=None, real=False):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in ('iuf' if real else 'iufc'):
            raise ValueError('numeric kind required')
        if shape is not None and raw.shape != shape:
            raise ValueError('shape mismatch')
        if not np.all(np.isfinite(raw)):
            raise ValueError('finite numeric values required')
        result = raw.astype(float if real else complex)
        if not np.all(np.isfinite(result)):
            raise ValueError('finite double-precision data required')
        return result
    except (TypeError, OverflowError) as exc:
        raise ValueError('numeric data required') from exc
def _r7_int(value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('integer required, excluding bool')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError('integer outside domain')
    return value
def _r7_scalar(value, low=None, high=None):
    value = float(_r7_numeric(value, (), True))
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError('scalar outside domain')
    return value
def _r7_bool(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('boolean required')
    return bool(value)
def _r7_dimension(size, modes):
    for d in range(2, 41):
        if d**modes == size:
            return d
    raise ValueError('dimension is not a supported tensor power')
def _r7_state_parts(value, size=None):
    rho = _r7_numeric(value)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2:
        raise ValueError('square state required')
    if size is not None and rho.shape != (size, size):
        raise ValueError('state dimension mismatch')
    if np.max(np.abs(rho-rho.conj().T)) > 1e-10:
        raise ValueError('Hermitian state required')
    h = (rho+rho.conj().T)/2
    trace = math.fsum(float(value) for value in h.diagonal().real)
    if trace < -1e-10 or trace > 1+1e-10:
        raise ValueError('state trace outside domain')
    scale = float(np.max(np.abs(h)))
    if scale == 0:
        return np.zeros_like(h), 0.
    try:
        values, vectors = np.linalg.eigh(h.real/scale+1j*(h.imag/scale))
    except np.linalg.LinAlgError as exc:
        raise ValueError('physical state required') from exc
    if values.min()*scale < -1e-10:
        raise ValueError('positive state required')
    mass = max(0., trace)
    if mass == 0:
        return np.zeros_like(h), 0.
    positive = np.maximum(values, 0.)
    root = vectors*np.sqrt(positive/positive.sum())
    return root, mass
def _r7_factor_parts(value, size):
    w = _r7_numeric(value)
    if w.ndim != 2 or w.shape[0] != size or w.shape[1] < 1:
        raise ValueError('amplitude-column shape mismatch')
    scale = float(np.max(np.abs(w)))
    if scale == 0:
        return w, 0.
    if scale > np.sqrt(1+1e-10):
        raise ValueError('amplitude mass exceeds one')
    scaled = w.real/scale+1j*(w.imag/scale)
    norm = float(np.sqrt(np.vdot(scaled, scaled).real))
    amplitude = scale*norm
    if amplitude**2 > 1+1e-10:
        raise ValueError('amplitude mass exceeds one')
    return scaled/norm, amplitude
def _r7_counts(value, length=None, distinct=False):
    raw = np.asarray(value, dtype=object)
    if raw.ndim != 1 or not 1 <= len(raw) <= 2:
        raise ValueError('one or two counts required')
    if length is not None and len(raw) != length:
        raise ValueError('count length mismatch')
    result = [_r7_int(v, 0) for v in raw]
    if distinct and len(set(result)) != len(result):
        raise ValueError('distinct first records required')
    return result
def _r7_target(value, d):
    target = _r7_numeric(value, (d,))
    if abs(np.vdot(target, target).real-1) > 1e-10:
        raise ValueError('normalized target required')
    return target
def attenuate(rho: object, eta: float, mode: int = 0, modes: int = 1) -> object:
    modes = _r7_int(modes, 1, 2)
    mode = _r7_int(mode, 0, modes-1)
    eta = _r7_scalar(eta, 0, 1)
    raw = _r7_numeric(rho)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError('square state required')
    d = _r7_dimension(raw.shape[0], modes)
    root, mass = _r7_state_parts(raw)
    if mass == 0:
        return np.zeros_like(raw)
    state = (root@root.conj().T).reshape((d,)*(2*modes))
    permutation = [mode, modes+mode]+[i for i in range(2*modes) if i not in (mode, modes+mode)]
    arranged = state.transpose(permutation)
    result = np.zeros_like(arranged)
    for ell in range(d):
        coefficients = np.array([math.sqrt(math.comb(i+ell, ell)*(1-eta)**ell*eta**i) for i in range(d-ell)])
        shape = (d-ell, d-ell)+(1,)*(2*modes-2)
        result[:d-ell, :d-ell] += arranged[ell:, ell:]*(coefficients[:, None]*coefficients[None, :]).reshape(shape)
    return mass*result.transpose(np.argsort(permutation)).reshape(raw.shape)

"""Destructive number-resolved measurement with a surviving subsystem."""
import math
import numpy as np
def _r7_numeric(value, shape=None, real=False):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in ('iuf' if real else 'iufc'):
            raise ValueError('numeric kind required')
        if shape is not None and raw.shape != shape:
            raise ValueError('shape mismatch')
        if not np.all(np.isfinite(raw)):
            raise ValueError('finite numeric values required')
        result = raw.astype(float if real else complex)
        if not np.all(np.isfinite(result)):
            raise ValueError('finite double-precision data required')
        return result
    except (TypeError, OverflowError) as exc:
        raise ValueError('numeric data required') from exc
def _r7_int(value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('integer required, excluding bool')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError('integer outside domain')
    return value
def _r7_scalar(value, low=None, high=None):
    value = float(_r7_numeric(value, (), True))
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError('scalar outside domain')
    return value
def _r7_bool(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('boolean required')
    return bool(value)
def _r7_dimension(size, modes):
    for d in range(2, 41):
        if d**modes == size:
            return d
    raise ValueError('dimension is not a supported tensor power')
def _r7_state_parts(value, size=None):
    rho = _r7_numeric(value)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2:
        raise ValueError('square state required')
    if size is not None and rho.shape != (size, size):
        raise ValueError('state dimension mismatch')
    if np.max(np.abs(rho-rho.conj().T)) > 1e-10:
        raise ValueError('Hermitian state required')
    h = (rho+rho.conj().T)/2
    trace = math.fsum(float(value) for value in h.diagonal().real)
    if trace < -1e-10 or trace > 1+1e-10:
        raise ValueError('state trace outside domain')
    scale = float(np.max(np.abs(h)))
    if scale == 0:
        return np.zeros_like(h), 0.
    try:
        values, vectors = np.linalg.eigh(h.real/scale+1j*(h.imag/scale))
    except np.linalg.LinAlgError as exc:
        raise ValueError('physical state required') from exc
    if values.min()*scale < -1e-10:
        raise ValueError('positive state required')
    mass = max(0., trace)
    if mass == 0:
        return np.zeros_like(h), 0.
    positive = np.maximum(values, 0.)
    root = vectors*np.sqrt(positive/positive.sum())
    return root, mass
def _r7_factor_parts(value, size):
    w = _r7_numeric(value)
    if w.ndim != 2 or w.shape[0] != size or w.shape[1] < 1:
        raise ValueError('amplitude-column shape mismatch')
    scale = float(np.max(np.abs(w)))
    if scale == 0:
        return w, 0.
    if scale > np.sqrt(1+1e-10):
        raise ValueError('amplitude mass exceeds one')
    scaled = w.real/scale+1j*(w.imag/scale)
    norm = float(np.sqrt(np.vdot(scaled, scaled).real))
    amplitude = scale*norm
    if amplitude**2 > 1+1e-10:
        raise ValueError('amplitude mass exceeds one')
    return scaled/norm, amplitude
def herald(state: object, count: int, efficiency: float, detected_mode: int = 0, factorized: bool = False, modes: int = 2) -> object:
    modes = _r7_int(modes, 2, 3)
    detected_mode = _r7_int(detected_mode, 0, modes-1)
    count = _r7_int(count, 0)
    efficiency = _r7_scalar(efficiency, 0, 1)
    factorized = _r7_bool(factorized)
    raw = _r7_numeric(state)
    if raw.ndim != 2:
        raise ValueError('matrix or amplitude columns required')
    d = _r7_dimension(raw.shape[0], modes)
    if factorized:
        root, amplitude = _r7_factor_parts(raw, d**modes)
        mass = amplitude**2
    else:
        root, mass = _r7_state_parts(raw, d**modes)
    size = d**(modes-1)
    if count >= d or mass == 0:
        return np.zeros((size, size), complex)
    shaped = root.reshape((d,)*modes+(root.shape[1],))
    order = [i for i in range(modes) if i != detected_mode]+[detected_mode, modes]
    retained = shaped.transpose(order).reshape(size, d, -1)
    result = np.zeros((size, size), complex)
    for n in range(count, d):
        weight = math.comb(n, count)*efficiency**count*(1-efficiency)**(n-count)
        result += weight*(retained[:, n]@retained[:, n].conj().T)
    return mass*result

"""Correlated quantum memory selected by a first optical measurement."""
import numpy as np
def _r7_numeric(value, shape=None, real=False):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in ('iuf' if real else 'iufc'):
            raise ValueError('numeric kind required')
        if shape is not None and raw.shape != shape:
            raise ValueError('shape mismatch')
        if not np.all(np.isfinite(raw)):
            raise ValueError('finite numeric values required')
        result = raw.astype(float if real else complex)
        if not np.all(np.isfinite(result)):
            raise ValueError('finite double-precision data required')
        return result
    except (TypeError, OverflowError) as exc:
        raise ValueError('numeric data required') from exc
def _r7_int(value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('integer required, excluding bool')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError('integer outside domain')
    return value
def _r7_counts(value, length=None, distinct=False):
    raw = np.asarray(value, dtype=object)
    if raw.ndim != 1 or not 1 <= len(raw) <= 2:
        raise ValueError('one or two counts required')
    if length is not None and len(raw) != length:
        raise ValueError('count length mismatch')
    result = [_r7_int(v, 0) for v in raw]
    if distinct and len(set(result)) != len(result):
        raise ValueError('distinct first records required')
    return result
def retained_states(d: int, squeezers: object, mixers: object, counts: object, efficiencies: object) -> object:
    d = _r7_int(d, 2, 40)
    squeezers = _r7_numeric(squeezers, (3, 2), True)
    mixers = _r7_numeric(mixers, (3, 2), True)
    efficiencies = _r7_numeric(efficiencies, (3,), True)
    counts = _r7_counts(counts, distinct=True)
    if np.any(squeezers[:, 0] < 0) or np.any(squeezers[:, 0] > 1):
        raise ValueError('squeezing magnitude outside [0,1]')
    if np.any(mixers[:, 0] < 0) or np.any(mixers[:, 0] > np.pi/2):
        raise ValueError('mixing angle outside [0,pi/2]')
    if np.any(efficiencies < 0) or np.any(efficiencies > 1):
        raise ValueError('efficiency outside [0,1]')
    vacua = [gaussian_gate(d, 'squeeze', row)[:, 0] for row in squeezers]
    psi = np.einsum('i,j,k->ijk', *vacua)
    for pair, row in zip(((0, 1), (1, 2), (0, 2)), mixers):
        permutation = list(pair)+[i for i in range(3) if i not in pair]
        columns = psi.transpose(permutation).reshape(d*d, d)
        moved = gaussian_gate(d, 'mix', row, columns)
        psi = moved.reshape(d, d, d).transpose(np.argsort(permutation))
    rows = []
    for count in counts:
        rho = herald(psi.reshape(d**3, 1), count, efficiencies[0], 0, True, 3)
        rho = attenuate(rho, efficiencies[1], 0, 2)
        rows.append(attenuate(rho, efficiencies[2], 1, 2))
    return np.array(rows)

"""Outcome-selected Gaussian processing of a retained two-mode state."""
import math
import numpy as np
def _r7_numeric(value, shape=None, real=False):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in ('iuf' if real else 'iufc'):
            raise ValueError('numeric kind required')
        if shape is not None and raw.shape != shape:
            raise ValueError('shape mismatch')
        if not np.all(np.isfinite(raw)):
            raise ValueError('finite numeric values required')
        result = raw.astype(float if real else complex)
        if not np.all(np.isfinite(result)):
            raise ValueError('finite double-precision data required')
        return result
    except (TypeError, OverflowError) as exc:
        raise ValueError('numeric data required') from exc
def _r7_scalar(value, low=None, high=None):
    value = float(_r7_numeric(value, (), True))
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError('scalar outside domain')
    return value
def _r7_bool(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('boolean required')
    return bool(value)
def _r7_dimension(size, modes):
    for d in range(2, 41):
        if d**modes == size:
            return d
    raise ValueError('dimension is not a supported tensor power')
def _r7_state_parts(value, size=None):
    rho = _r7_numeric(value)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2:
        raise ValueError('square state required')
    if size is not None and rho.shape != (size, size):
        raise ValueError('state dimension mismatch')
    if np.max(np.abs(rho-rho.conj().T)) > 1e-10:
        raise ValueError('Hermitian state required')
    h = (rho+rho.conj().T)/2
    trace = math.fsum(float(value) for value in h.diagonal().real)
    if trace < -1e-10 or trace > 1+1e-10:
        raise ValueError('state trace outside domain')
    scale = float(np.max(np.abs(h)))
    if scale == 0:
        return np.zeros_like(h), 0.
    try:
        values, vectors = np.linalg.eigh(h.real/scale+1j*(h.imag/scale))
    except np.linalg.LinAlgError as exc:
        raise ValueError('physical state required') from exc
    if values.min()*scale < -1e-10:
        raise ValueError('positive state required')
    mass = max(0., trace)
    if mass == 0:
        return np.zeros_like(h), 0.
    positive = np.maximum(values, 0.)
    root = vectors*np.sqrt(positive/positive.sum())
    return root, mass
def _r7_factor_parts(value, size):
    w = _r7_numeric(value)
    if w.ndim != 2 or w.shape[0] != size or w.shape[1] < 1:
        raise ValueError('amplitude-column shape mismatch')
    scale = float(np.max(np.abs(w)))
    if scale == 0:
        return w, 0.
    if scale > np.sqrt(1+1e-10):
        raise ValueError('amplitude mass exceeds one')
    scaled = w.real/scale+1j*(w.imag/scale)
    norm = float(np.sqrt(np.vdot(scaled, scaled).real))
    amplitude = scale*norm
    if amplitude**2 > 1+1e-10:
        raise ValueError('amplitude mass exceeds one')
    return scaled/norm, amplitude
def feedforward_state(state: object, controls: object, factorized: bool = False) -> object:
    x = _r7_numeric(controls, (8,), True)
    factorized = _r7_bool(factorized)
    for i in (0, 6):
        _r7_scalar(x[i], 0, np.pi/2)
    for i in (2, 4):
        _r7_scalar(x[i], 0, .5)
    raw = _r7_numeric(state)
    if raw.ndim != 2:
        raise ValueError('matrix or amplitude columns required')
    d = _r7_dimension(raw.shape[0], 2)
    if factorized:
        root, amplitude = _r7_factor_parts(raw, d*d)
        mass = amplitude**2
    else:
        root, mass = _r7_state_parts(raw, d*d)
        amplitude = np.sqrt(mass)
    if amplitude == 0:
        return np.zeros_like(raw)
    w = gaussian_gate(d, 'mix', x[:2], root)
    tensor = w.reshape(d, d, -1)
    s0 = gaussian_gate(d, 'squeeze', x[2:4])
    s1 = gaussian_gate(d, 'squeeze', x[4:6])
    tensor = np.einsum('ai,bj,ijr->abr', s0, s1, tensor, optimize=True)
    w = gaussian_gate(d, 'mix', x[6:8], tensor.reshape(d*d, -1))
    return amplitude*w if factorized else mass*(w@w.conj().T)

"""Physical success and target overlap for a record-indexed instrument."""
import math
import numpy as np
def _r7_numeric(value, shape=None, real=False):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in ('iuf' if real else 'iufc'):
            raise ValueError('numeric kind required')
        if shape is not None and raw.shape != shape:
            raise ValueError('shape mismatch')
        if not np.all(np.isfinite(raw)):
            raise ValueError('finite numeric values required')
        result = raw.astype(float if real else complex)
        if not np.all(np.isfinite(result)):
            raise ValueError('finite double-precision data required')
        return result
    except (TypeError, OverflowError) as exc:
        raise ValueError('numeric data required') from exc
def _r7_int(value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('integer required, excluding bool')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError('integer outside domain')
    return value
def _r7_bool(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('boolean required')
    return bool(value)
def _r7_dimension(size, modes):
    for d in range(2, 41):
        if d**modes == size:
            return d
    raise ValueError('dimension is not a supported tensor power')
def _r7_state_parts(value, size=None):
    rho = _r7_numeric(value)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2:
        raise ValueError('square state required')
    if size is not None and rho.shape != (size, size):
        raise ValueError('state dimension mismatch')
    if np.max(np.abs(rho-rho.conj().T)) > 1e-10:
        raise ValueError('Hermitian state required')
    h = (rho+rho.conj().T)/2
    trace = math.fsum(float(value) for value in h.diagonal().real)
    if trace < -1e-10 or trace > 1+1e-10:
        raise ValueError('state trace outside domain')
    scale = float(np.max(np.abs(h)))
    if scale == 0:
        return np.zeros_like(h), 0.
    try:
        values, vectors = np.linalg.eigh(h.real/scale+1j*(h.imag/scale))
    except np.linalg.LinAlgError as exc:
        raise ValueError('physical state required') from exc
    if values.min()*scale < -1e-10:
        raise ValueError('positive state required')
    mass = max(0., trace)
    if mass == 0:
        return np.zeros_like(h), 0.
    positive = np.maximum(values, 0.)
    root = vectors*np.sqrt(positive/positive.sum())
    return root, mass
def _r7_factor_parts(value, size):
    w = _r7_numeric(value)
    if w.ndim != 2 or w.shape[0] != size or w.shape[1] < 1:
        raise ValueError('amplitude-column shape mismatch')
    scale = float(np.max(np.abs(w)))
    if scale == 0:
        return w, 0.
    if scale > np.sqrt(1+1e-10):
        raise ValueError('amplitude mass exceeds one')
    scaled = w.real/scale+1j*(w.imag/scale)
    norm = float(np.sqrt(np.vdot(scaled, scaled).real))
    amplitude = scale*norm
    if amplitude**2 > 1+1e-10:
        raise ValueError('amplitude mass exceeds one')
    return scaled/norm, amplitude
def _r7_counts(value, length=None, distinct=False):
    raw = np.asarray(value, dtype=object)
    if raw.ndim != 1 or not 1 <= len(raw) <= 2:
        raise ValueError('one or two counts required')
    if length is not None and len(raw) != length:
        raise ValueError('count length mismatch')
    result = [_r7_int(v, 0) for v in raw]
    if distinct and len(set(result)) != len(result):
        raise ValueError('distinct first records required')
    return result
def _r7_target(value, d):
    target = _r7_numeric(value, (d,))
    if abs(np.vdot(target, target).real-1) > 1e-10:
        raise ValueError('normalized target required')
    return target
def _r7_policy_inputs(states, target, controls, counts, efficiencies, factorized):
    factorized = _r7_bool(factorized)
    raw = _r7_numeric(states)
    if raw.ndim != 3 or raw.shape[0] not in (1, 2):
        raise ValueError('one or two branch matrices required')
    d = _r7_dimension(raw.shape[1], 2)
    target = _r7_target(target, d)
    controls = _r7_numeric(controls, (len(raw), 8), True)
    if np.any(controls[:, [0, 6]] < 0) or np.any(controls[:, [0, 6]] > np.pi/2):
        raise ValueError('mixing angle outside domain')
    if np.any(controls[:, [2, 4]] < 0) or np.any(controls[:, [2, 4]] > .5):
        raise ValueError('inline squeezing outside domain')
    counts = _r7_counts(counts, len(raw))
    efficiencies = _r7_numeric(efficiencies, (2,), True)
    if np.any(efficiencies < 0) or np.any(efficiencies > 1):
        raise ValueError('efficiency outside domain')
    roots, masses = [], []
    for state in raw:
        if factorized:
            root, amplitude = _r7_factor_parts(state, d*d)
            mass = amplitude**2
        else:
            root, mass = _r7_state_parts(state, d*d)
        roots.append(root)
        masses.append(mass)
    if sum(masses) > 1+1e-10:
        raise ValueError('total physical mass exceeds one')
    return d, roots, masses, target, controls, counts, efficiencies
def policy_statistics(states: object, target: object, controls: object, counts: object, efficiencies: object, factorized: bool = False) -> object:
    if states is None:
        raise ValueError('states are required')
    d, roots, masses, target, controls, counts, efficiencies = _r7_policy_inputs(states, target, controls, counts, efficiencies, factorized)
    result = np.zeros((len(roots), 2))
    for k, (root, mass) in enumerate(zip(roots, masses)):
        if mass == 0:
            continue
        moved = feedforward_state(root, controls[k], True)
        conditional = herald(moved, counts[k], efficiencies[0], 1, True, 2)
        signal = attenuate(conditional, efficiencies[1])
        result[k] = mass*np.array([np.trace(signal).real, np.vdot(target, signal@target).real])
    return result

"""Bounded optimization of the complete adaptive optical instrument."""
import math
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
def _r7_numeric(value, shape=None, real=False):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in ('iuf' if real else 'iufc'):
            raise ValueError('numeric kind required')
        if shape is not None and raw.shape != shape:
            raise ValueError('shape mismatch')
        if not np.all(np.isfinite(raw)):
            raise ValueError('finite numeric values required')
        result = raw.astype(float if real else complex)
        if not np.all(np.isfinite(result)):
            raise ValueError('finite double-precision data required')
        return result
    except (TypeError, OverflowError) as exc:
        raise ValueError('numeric data required') from exc
def _r7_int(value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('integer required, excluding bool')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError('integer outside domain')
    return value
def _r7_scalar(value, low=None, high=None):
    value = float(_r7_numeric(value, (), True))
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError('scalar outside domain')
    return value
def _r7_bool(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('boolean required')
    return bool(value)
def _r7_dimension(size, modes):
    for d in range(2, 41):
        if d**modes == size:
            return d
    raise ValueError('dimension is not a supported tensor power')
def _r7_state_parts(value, size=None):
    rho = _r7_numeric(value)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2:
        raise ValueError('square state required')
    if size is not None and rho.shape != (size, size):
        raise ValueError('state dimension mismatch')
    if np.max(np.abs(rho-rho.conj().T)) > 1e-10:
        raise ValueError('Hermitian state required')
    h = (rho+rho.conj().T)/2
    trace = math.fsum(float(value) for value in h.diagonal().real)
    if trace < -1e-10 or trace > 1+1e-10:
        raise ValueError('state trace outside domain')
    scale = float(np.max(np.abs(h)))
    if scale == 0:
        return np.zeros_like(h), 0.
    try:
        values, vectors = np.linalg.eigh(h.real/scale+1j*(h.imag/scale))
    except np.linalg.LinAlgError as exc:
        raise ValueError('physical state required') from exc
    if values.min()*scale < -1e-10:
        raise ValueError('positive state required')
    mass = max(0., trace)
    if mass == 0:
        return np.zeros_like(h), 0.
    positive = np.maximum(values, 0.)
    root = vectors*np.sqrt(positive/positive.sum())
    return root, mass
def _r7_counts(value, length=None, distinct=False):
    raw = np.asarray(value, dtype=object)
    if raw.ndim != 1 or not 1 <= len(raw) <= 2:
        raise ValueError('one or two counts required')
    if length is not None and len(raw) != length:
        raise ValueError('count length mismatch')
    result = [_r7_int(v, 0) for v in raw]
    if distinct and len(set(result)) != len(result):
        raise ValueError('distinct first records required')
    return result
def _r7_target(value, d):
    target = _r7_numeric(value, (d,))
    if abs(np.vdot(target, target).real-1) > 1e-10:
        raise ValueError('normalized target required')
    return target
def _r7_optimize(states, target, counts, efficiencies, rmax, adaptive):
    class _r7_adjoint:
        def __init__(self, d, factors, target, counts, efficiencies):
            self.d, self.factors, self.counts = d, factors, counts
            a = np.diag(np.sqrt(np.arange(1, d)), 1)
            self.generator = (a@a-a.T@a.T)/2
            self.values, self.vectors = eigh(-1j*self.generator)
            self.vectors_h = self.vectors.conj().T
            self.n = np.arange(d)
            # These factors depend only on the fixed local dimension.
            self.number_grid = np.repeat(self.n, d)[:, None]
            root = np.sqrt(np.arange(1, d))
            self.mixing_weights = root[:, None, None]*root[None, :, None]
            self.blocks = []
            for total in range(2*d-1):
                ids = np.array([i*d+total-i for i in range(d) if 0 <= total-i < d])
                g = np.zeros((len(ids), len(ids)), complex)
                for col, index in enumerate(ids):
                    i, j = divmod(int(index), d)
                    if i+1 < d and j:
                        g[col+1, col] = np.sqrt((i+1)*j)
                        g[col, col+1] = -np.sqrt((i+1)*j)
                val, vec = eigh(-1j*g)
                self.blocks.append((ids, ids//d, val, vec, vec.conj().T))
            self.q = np.zeros((d, d), complex)
            eta = efficiencies[1]
            for ell in range(d):
                v = np.zeros(d, complex)
                for n in range(ell, d):
                    v[n] = target[n-ell]*np.sqrt(math.comb(n, ell)*(1-eta)**ell*eta**(n-ell))
                self.q += np.outer(v, v.conj())
            eta = efficiencies[0]
            self.effects, self.effect_logs, self.relative_logs = [], [], []
            for k in counts:
                logs = np.full(d, -np.inf)
                for n in range(k, d):
                    if (eta == 0 and k != 0) or (eta == 1 and n != k):
                        continue
                    logs[n] = (math.log(math.comb(n, k))
                               + (k*math.log(eta) if k else 0.)
                               + ((n-k)*math.log1p(-eta) if n > k else 0.))
                shift = float(np.max(logs))
                self.effect_logs.append(shift)
                self.relative_logs.append(logs-shift if np.isfinite(shift) else logs)
                self.effects.append(np.exp(logs-shift) if np.isfinite(shift) else np.zeros(d))

        def mix(self, w, theta, phase):
            if theta == 0:
                return w.copy()
            out = np.empty_like(w)
            phases = np.exp(1j*phase)**self.n
            for ids, ns, values, vectors, vectors_h in self.blocks:
                ph = phases[ns]
                z = vectors_h@(w[ids]*ph.conj()[:, None])
                out[ids] = ph[:, None]*(vectors@(np.exp(1j*theta*values)[:, None]*z))
            return out

        def mix_generator(self, w, phase):
            t = w.reshape(self.d, self.d, -1)
            out = np.zeros_like(t)
            factors = self.mixing_weights
            out[1:, :-1] += np.exp(1j*phase)*factors*t[:-1, 1:]
            out[:-1, 1:] -= np.exp(-1j*phase)*factors*t[1:, :-1]
            return out.reshape(w.shape)

        def squeeze(self, r, phase):
            ph = np.exp(.5j*phase)**self.n
            u = np.eye(self.d, dtype=complex) if r == 0 else (self.vectors*np.exp(1j*r*self.values))@self.vectors_h
            u = ph[:, None]*u*ph.conj()[None, :]
            g = ph[:, None]*self.generator*ph.conj()[None, :]
            return u, g@u, .5j*(self.n[:, None]-self.n[None, :])*u

        def forward(self, branch, x):
            d = self.d
            w0 = self.factors[branch]
            w1 = self.mix(w0, *x[:2])
            s0, s1 = self.squeeze(*x[2:4]), self.squeeze(*x[4:6])
            w2 = np.einsum('ai,bj,ijr->abr', s0[0], s1[0], w1.reshape(d, d, -1), optimize=True).reshape(d*d, -1)
            w3 = self.mix(w2, *x[6:8])
            t = w3.reshape(d, d, -1)
            q = np.einsum('ik,kjr->ijr', self.q, t, optimize=True)
            effect = self.effects[branch]
            p = np.einsum('ijr,ijr,j->', t.conj(), t, effect).real
            m = np.einsum('ijr,ijr,j->', t.conj(), q, effect).real
            offset = 0.
            rescaled = False
            # This threshold selects an arithmetic representation; it never
            # discards a positive success or changes the physical objective.
            if p < 1e-150:
                magnitude = np.abs(t)
                nonzero = magnitude > 0
                log_amplitude = np.full(t.shape, -np.inf)
                log_amplitude[nonzero] = np.log(magnitude[nonzero])
                log_amplitude += .5*self.relative_logs[branch][None, :, None]
                shift = float(np.max(log_amplitude))
                if np.isfinite(shift):
                    phase = np.zeros_like(t)
                    phase[nonzero] = (t.real[nonzero]/magnitude[nonzero]
                                     +1j*(t.imag[nonzero]/magnitude[nonzero]))
                    weighted = phase*np.exp(log_amplitude-shift)
                    weighted_q = np.einsum('ik,kjr->ijr', self.q, weighted, optimize=True)
                    p = float(np.vdot(weighted, weighted).real)
                    m = float(np.vdot(weighted, weighted_q).real)
                    offset, rescaled = 2*shift, True
            return np.array([p, m]), (w0, w1, w2, w3, s0, s1, q, effect, rescaled), offset

        def gradient(self, x, cache, alpha, beta):
            d = self.d
            w0, w1, w2, w3, s0, s1, q, effect, _ = cache
            z3 = ((alpha*w3.reshape(d, d, -1)+beta*q)*effect[None, :, None]).reshape(d*d, -1)
            grad = np.zeros(8)
            n = self.number_grid
            pair = lambda z, w: 2*float(np.vdot(z, w).real)
            grad[6] = pair(z3, self.mix_generator(w3, x[7]))
            grad[7] = pair(z3, 1j*(n*w3-self.mix(n*w2, *x[6:8])))
            z2 = self.mix(z3, -x[6], x[7]).reshape(d, d, -1)
            t1 = w1.reshape(d, d, -1)
            for j in (1, 2):
                grad[1+j] = pair(z2, np.einsum('ai,bj,ijr->abr', s0[j], s1[0], t1, optimize=True))
                grad[3+j] = pair(z2, np.einsum('ai,bj,ijr->abr', s0[0], s1[j], t1, optimize=True))
            z1 = np.einsum('ia,jb,abr->ijr', s0[0].conj().T, s1[0].conj().T, z2, optimize=True).reshape(d*d, -1)
            grad[0] = pair(z1, self.mix_generator(w1, x[1]))
            grad[1] = pair(z1, 1j*(n*w1-self.mix(n*w0, *x[:2])))
            return grad

    rmax = _r7_scalar(rmax, 0, .5)
    adaptive = _r7_bool(adaptive)
    raw = _r7_numeric(states)
    if raw.ndim != 3 or raw.shape[0] not in (1, 2):
        raise ValueError('one or two branch states required')
    d = _r7_dimension(raw.shape[1], 2)
    branches = len(raw)
    policy_statistics(raw, target, np.zeros((branches, 8)), counts, efficiencies)
    target = _r7_target(target, d)
    counts = _r7_counts(counts, branches)
    efficiencies = _r7_numeric(efficiencies, (2,), True)
    pieces = [_r7_state_parts(rho, d*d) for rho in raw]
    mass_logs = np.array([math.log(mass) if mass > 0 else -np.inf for root, mass in pieces])
    if not np.any(np.isfinite(mass_logs)) or all(k >= d for k in counts):
        return 0., np.zeros((branches, 8)), np.zeros((branches, 2))
    full = [root for root, mass in pieces]
    compressed, omissions = [], []
    for w in full:
        weights = np.sum(np.abs(w)**2, axis=0)
        order = np.argsort(weights)[::-1]
        tail = np.cumsum(weights[order][::-1])[::-1]
        rank = max(1, int(np.count_nonzero(tail > 1e-14*weights.sum())))
        compressed.append(w[:, order[:rank]])
        omissions.append(float(weights[order[rank:]].sum()))
    adjoint = _r7_adjoint(d, compressed, target, counts, efficiencies)
    branch_logs = mass_logs+np.array(adjoint.effect_logs)

    def pool(moments, offsets):
        live = (moments[:, 0] > 0) & np.isfinite(branch_logs)
        if not np.any(live):
            return 0., 0., np.zeros(branches)
        log_contributions = branch_logs[live]+offsets[live]+np.log(moments[live, 0])
        shift = float(np.max(log_contributions))
        relative = np.exp(log_contributions-shift)
        total = float(relative.sum())
        fractions = relative/total
        probability = math.exp(shift+math.log(total))
        fidelity = float(np.dot(fractions, moments[live, 1]/moments[live, 0]))
        coefficients = np.zeros(branches)
        coefficients[live] = fractions/moments[live, 0]
        return probability, fidelity, coefficients

    block = [(0., np.pi/2), (-np.pi, np.pi), (0., rmax), (-np.pi, np.pi),
             (0., rmax), (-np.pi, np.pi), (0., np.pi/2), (-np.pi, np.pi)]
    active = [i for i in range(8) if block[i][1] > block[i][0] and not (rmax == 0 and i in (3, 5))]
    bounds = [block[i] for i in active]
    local = [(None, None) if i in (1, 3, 5, 7) else block[i] for i in active]

    def expand(z):
        result = np.zeros(8)
        result[active] = z
        return result

    rng = np.random.default_rng(716)
    candidates = [[] for k in range(branches)]
    starts_per_support = 16
    for k in range(branches):
        for lam in (.7, .85, .95, 1.02):
            def support(z):
                x = expand(z)
                pm, cache, offset = adjoint.forward(k, x)
                if cache[-1]:
                    # The support is below the support search's absolute
                    # tolerance. Joint refinement still evaluates its ratio.
                    return -100.*math.exp(offset)*float(pm[1]-lam*pm[0]), np.zeros(len(active))
                grad = adjoint.gradient(x, cache, -lam, 1.)
                return -100.*float(pm[1]-lam*pm[0]), -100.*grad[active]
            seeds = [np.zeros(len(active))]+[np.array(c['x'])[active] for c in candidates[k]]
            seeds += [np.array([rng.uniform(lo, hi) for lo, hi in bounds]) for j in range(starts_per_support)]
            results = []
            for seed in seeds:
                fit = minimize(support, seed, method='L-BFGS-B', jac=True, bounds=local,
                               options={'maxiter':600, 'ftol':1e-13, 'gtol':1e-7})
                x = expand(fit.x)
                pm, _, offset = adjoint.forward(k, x)
                results.append({'support': -float(fit.fun), 'x': x, 'pm': pm, 'offset': offset})
            candidates[k].extend(sorted(results, key=lambda t: -t['support'])[:3])
    if adaptive and branches == 2:
        starts = []
        for first in candidates[0]:
            for second in candidates[1]:
                probability, fidelity, _ = pool(np.array([first['pm'], second['pm']]),
                                                  np.array([first['offset'], second['offset']]))
                reward = probability+fidelity
                starts.append((reward, np.r_[first['x'], second['x']]))
        starts = [x for value, x in sorted(starts, key=lambda t: -t[0])[:16]]
    else:
        starts = [c['x'] for row in candidates for c in row]
        starts += [expand(np.array([rng.uniform(lo, hi) for lo, hi in bounds])) for j in range(starts_per_support)]
    starts.append(np.zeros(8*branches if adaptive else 8))
    controllers = branches if adaptive else 1
    chosen = active if controllers == 1 else active+[i+8 for i in active]
    local_joint = local*controllers

    def inflate(z):
        return expand(z) if controllers == 1 else np.r_[expand(z[:len(active)]), expand(z[len(active):])]

    def evaluate(z, exact=False, need_gradient=True):
        x = inflate(z).reshape(controllers, 8)
        if not adaptive:
            x = np.repeat(x, branches, axis=0)
        try:
            adjoint.factors = full if exact else compressed
            forward = [adjoint.forward(k, x[k]) for k in range(branches)]
            moments = np.array([item[0] for item in forward])
            offsets = np.array([item[2] for item in forward])
            probability, fidelity, coefficients = pool(moments, offsets)
            # Bound PSD compression error in units of the accepted mass.
            # Any rare-event rescaling uses full factors before acceptance.
            omitted_ratio = float(np.dot(coefficients, omissions))
            if not exact and (np.any(moments[:, 0] == 0)
                              or any(item[1][-1] for item in forward)
                              or (probability+2)*omitted_ratio > 1e-10):
                adjoint.factors = full
                forward = [adjoint.forward(k, x[k]) for k in range(branches)]
                moments = np.array([item[0] for item in forward])
                offsets = np.array([item[2] for item in forward])
                probability, fidelity, coefficients = pool(moments, offsets)
            physical_moments = np.exp(branch_logs+offsets)[:, None]*moments
            if not np.any(coefficients):
                return 0., np.zeros_like(z), physical_moments
            reward = float(probability+fidelity)
            if not need_gradient:
                return reward, None, physical_moments
            if any(item[1][-1] for item in forward):
                # Finite differences of the stabilized scalar avoid a
                # reciprocal amplitude beyond binary64's exponent range.
                gradient = np.empty_like(z)
                for j in range(len(z)):
                    step = 1e-6
                    plus, minus = z.copy(), z.copy()
                    plus[j] += step
                    minus[j] -= step
                    gradient[j] = (evaluate(plus, True, False)[0]
                                   -evaluate(minus, True, False)[0])/(2*step)
                return reward, gradient, physical_moments
            # dR = dP + (dM-F*dP)/P; the common scale cancels before
            # differentiation. Neither physical P squared nor 1/P is formed.
            gradients = [adjoint.gradient(x[k], item[1],
                         (probability-fidelity)*coefficients[k], coefficients[k])
                         for k, item in enumerate(forward)]
            gradient = np.array(gradients).ravel() if adaptive else np.array(gradients).sum(axis=0)
            return reward, gradient[chosen], physical_moments
        finally:
            adjoint.factors = compressed

    def objective(z):
        value, gradient, _ = evaluate(z)
        return -value, -gradient

    results = []
    for x in starts:
        seed = x[chosen]
        value, _, moments = evaluate(seed, True, need_gradient=False)
        results.append((value, seed, moments))
        fit = minimize(objective, seed, method='L-BFGS-B', jac=True, bounds=local_joint,
                       options={'maxiter':1000, 'ftol':5e-14, 'gtol':1e-8, 'maxls':40})
        value, _, moments = evaluate(fit.x, True, need_gradient=False)
        results.append((value, fit.x, moments))
    value, best, best_moments = max(results, key=lambda t: t[0])
    x = inflate(best).reshape(controllers, 8)
    if not adaptive:
        x = np.repeat(x, branches, axis=0)
    return value, x, best_moments
def optimal_reward(states: object, target: object, counts: object, efficiencies: object, rmax: float, adaptive: bool) -> float:
    if states is None:
        raise ValueError('states are required')
    return _r7_optimize(states, target, counts, efficiencies, rmax, adaptive)[0]

import math
import numpy as np

def source_reward(configuration: str = "benchmark") -> float:
    if not isinstance(configuration, str) or configuration not in (
            "benchmark", "vacuum", "common", "adaptive", "rare_detector"):
        raise ValueError('unknown source configuration')
    d = 28
    squeezers = np.array([[.50, .18], [.41, 2.72], [.46, .39]])
    mixers = np.array([[.63, .27], [.51, 1.03], [.37, -.42]])
    first_counts, second_counts = (1, 2), (2, 1)
    efficiencies = np.array([.99, .98, .95, .99, .99])
    target_spec, rmax, adaptive = (6**.5, .5, 0.), .5, True
    if configuration == "vacuum":
        d, squeezers = 3, np.zeros((3, 2))
        mixers = np.array([[.31, .2], [.27, -.4], [.19, .7]])
        first_counts, second_counts = (1,), (0,)
        efficiencies = np.array([.96, .91, .93, .89, .94])
        target_spec, rmax = (1.1, .2, .1), .18
    elif configuration in ("common", "adaptive"):
        d = 4
        squeezers = np.array([[.27, .1], [.18, -.5], [.33, .7]])
        mixers = np.array([[.34, -.1], [.22, .6], [.41, -.4]])
        first_counts, second_counts = (0, 1), (1, 0)
        efficiencies = np.array([.91, .88, .93, .90, .95])
        target_spec, rmax = (1.3, .22, .1), .15
        adaptive = configuration == "adaptive"
    elif configuration == "rare_detector":
        d = 3
        first_counts, second_counts = (1,), (2,)
        efficiencies = np.array([1., 1., 1., 1e-200, 1.])
        target_spec, rmax = (.9, 0., 0.), 0.
    states = retained_states(d, squeezers, mixers, first_counts, efficiencies[:3])
    alpha, r, phase = target_spec
    core = np.zeros(d, complex)
    for n in range(1, d, 2):
        core[n] = alpha**(n-1)/math.sqrt(float(math.factorial(n)))
    core /= np.linalg.norm(core)
    target = gaussian_gate(d, 'squeeze', [r, phase])@core
    baseline_moments = policy_statistics(
        states, target, np.zeros((len(first_counts), 8)), second_counts, efficiencies[3:])
    baseline_probability, baseline_overlap = baseline_moments.sum(axis=0)
    baseline_reward = (0. if baseline_probability == 0 else
                       baseline_probability+baseline_overlap/baseline_probability)
    reward = float(optimal_reward(
        states, target, second_counts, efficiencies[3:], rmax, adaptive))
    if reward+5e-10 < baseline_reward:
        raise RuntimeError('optimized reward is below a valid baseline policy')
    return reward
SCICODE_GOLD_EOF
