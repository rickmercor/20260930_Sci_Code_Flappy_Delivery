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
from scipy.linalg import block_diag, eigh, null_space


def crystal_acoustic_tensors(euler_angles, constants):
    angles = np.asarray(euler_angles, dtype=float)
    if angles.ndim != 2 or angles.shape[1] != 3 or not np.isfinite(angles).all():
        raise ValueError('euler_angles must be a finite (n_grains, 3) array')
    c11, c33, c12, c13, c44 = [float(constants[key]) for key in ('c11', 'c33', 'c12', 'c13', 'c44')]
    voigt = np.array([[c11, c12, c13, 0, 0, 0], [c12, c11, c13, 0, 0, 0],
                      [c13, c13, c33, 0, 0, 0], [0, 0, 0, c44, 0, 0],
                      [0, 0, 0, 0, c44, 0], [0, 0, 0, 0, 0, (c11-c12)/2]])
    if not np.isfinite(voigt).all() or np.linalg.eigvalsh(voigt).min() <= 0:
        raise ValueError('crystal stiffness must be positive definite')
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    tensor = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    tensor[ii, jj, kk, ll] = voigt[a, b]
    result = []
    for a, b, c in np.deg2rad(angles):
        za = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        xb = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        zc = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        rotation = za @ xb @ zc
        acoustic = np.einsum('ia,b,kc,d,abcd->ik', rotation, rotation[0], rotation, rotation[0], tensor)
        result.append((acoustic + acoustic.T) / 2)
    return np.asarray(result).reshape((-1, 3, 3))

import numpy as np
from scipy.linalg import block_diag, eigh, null_space


def buffered_stiffness(acoustic, grain_voxels, density, length, n_buffer, buffer_ratio):
    acoustic = np.asarray(acoustic, dtype=float)
    counts = np.asarray(grain_voxels)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer) or counts.size == 0 or np.any(counts < 1):
        raise ValueError('grain_voxels must contain positive integers')
    if acoustic.shape != (counts.size, 3, 3) or not np.isfinite(acoustic).all():
        raise ValueError('acoustic must have shape (n_grains, 3, 3)')
    if not np.allclose(acoustic, acoustic.swapaxes(-1, -2), rtol=1e-12, atol=0) or np.linalg.eigvalsh(acoustic).min() <= 0:
        raise ValueError('acoustic tensors must be symmetric positive definite')
    if not isinstance(n_buffer, (int, np.integer)) or n_buffer < 1:
        raise ValueError('n_buffer must be a positive integer')
    if not all(np.isfinite(x) and x > 0 for x in (density, length, buffer_ratio)):
        raise ValueError('density, length and buffer_ratio must be positive')
    n_specimen = int(counts.sum())
    n_total = n_specimen + int(n_buffer)
    if n_total % 2 == 0:
        raise ValueError('the periodic grid must have odd length')
    h = float(length) / n_specimen
    specimen = np.repeat(acoustic, counts, axis=0)
    average = specimen.mean(axis=0)
    buffer = np.diag([average[0, 0], (average[1, 1]+average[2, 2])/2,
                      (average[1, 1]+average[2, 2])/2]) * buffer_ratio
    material = np.concatenate([specimen, np.repeat(buffer[None], n_buffer, axis=0)])
    symbol = (np.exp(2j*np.pi*np.fft.fftfreq(n_total)) - 1) / h
    derivative = np.fft.ifft(symbol[:, None] * np.fft.fft(np.eye(n_total), axis=0), axis=0).real
    gradient = np.kron(derivative, np.eye(3))
    stiffness = gradient.T @ block_diag(*material) @ gradient
    mass = np.repeat(np.r_[np.full(n_specimen, density), np.zeros(n_buffer)], 3)
    return {'stiffness': (stiffness+stiffness.T)/2, 'mass': mass,
            'n_specimen': n_specimen, 'voxel_size': h}

import numpy as np
from scipy.linalg import block_diag, eigh, null_space


def elastic_modes(stiffness, mass, n_specimen):
    stiffness = np.asarray(stiffness, dtype=float)
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 1 or mass.size % 3 or stiffness.shape != (mass.size, mass.size):
        raise ValueError('stiffness and mass dimensions disagree')
    if not isinstance(n_specimen, (int, np.integer)) or not 2 <= n_specimen < mass.size//3:
        raise ValueError('n_specimen must leave a nonempty buffer')
    n = 3*int(n_specimen)
    if not np.isfinite(stiffness).all() or not np.isfinite(mass).all() or np.any(mass[:n] <= 0) or np.any(mass[n:] != 0):
        raise ValueError('mass is positive on the specimen and zero on the buffer')
    if not np.allclose(stiffness, stiffness.T, rtol=1e-12, atol=1e-12*np.abs(stiffness).max()):
        raise ValueError('stiffness must be symmetric')
    condensed = stiffness[:n, :n] - stiffness[:n, n:] @ np.linalg.solve(stiffness[n:, n:], stiffness[n:, :n])
    condensed = (condensed+condensed.T)/2
    root_mass = np.sqrt(mass[:n])
    weighted = condensed / root_mass[:, None] / root_mass[None, :]
    translations = np.tile(np.eye(3), (n_specimen, 1))
    basis = null_space((root_mass[:, None]*translations).T)
    reduced = basis.T @ weighted @ basis
    eigenvalues, eigenvectors = eigh((reduced+reduced.T)/2)
    if np.any(eigenvalues <= 0):
        raise ValueError('elastic subspace must be positive definite')
    modes = (basis @ eigenvectors) / root_mass[:, None]
    return {'frequencies': np.sqrt(eigenvalues)/(2*np.pi), 'modes': modes,
            'condensed_stiffness': condensed}

import numpy as np
from scipy.linalg import block_diag, eigh, null_space


def modal_residues(frequencies, modes, source, receptor, direction, voxel_size, cluster_rtol):
    frequencies = np.asarray(frequencies, dtype=float)
    modes = np.asarray(modes, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0 or np.any(frequencies <= 0) or np.any(np.diff(frequencies) < 0):
        raise ValueError('frequencies must be positive and sorted')
    if modes.ndim != 2 or modes.shape[1] != frequencies.size or modes.shape[0] % 3:
        raise ValueError('modes must have shape (3*n_specimen, n_modes)')
    n = modes.shape[0]//3
    if not all(isinstance(x, (int, np.integer)) and 0 <= x < n for x in (source, receptor)):
        raise ValueError('source and receptor must index specimen voxels')
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError('direction must be a finite nonzero three-vector')
    if not np.isfinite(frequencies).all() or not np.isfinite(modes).all() or voxel_size <= 0 or not 0 <= cluster_rtol < 1:
        raise ValueError('invalid numerical inputs')
    traction = direction / np.linalg.norm(direction) / voxel_size
    excitation = modes[3*source:3*source+3].T @ traction
    residues = modes[3*receptor:3*receptor+3].T * excitation[:, None]
    groups, centres, sizes = [], [], []
    start = 0
    while start < frequencies.size:
        end = start+1
        while end < frequencies.size and frequencies[end]-frequencies[start] <= cluster_rtol*frequencies[start]:
            end += 1
        groups.append(residues[start:end].sum(axis=0))
        centres.append(frequencies[start:end].mean())
        sizes.append(end-start)
        start = end
    return {'frequencies': np.array(centres), 'residues': np.array(groups), 'sizes': np.array(sizes, dtype=int)}

import numpy as np
from scipy.linalg import block_diag, eigh, null_space


def modal_impact_record(frequencies, residues, dt, n_steps, pulse_amplitude, pulse_width, pulse_centre):
    frequencies = np.asarray(frequencies, dtype=float)
    residues = np.asarray(residues, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0 or residues.shape != (frequencies.size, 3):
        raise ValueError('frequencies and residues have incompatible shapes')
    if np.any(frequencies <= 0) or not np.isfinite(frequencies).all() or not np.isfinite(residues).all():
        raise ValueError('modal inputs must be finite with positive frequencies')
    if dt <= 0 or pulse_width <= 0 or not all(np.isfinite(x) for x in (dt, pulse_width, pulse_amplitude, pulse_centre)):
        raise ValueError('invalid pulse parameters')
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError('n_steps must be positive')
    omega2 = (2*np.pi*frequencies)**2
    q, v, a = [np.zeros_like(frequencies) for _ in range(3)]
    history = np.zeros((int(n_steps)+1, frequencies.size, 3))
    coefficient = 0.25*dt*dt
    for step in range(1, int(n_steps)+1):
        t = step*dt
        pulse = pulse_amplitude*(np.exp(-0.5*((t-pulse_centre)/pulse_width)**2)-np.exp(-0.5*((t-pulse_centre-pulse_width)/pulse_width)**2))
        predictor = q + dt*v + coefficient*a
        q_new = (predictor+coefficient*pulse)/(1+coefficient*omega2)
        a_new = pulse-omega2*q_new
        v += 0.5*dt*(a+a_new)
        q, a = q_new, a_new
        history[step] = q[:, None]*residues
    return history

import numpy as np
from scipy.linalg import block_diag, eigh, null_space


def finite_record_coherence(history, dt, frequency_band):
    history = np.asarray(history, dtype=float)
    band = np.asarray(frequency_band, dtype=float)
    if history.ndim != 3 or history.shape[0] < 2 or history.shape[1] < 1 or history.shape[2] != 3 or not np.isfinite(history).all():
        raise ValueError('history must be finite with shape (n_samples, n_clusters, 3)')
    if not np.isfinite(dt) or dt <= 0 or band.shape != (2,) or not np.isfinite(band).all() or not 0 < band[0] < band[1] <= 0.5/dt:
        raise ValueError('frequency_band must lie above zero and at or below Nyquist')
    frequencies = np.fft.rfftfreq(history.shape[0], dt)
    selected = (frequencies >= band[0]) & (frequencies <= band[1])
    if not selected.any():
        raise ValueError('the frequency band contains no DFT bin')
    transformed = np.fft.rfft(history, axis=0)[selected]
    coherent = float(np.sum(np.abs(transformed.sum(axis=1))**2))
    incoherent = float(np.sum(np.abs(transformed)**2))
    if incoherent == 0:
        raise ValueError('the selected modal response has zero power')
    return {'ratio': coherent/incoherent, 'coherent_power': coherent,
            'incoherent_power': incoherent, 'n_bins': int(selected.sum())}

def polycrystal_coherence(euler_angles, constants, grain_voxels, density, length, n_buffer,
                         buffer_ratio, direction, receptor, dt, n_steps, pulse_amplitude,
                         pulse_width, pulse_centre, frequency_band, cluster_rtol):
    acoustic = crystal_acoustic_tensors(euler_angles, constants)  # noqa: F821
    domain = buffered_stiffness(acoustic, grain_voxels, density, length, n_buffer, buffer_ratio)  # noqa: F821
    eigensystem = elastic_modes(domain['stiffness'], domain['mass'], domain['n_specimen'])  # noqa: F821
    grouped = modal_residues(eigensystem['frequencies'], eigensystem['modes'], 0, receptor,  # noqa: F821
                             direction, domain['voxel_size'], cluster_rtol)
    history = modal_impact_record(grouped['frequencies'], grouped['residues'], dt, n_steps,  # noqa: F821
                                  pulse_amplitude, pulse_width, pulse_centre)
    result = finite_record_coherence(history, dt, frequency_band)  # noqa: F821
    return result
SCICODE_GOLD_EOF
