#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

from itertools import combinations
import numpy as np


def pair_occupations(norb: int, npair: int) -> "np.ndarray":
    if not isinstance(norb, (int, np.integer)) or not isinstance(npair, (int, np.integer)) or not 1 <= npair <= norb <= 6:
        raise ValueError('Require 1 <= npair <= norb <= 6.')
    return np.array(list(combinations(range(norb), npair)), dtype=int)

import numpy as np


def doci_matrix(h: "np.ndarray", factors: "np.ndarray", occupations: "np.ndarray") -> "np.ndarray":
    h = np.asarray(h, dtype=float)
    factors = np.asarray(factors, dtype=float)
    occupations = np.asarray(occupations, dtype=int)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.allclose(h, h.T):
        raise ValueError('h must be real symmetric.')
    n = len(h)
    if factors.ndim != 3 or factors.shape[1:] != (n,n) or not np.allclose(factors, factors.transpose(0,2,1)):
        raise ValueError('factors must be real symmetric matrices.')
    if occupations.ndim != 2 or np.any(occupations < 0) or np.any(occupations >= n) or np.any(np.diff(occupations,axis=1) <= 0):
        raise ValueError('Invalid occupations.')
    d = len(occupations)
    result = np.zeros((d,d))
    for a, occ in enumerate(occupations):
        result[a,a] = 2 * np.trace(h[np.ix_(occ,occ)])
        for l in factors:
            block = l[np.ix_(occ,occ)]
            result[a,a] += 2*np.trace(block)**2 - np.trace(block@block)
        for b in range(a):
            other = occupations[b]
            removed = sorted(set(other)-set(occ))
            added = sorted(set(occ)-set(other))
            if len(removed) == len(added) == 1:
                value = np.sum(factors[:,added[0],removed[0]]**2)
                result[a,b] = result[b,a] = value
    return result

import numpy as np


def _oo_generators(n):
    pairs = [(p, q) for p in range(n) for q in range(p+1, n)]
    result = np.zeros((len(pairs), n, n))
    for a, (p, q) in enumerate(pairs):
        result[a, p, q] = 1.0
        result[a, q, p] = -1.0
    return result


def _oo_integral_jets(matrix, generators):
    m, n, _ = generators.shape
    first = np.zeros((m, n, n))
    second = np.zeros((m, m, n, n))
    for a, ka in enumerate(generators):
        first[a] = matrix @ ka - ka @ matrix
    for a, ka in enumerate(generators):
        for b, kb in enumerate(generators):
            second[a, b] = 0.5 * (
                first[a] @ kb - kb @ first[a]
                + first[b] @ ka - ka @ first[b]
            )
    return first, second


def _oo_projected_derivatives(h, factors, occupations, generators):
    h1, h2 = _oo_integral_jets(h, generators)
    factor_jets = [_oo_integral_jets(l, generators) for l in factors]
    m, nd = len(generators), len(occupations)
    first = np.zeros((m, nd, nd))
    second = np.zeros((m, m, nd, nd))
    for row, occ in enumerate(occupations):
        first[:, row, row] = 2*np.sum(h1[:, occ, occ], axis=-1)
        second[:, :, row, row] = 2*np.sum(h2[:, :, occ, occ], axis=-1)
        for l, (dl, ddl) in zip(factors, factor_jets):
            block = l[np.ix_(occ, occ)]
            dblock = dl[:, occ][:, :, occ]
            ddblock = ddl[:, :, occ][:, :, :, occ]
            trace = np.trace(block)
            dtrace = np.trace(dblock, axis1=-2, axis2=-1)
            ddtrace = np.trace(ddblock, axis1=-2, axis2=-1)
            first[:, row, row] += (
                4*trace*dtrace - 2*np.einsum('ij,aji->a', block, dblock)
            )
            second[:, :, row, row] += (
                4*np.outer(dtrace, dtrace) + 4*trace*ddtrace
                - 2*np.einsum('aij,bji->ab', dblock, dblock)
                - 2*np.einsum('ij,abji->ab', block, ddblock)
            )
        for col in range(row):
            added = set(occ) - set(occupations[col])
            removed = set(occupations[col]) - set(occ)
            if len(added) == len(removed) == 1:
                p, q = next(iter(added)), next(iter(removed))
                for l, (dl, ddl) in zip(factors, factor_jets):
                    value, dvalue, ddvalue = l[p,q], dl[:,p,q], ddl[:,:,p,q]
                    first[:, row, col] += 2*value*dvalue
                    second[:, :, row, col] += 2*(
                        np.outer(dvalue, dvalue) + value*ddvalue
                    )
                first[:, col, row] = first[:, row, col]
                second[:, :, col, row] = second[:, :, row, col]
    return first, second


def doci_orbital_response(h: "np.ndarray", factors: "np.ndarray", occupations: "np.ndarray", orbitals: "np.ndarray") -> "tuple[float, np.ndarray, np.ndarray, np.ndarray]":
    h = np.asarray(h, dtype=float)
    factors = np.asarray(factors, dtype=float)
    occupations = np.asarray(occupations, dtype=int)
    orbitals = np.asarray(orbitals, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not 1 <= len(h) <= 6:
        raise ValueError("h must be square with 1 <= n <= 6")
    n = len(h)
    if factors.ndim != 3 or factors.shape[1:] != (n,n) or not 1 <= len(factors) <= 8:
        raise ValueError("Incompatible two-electron factors")
    if (occupations.ndim != 2 or len(occupations) == 0
            or not 1 <= occupations.shape[1] <= n
            or len(np.unique(occupations, axis=0)) != len(occupations)):
        raise ValueError("Invalid occupations")
    if orbitals.shape != (n,n) or not np.allclose(
            orbitals.T @ orbitals, np.eye(n), rtol=0.0, atol=1e-10):
        raise ValueError("Orbitals must be orthogonal with shape (n,n)")
    transformed_h = orbitals.T @ h @ orbitals
    transformed_factors = np.asarray([orbitals.T @ l @ orbitals for l in factors])
    matrix = doci_matrix(transformed_h, transformed_factors, occupations)
    energies, vectors = np.linalg.eigh(matrix)
    if len(energies) > 1 and energies[1] - energies[0] <= 1e-10:
        raise ValueError("The projected ground state is not isolated")
    coefficients = vectors[:, 0].copy()
    if coefficients[np.argmax(np.abs(coefficients))] < 0:
        coefficients *= -1
    first, second = _oo_projected_derivatives(
        transformed_h, transformed_factors, occupations, _oo_generators(n)
    )
    gradient = np.einsum('i,aij,j->a', coefficients, first, coefficients)
    hessian = np.einsum('i,abij,j->ab', coefficients, second, coefficients)
    couplings = np.einsum('im,aij,j->am', vectors[:,1:], first, coefficients)
    hessian += 2*(couplings / (energies[0] - energies[1:])) @ couplings.T
    return float(energies[0]), coefficients, gradient, 0.5*(hessian+hessian.T)

import numpy as np
from scipy.linalg import expm


def relax_doci_trial(h: "np.ndarray", factors: "np.ndarray", npair: int, n_updates: int) -> "tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    h = np.asarray(h, dtype=float)
    factors = np.asarray(factors, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not 1 <= len(h) <= 6:
        raise ValueError("h must be square with 1 <= n <= 6")
    if (not isinstance(n_updates, (int, np.integer))
            or not 0 <= n_updates <= 6):
        raise ValueError("n_updates must be an integer from 0 through 6")
    if not isinstance(npair, (int, np.integer)) or not 1 <= npair <= len(h):
        raise ValueError("Invalid pair count")
    occupations = pair_occupations(len(h), npair)
    orbitals = np.eye(len(h))
    generators = _oo_generators(len(h))
    energy, coefficients, gradient, hessian = doci_orbital_response(
        h, factors, occupations, orbitals
    )
    energies = [energy]
    accepted_scales = np.zeros(n_updates)
    for update in range(n_updates):
        if np.linalg.norm(gradient) > 1e-10:
            shift = max(0.0, 0.05 - np.linalg.eigvalsh(hessian)[0])
            step = -np.linalg.solve(hessian + shift*np.eye(len(gradient)), gradient)
            norm = np.linalg.norm(step)
            if norm > 0.25:
                step *= 0.25/norm
            directional = float(gradient @ step)
            if -directional > 1e-14:
                for power in range(21):
                    alpha = 2.0**(-power)
                    rotation = expm(alpha*np.einsum('a,aij->ij', step, generators))
                    candidate = orbitals @ rotation
                    transformed_h = candidate.T @ h @ candidate
                    transformed_factors = np.asarray([
                        candidate.T @ l @ candidate for l in factors
                    ])
                    matrix = doci_matrix(
                        transformed_h, transformed_factors, occupations
                    )
                    candidate_energy = np.linalg.eigvalsh(matrix)[0]
                    if candidate_energy <= energy + 1e-4*alpha*directional:
                        orbitals = candidate
                        accepted_scales[update] = alpha
                        break
                else:
                    raise ValueError("No orbital line-search scale was accepted")
                energy, coefficients, gradient, hessian = doci_orbital_response(
                    h, factors, occupations, orbitals
                )
        energies.append(energy)
    return energy, coefficients, orbitals, np.asarray(energies), accepted_scales

from itertools import combinations
import numpy as np
from scipy.linalg import expm


def _spin_operator_numerators(overlap_matrices, operator_matrices):
    """Return det(S), first derivatives, and factor second derivatives.

    S has shape (nd,k,k). X has shape (nd,1+r,k,k), h first, factors next.
    The second result is d/dt det(S+t X)|0 for each operator. The final
    result is d^2/dt^2 det(S+t X)|0 for each two-electron factor. These
    polynomial derivatives are well-defined at singular S.
    """
    s = overlap_matrices
    x = operator_matrices
    k = s.shape[-1]
    det_s = np.linalg.det(s)
    if k == 1:
        first = x[..., 0, 0].copy()
        return det_s, first, np.zeros_like(first[:, 1:])
    idx = np.arange(k)
    keep = np.array([idx[idx != i] for i in idx], dtype=int)
    minors = s[:, keep[:, None, :, None], keep[None, :, None, :]]
    signs = (-1.0) ** (idx[:, None] + idx[None, :])
    cofactors = signs * np.linalg.det(minors)
    first = np.einsum('dij,dlij->dl', cofactors, x)
    pairs = np.array(list(combinations(range(k), 2)), dtype=int)
    i, j = pairs.T
    keep2 = np.array([idx[(idx != a) & (idx != b)] for a, b in pairs])
    if k == 2:
        second_cofactors = np.ones((len(s), 1, 1), dtype=complex)
    else:
        minors2 = s[:, keep2[:, None, :, None], keep2[None, :, None, :]]
        signs2 = (-1.0) ** (i[:, None] + j[:, None] + i[None, :] + j[None, :])
        second_cofactors = signs2 * np.linalg.det(minors2)
    y = x[:, 1:]
    wedge = (y[:, :, i[:, None], i[None, :]] * y[:, :, j[:, None], j[None, :]]
             - y[:, :, i[:, None], j[None, :]] * y[:, :, j[:, None], i[None, :]])
    second = 2.0 * np.einsum('dij,dlij->dl', second_cofactors, wedge)
    return det_s, first, second


def _mixed_estimators(occupations: "np.ndarray", coefficients: "np.ndarray", phi_alpha: "np.ndarray", phi_beta: "np.ndarray", h: "np.ndarray", factors: "np.ndarray", trial_orbitals: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    occ = np.asarray(occupations)
    cp = np.asarray(coefficients, dtype=float)
    pa = np.asarray(phi_alpha, dtype=float)
    pb = np.asarray(phi_beta, dtype=float)
    h = np.asarray(h, dtype=float)
    factors = np.asarray(factors, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not 1 <= len(h) <= 28:
        raise ValueError('Require a square one-electron matrix with n <= 28.')
    n = len(h)
    if (occ.ndim != 2 or not 1 <= len(occ) <= 32
            or not np.issubdtype(occ.dtype, np.integer)
            or not 1 <= occ.shape[1] <= min(n, 6)
            or np.any(occ < 0) or np.any(occ >= n)
            or np.any(np.diff(occ, axis=1) <= 0)
            or len(np.unique(occ, axis=0)) != len(occ)):
        raise ValueError('Invalid paired occupations or selected-trial dimensions.')
    if (cp.shape != (len(occ), 2) or pa.shape != (n, occ.shape[1], 2)
            or pb.shape != pa.shape or factors.ndim != 3
            or factors.shape[1:] != (n, n) or not 1 <= len(factors) <= 8):
        raise ValueError('Incompatible trial, walker, or factor shapes.')
    if not np.allclose(h, h.T) or not np.allclose(factors, factors.transpose(0, 2, 1)):
        raise ValueError('Hamiltonian matrices must be symmetric.')
    coeff = cp[:, 0] + 1j * cp[:, 1]
    a = pa[..., 0] + 1j * pa[..., 1]
    b = pb[..., 0] + 1j * pb[..., 1]
    if trial_orbitals is None:
        u = np.eye(n, dtype=complex)
    else:
        up = np.asarray(trial_orbitals, dtype=float)
        if up.shape != (n, n, 2):
            raise ValueError('Require packed trial orbitals with shape (n,n,2).')
        u = up[..., 0] + 1j * up[..., 1]
    udag = u.conj().T
    ops = np.concatenate((h[None], factors), axis=0)
    xa = (udag @ (ops @ a))[:, occ, :].transpose(1, 0, 2, 3)
    xb = (udag @ (ops @ b))[:, occ, :].transpose(1, 0, 2, 3)
    sa, first_a, second_a = _spin_operator_numerators((udag @ a)[occ, :], xa)
    sb, first_b, second_b = _spin_operator_numerators((udag @ b)[occ, :], xb)
    ca = coeff.conjugate()
    overlap = np.dot(ca, sa * sb)
    if abs(overlap) <= 1e-12:
        raise ValueError('Total trial overlap must exceed 1e-12.')
    linear_a, linear_b = first_a[:, 1:], first_b[:, 1:]
    one = sb * first_a[:, 0] + sa * first_b[:, 0]
    two = (sb[:, None] * second_a + sa[:, None] * second_b
           + 2.0 * linear_a * linear_b)
    energy = np.dot(ca, one + 0.5 * np.sum(two, axis=1)) / overlap
    force = 1j * np.einsum('d,dl->l', ca,
                          sb[:, None] * linear_a + sa[:, None] * linear_b) / overlap
    return (np.array([overlap.real, overlap.imag]),
            np.array([energy.real, energy.imag]),
            np.stack((force.real, force.imag), axis=-1))


def advance_walker(occupations: "np.ndarray", coefficients: "np.ndarray", phi_alpha: "np.ndarray", phi_beta: "np.ndarray", h: "np.ndarray", factors: "np.ndarray", field: "np.ndarray", dt: float, trial_orbitals: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    if dt < 0:
        raise ValueError('Require a nonnegative imaginary-time step.')
    old_overlap, old_energy, old_force = _mixed_estimators(
        occupations, coefficients, phi_alpha, phi_beta, h, factors, trial_orbitals)
    h = np.asarray(h, dtype=float)
    factors = np.asarray(factors, dtype=float)
    field = np.asarray(field, dtype=float)
    if field.shape != (len(factors),):
        raise ValueError('Require one real field coordinate per factor.')
    force = old_force[:, 0] + 1j*old_force[:, 1]
    shift = -np.sqrt(dt)*force
    h0 = h - 0.5*np.einsum('lpq,lqr->pr', factors, factors)
    half = expm(-0.5*dt*h0)
    middle = expm(1j*np.sqrt(dt)*np.einsum('l,lpq->pq', field-shift, factors))
    propagator = half @ middle @ half
    log_gaussian = np.dot(field, shift) - 0.5*np.dot(shift, shift)
    normalized = []
    gauge = 1.0 + 0.0j
    for packed in (phi_alpha, phi_beta):
        packed = np.asarray(packed, dtype=float)
        phi = packed[..., 0] + 1j*packed[..., 1]
        q, r = np.linalg.qr(propagator @ phi, mode='reduced')
        diagonal = np.diag(r)
        if np.min(np.abs(diagonal)) <= 1e-12:
            raise ValueError('Propagated QR diagonal must exceed 1e-12.')
        phases = diagonal/np.abs(diagonal)
        q = q*phases[None, :]
        r = phases.conjugate()[:, None]*r
        gauge *= np.prod(np.diag(r))
        normalized.append(np.stack((q.real, q.imag), axis=-1))
    q_alpha, q_beta = normalized
    new_overlap, new_energy, _ = _mixed_estimators(
        occupations, coefficients, q_alpha, q_beta, h, factors, trial_orbitals)
    return (q_alpha, q_beta, old_overlap, new_overlap, old_energy, new_energy,
            old_force, np.array([log_gaussian.real, log_gaussian.imag]),
            np.array([gauge.real, gauge.imag]))

import numpy as np


def phaseless_weight(old_overlap: "np.ndarray", new_overlap: "np.ndarray", gauge: "np.ndarray", log_gaussian: "np.ndarray", weight: float, dt: float, energy_shift: float) -> "tuple[float, float]":
    def _unpack(a):
        a = np.asarray(a,dtype=float)
        return a[0]+1j*a[1]
    old,new,g,lg = [_unpack(x) for x in (old_overlap,new_overlap,gauge,log_gaussian)]
    if abs(old)<=1e-12 or weight < 0 or dt < 0:
        raise ValueError('Nonzero old overlap, nonnegative weight and dt required.')
    ratio = g*new/old
    phase = float(np.angle(ratio))
    importance = ratio*np.exp(lg+dt*energy_shift)
    updated = weight*abs(importance)*max(0.0,np.cos(phase))
    return float(updated),phase

import numpy as np


def projection_block(h: "np.ndarray", factors: "np.ndarray", npair: int, walkers: "np.ndarray", weights: "np.ndarray", fields: "np.ndarray", dt: float, energy_shift: float, n_updates: int = 4) -> "tuple[float, np.ndarray, np.ndarray, np.ndarray]":
    h = np.asarray(h, dtype=float)
    factors = np.asarray(factors, dtype=float)
    walkers = np.array(walkers, dtype=float, copy=True)
    weights = np.array(weights, dtype=float, copy=True)
    fields = np.asarray(fields, dtype=float)
    n = len(h)
    if (walkers.ndim != 5 or walkers.shape[1:] != (2, n, npair, 2)
            or weights.shape != (len(walkers),) or fields.ndim != 3
            or fields.shape[1:] != (len(walkers), len(factors))
            or np.any(weights < 0) or np.sum(weights) <= 0 or dt < 0):
        raise ValueError('Invalid block dimensions or nonpositive total weight.')
    occ = pair_occupations(n, npair)
    trial_energy, coeff, orbitals, _, _ = relax_doci_trial(
        h, factors, npair, n_updates)
    cp = np.stack((coeff, np.zeros_like(coeff)), axis=-1)
    up = np.stack((orbitals, np.zeros_like(orbitals)), axis=-1)
    values = np.zeros(len(weights))
    for w in range(len(weights)):
        if weights[w] > 0:
            initial = advance_walker(
                occ, cp, walkers[w, 0], walkers[w, 1], h, factors,
                np.zeros(len(factors)), 0.0, up)
            values[w] = initial[4][0]
    trajectory, history, angles = [], [], []

    def _measure():
        total = weights.sum()
        if total <= 1e-14:
            raise ValueError('All walkers have died (total weight <= 1e-14).')
        trajectory.append(float(np.dot(weights, values) / total))
        history.append(weights.copy())

    _measure()
    for row in fields:
        phases = np.zeros(len(weights))
        for w, x in enumerate(row):
            if weights[w] == 0:
                continue
            pa, pb, old, new, _, el, _, lg, gauge = advance_walker(
                occ, cp, walkers[w, 0], walkers[w, 1], h, factors, x, dt, up)
            weights[w], phases[w] = phaseless_weight(
                old, new, gauge, lg, float(weights[w]), dt, energy_shift)
            walkers[w, 0], walkers[w, 1] = pa, pb
            values[w] = el[0]
        angles.append(phases)
        _measure()
    return (trial_energy, np.array(trajectory), np.array(history),
            np.array(angles).reshape(len(fields), len(weights)))

def solve_doci_projection(h: "np.ndarray", factors: "np.ndarray", npair: int, walkers: "np.ndarray", weights: "np.ndarray", fields: "np.ndarray", dt: float, energy_shift: float, n_updates: int = 4) -> float:
    trial_energy,trajectory,_,_ = projection_block(h,factors,npair,walkers,weights,fields,dt,energy_shift,n_updates)
    return float(1000*(trajectory[-1]-trial_energy))
SCICODE_GOLD_EOF
