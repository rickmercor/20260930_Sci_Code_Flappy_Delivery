#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def symmetric_eigensystem(matrix: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _sweep in range(100):
        off = sum(a[p][q]*a[p][q] for p in range(n) for q in range(n) if p != q)
        if off < 1e-28:
            break
        for p in range(n):
            for q in range(p+1, n):
                apq = a[p][q]
                if abs(apq) < 1e-300:
                    continue
                theta = (a[q][q]-a[p][p])/(2.0*apq)
                t = (1.0 if theta >= 0.0 else -1.0)/(abs(theta)+(theta*theta+1.0)**0.5)
                c = 1.0/((t*t+1.0)**0.5)
                s = t*c
                app, aqq = a[p][p], a[q][q]
                a[p][p] = c*c*app - 2.0*s*c*apq + s*s*aqq
                a[q][q] = s*s*app + 2.0*s*c*apq + c*c*aqq
                a[p][q] = 0.0
                a[q][p] = 0.0
                for k in range(n):
                    if k != p and k != q:
                        akp, akq = a[k][p], a[k][q]
                        a[k][p] = c*akp - s*akq
                        a[p][k] = a[k][p]
                        a[k][q] = s*akp + c*akq
                        a[q][k] = a[k][q]
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c*vkp - s*vkq
                    v[k][q] = s*vkp + c*vkq
    eigvals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigvals[i])
    eigvals_sorted = [eigvals[i] for i in order]
    eigvecs_sorted = []
    for i in order:
        vec = [v[k][i] for k in range(n)]
        norm = sum(x*x for x in vec)**0.5
        if norm > 0.0:
            vec = [x/norm for x in vec]
        best_k = 0
        best_mag = -1.0
        for k in range(n):
            mag = abs(vec[k])
            if mag > best_mag + 1e-14:
                best_mag = mag
                best_k = k
        if vec[best_k] < 0.0:
            vec = [-x for x in vec]
        eigvecs_sorted.append(vec)
    return (eigvals_sorted, eigvecs_sorted)

def direct_rpa(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    ov = [(i, a) for i in occ for a in vir if i % 2 == a % 2]
    npairs = len(ov)

    def _phys(p, q, r, s):
        if p % 2 != r % 2 or q % 2 != s % 2:
            return 0.0
        return chem_spatial.get((p//2, r//2, q//2, s//2), 0.0)

    A = [[0.0]*npairs for _ in range(npairs)]
    B = [[0.0]*npairs for _ in range(npairs)]
    for I, (i, a) in enumerate(ov):
        for J, (j, b) in enumerate(ov):
            A[I][J] = (eps[a]-eps[i])*(1.0 if (i == j and a == b) else 0.0) + _phys(a, j, i, b)
            B[I][J] = _phys(a, b, i, j)
    AmB = [[A[p][q]-B[p][q] for q in range(npairs)] for p in range(npairs)]
    ApB = [[A[p][q]+B[p][q] for q in range(npairs)] for p in range(npairs)]

    wA, vA = symmetric_eigensystem(AmB)
    # vA[k] is the k-th eigenvector (a list over components); vA[k][p] is its p-th component.
    sqrtwA = [max(w, 1e-300)**0.5 for w in wA]
    s_mat = [[sum(vA[k][p]*sqrtwA[k]*vA[k][q] for k in range(npairs)) for q in range(npairs)] for p in range(npairs)]
    sinv_mat = [[sum(vA[k][p]*(1.0/sqrtwA[k])*vA[k][q] for k in range(npairs)) for q in range(npairs)] for p in range(npairs)]

    prod = [[sum(s_mat[p][k]*ApB[k][q] for k in range(npairs)) for q in range(npairs)] for p in range(npairs)]
    Mmid = [[sum(prod[p][k]*s_mat[k][q] for k in range(npairs)) for q in range(npairs)] for p in range(npairs)]
    Mmid = [[0.5*(Mmid[p][q]+Mmid[q][p]) for q in range(npairs)] for p in range(npairs)]

    o2, T = symmetric_eigensystem(Mmid)
    omega = [max(v, 0.0)**0.5 for v in o2]

    X = [[0.0]*npairs for _ in range(npairs)]
    Y = [[0.0]*npairs for _ in range(npairs)]
    for nu in range(npairs):
        om = max(omega[nu], 1e-300)
        xpy = [sum(s_mat[p][k]*T[nu][k] for k in range(npairs))/(om**0.5) for p in range(npairs)]
        xmy = [sum(sinv_mat[p][k]*T[nu][k] for k in range(npairs))*(om**0.5) for p in range(npairs)]
        for p in range(npairs):
            X[p][nu] = 0.5*(xpy[p]+xmy[p])
            Y[p][nu] = 0.5*(xpy[p]-xmy[p])

    order = sorted(range(npairs), key=lambda nu: omega[nu])
    omega_sorted = [omega[nu] for nu in order]
    X_sorted = [[X[p][nu] for p in range(npairs)] for nu in order]
    Y_sorted = [[Y[p][nu] for p in range(npairs)] for nu in order]
    return (omega_sorted, X_sorted, Y_sorted)

def gw_effective_integrals(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
    omega: list[float],
    X: list[list[float]],
    Y: list[list[float]],
) -> list[list[list[float]]]:
    ov = [(i, a) for i in occ for a in vir if i % 2 == a % 2]
    n = len(eps)
    nexc = len(omega)

    def _phys(p, q, r, s):
        if p % 2 != r % 2 or q % 2 != s % 2:
            return 0.0
        return chem_spatial.get((p//2, r//2, q//2, s//2), 0.0)

    M = [[[0.0]*nexc for _ in range(n)] for _ in range(n)]
    for nu in range(nexc):
        for p in range(n):
            for q in range(n):
                M[p][q][nu] = sum(
                    _phys(p, a, q, i)*X[nu][I] + _phys(p, i, q, a)*Y[nu][I]
                    for I, (i, a) in enumerate(ov)
                )
    return M

def adc_gw_tier1_matrix(
    eps: list[float],
    occ: list[int],
    vir: list[int],
    omega: list[float],
    M: list[list[list[float]]],
    probe: int,
) -> list[list[float]]:
    nexc = len(omega)
    d = 1 + len(occ)*nexc + len(vir)*nexc
    H = [[0.0]*d for _ in range(d)]
    H[0][0] = eps[probe]
    row = 1
    for j in occ:
        for nu in range(nexc):
            H[row][row] = eps[j] - omega[nu]
            u = M[probe][j][nu]
            H[0][row] = u
            H[row][0] = u
            row += 1
    for b in vir:
        for nu in range(nexc):
            H[row][row] = eps[b] + omega[nu]
            u = M[b][probe][nu]
            H[0][row] = u
            H[row][0] = u
            row += 1
    return H

def adc_2sosex_matrix(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
    omega: list[float],
    M: list[list[list[float]]],
    probe: int,
) -> list[list[float]]:
    H = adc_gw_tier1_matrix(eps, occ, vir, omega, M, probe)
    nexc = len(omega)

    def _phys(p, q, r, s):
        if p % 2 != r % 2 or q % 2 != s % 2:
            return 0.0
        return chem_spatial.get((p//2, r//2, q//2, s//2), 0.0)

    def _safe_term(m_factor, rest, denom):
        if abs(m_factor) < 1e-9 and abs(denom) < 1e-9:
            return 0.0
        return m_factor*rest/denom

    row = 1
    for j in occ:
        for nu in range(nexc):
            add = 0.0
            for k in occ:
                for c in vir:
                    add += _safe_term(M[c][k][nu], _phys(j, c, k, probe), eps[c]-eps[k]+omega[nu])
                    add += _safe_term(M[k][c][nu], _phys(j, k, c, probe), eps[c]-eps[k]-omega[nu])
            H[0][row] += add
            H[row][0] += add
            row += 1
    for b in vir:
        for nu in range(nexc):
            add = 0.0
            for k in occ:
                for c in vir:
                    add += _safe_term(M[c][k][nu], _phys(b, k, c, probe), eps[c]-eps[k]+omega[nu])
                    add += _safe_term(M[k][c][nu], _phys(b, c, k, probe), eps[c]-eps[k]-omega[nu])
            H[0][row] += add
            H[row][0] += add
            row += 1
    return H

def adc3_g3w2_matrix(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
    omega: list[float],
    M: list[list[list[float]]],
    probe: int,
) -> list[list[float]]:
    H = adc_2sosex_matrix(eps, chem_spatial, occ, vir, omega, M, probe)
    nexc = len(omega)

    def _safe_term(m1, m2, denom):
        if (abs(m1) < 1e-9 or abs(m2) < 1e-9) and abs(denom) < 1e-9:
            return 0.0
        return m1*m2/denom

    row0 = 1
    idx_h = {}
    for j in occ:
        for nu in range(nexc):
            idx_h[(j, nu)] = row0
            row0 += 1
    idx_p = {}
    for b in vir:
        for nu in range(nexc):
            idx_p[(b, nu)] = row0
            row0 += 1

    for (j, nu), I in idx_h.items():
        for (k, mu), J in idx_h.items():
            c1 = 0.0
            for c in vir:
                c1 += 0.5*_safe_term(M[j][c][mu], M[k][c][nu], eps[j]-eps[c]+omega[mu])
                c1 += 0.5*_safe_term(M[j][c][mu], M[k][c][nu], eps[k]-eps[c]+omega[nu])
            if c1 != 0.0:
                H[I][J] += c1
    for (b, nu), I in idx_p.items():
        for (d, mu), J in idx_p.items():
            c1 = 0.0
            for k in occ:
                c1 += 0.5*_safe_term(M[k][b][mu], M[k][d][nu], eps[b]-eps[k]-omega[mu])
                c1 += 0.5*_safe_term(M[k][b][mu], M[k][d][nu], eps[d]-eps[k]-omega[nu])
            if c1 != 0.0:
                H[I][J] += c1
    return H

def adc_g3w2_third_coupling(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
    omega: list[float],
    M: list[list[list[float]]],
    probe: int,
) -> list[list[float]]:
    H = adc3_g3w2_matrix(eps, chem_spatial, occ, vir, omega, M, probe)
    nexc = len(omega)

    def _safe3(m1, m2, m3, d1, d2):
        if (abs(m1) < 1e-9 or abs(m2) < 1e-9 or abs(m3) < 1e-9) and (abs(d1) < 1e-9 or abs(d2) < 1e-9):
            return 0.0
        return m1*m2*m3/(d1*d2)

    row = 1
    for j in occ:
        for nu in range(nexc):
            u = 0.0
            for k in occ:
                for c in vir:
                    for mu in range(nexc):
                        u += 0.5*_safe3(M[j][c][mu], M[k][c][nu], M[probe][k][mu],
                                       eps[c]-eps[k]-omega[nu], eps[c]-eps[j]-omega[mu])
                        u -= _safe3(M[c][j][mu], M[k][c][nu], M[k][probe][mu],
                                   eps[c]-eps[k]-omega[nu], eps[c]-eps[j]+omega[mu])
                        u -= _safe3(M[k][j][mu], M[c][k][nu], M[c][probe][mu],
                                   eps[c]-eps[j]+omega[nu]+omega[mu], eps[c]-eps[k]+omega[nu])
            for c in vir:
                for d in vir:
                    for mu in range(nexc):
                        u += _safe3(M[d][j][mu], M[c][d][nu], M[c][probe][mu],
                                   eps[c]-eps[j]+omega[nu]+omega[mu], eps[d]-eps[j]+omega[mu])
            H[0][row] += u
            H[row][0] += u
            row += 1
    for b in vir:
        for nu in range(nexc):
            u = 0.0
            for k in occ:
                for c in vir:
                    for mu in range(nexc):
                        u += 0.5*_safe3(M[k][b][mu], M[k][c][nu], M[c][probe][mu],
                                       eps[c]-eps[k]-omega[nu], eps[b]-eps[k]-omega[mu])
                        u -= _safe3(M[b][k][mu], M[k][c][nu], M[probe][c][mu],
                                   eps[c]-eps[k]-omega[nu], eps[b]-eps[k]+omega[mu])
                        u -= _safe3(M[b][c][mu], M[c][k][nu], M[probe][k][mu],
                                   eps[b]-eps[k]+omega[nu]+omega[mu], eps[c]-eps[k]+omega[nu])
            for k in occ:
                for l in occ:
                    for mu in range(nexc):
                        u += _safe3(M[b][l][mu], M[l][k][nu], M[probe][k][mu],
                                   eps[b]-eps[k]+omega[nu]+omega[mu], eps[b]-eps[l]+omega[mu])
            H[0][row] += u
            H[row][0] += u
            row += 1
    return H

def three_hole_two_particle_augmented_matrix(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
    omega: list[float],
    M: list[list[list[float]]],
    probe: int,
) -> list[list[float]]:
    H_small = adc_g3w2_third_coupling(eps, chem_spatial, occ, vir, omega, M, probe)
    nexc = len(omega)
    d_small = len(H_small)
    n_h3 = len(occ)*nexc*nexc
    n_p3 = len(vir)*nexc*nexc
    d = d_small + n_h3 + n_p3
    H = [[0.0]*d for _ in range(d)]
    for i in range(d_small):
        for j in range(d_small):
            H[i][j] = H_small[i][j]

    def _safe_term(m1, m2, denom):
        if (abs(m1) < 1e-9 or abs(m2) < 1e-9) and abs(denom) < 1e-9:
            return 0.0
        return m1*m2/denom

    idx_h = {}
    r = 1
    for j in occ:
        for nu in range(nexc):
            idx_h[(j, nu)] = r
            r += 1
    idx_p = {}
    for b in vir:
        for nu in range(nexc):
            idx_p[(b, nu)] = r
            r += 1

    row = d_small
    for j in occ:
        for nu in range(nexc):
            for mu in range(nexc):
                H[row][row] = eps[j] - omega[nu] - omega[mu]
                u2 = 0.0
                for c in vir:
                    u2 += -_safe_term(M[c][j][mu], M[probe][c][nu], eps[c]-eps[j]+omega[mu])
                H[0][row] = u2
                H[row][0] = u2
                for k in occ:
                    lam = nu
                    if (k, lam) in idx_h:
                        Jc = idx_h[(k, lam)]
                        c21 = M[k][j][mu]
                        if c21 != 0.0:
                            H[row][Jc] += c21
                            H[Jc][row] += c21
                row += 1
    for b in vir:
        for nu in range(nexc):
            for mu in range(nexc):
                H[row][row] = eps[b] + omega[nu] + omega[mu]
                u2 = 0.0
                for k in occ:
                    u2 += _safe_term(M[b][k][mu], M[k][probe][nu], eps[b]-eps[k]+omega[mu])
                H[0][row] = u2
                H[row][0] = u2
                for d_ in vir:
                    lam = nu
                    if (d_, lam) in idx_p:
                        Jc = idx_p[(d_, lam)]
                        c21 = M[d_][b][mu]
                        if c21 != 0.0:
                            H[row][Jc] += c21
                            H[Jc][row] += c21
                row += 1
    return H

def adc_g3w2_ionization_potential(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
    probe: int,
) -> float:
    omega, X, Y = direct_rpa(eps, chem_spatial, occ, vir)
    M = gw_effective_integrals(eps, chem_spatial, occ, vir, omega, X, Y)
    H = three_hole_two_particle_augmented_matrix(eps, chem_spatial, occ, vir, omega, M, probe)
    evals, evecs = symmetric_eigensystem(H)
    best_i = 0
    best_w = -1.0
    for i in range(len(evals)):
        w = evecs[i][0]*evecs[i][0]
        if w > best_w:
            best_w = w
            best_i = i
    return -evals[best_i]
SCICODE_GOLD_EOF
