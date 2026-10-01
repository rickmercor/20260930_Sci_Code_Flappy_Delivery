"""
Construct the regular quark-pair and polarization jets in any of the seven sectors.

Implement the scalar radiators of Sections III.C and Appendix A, in the back-to-back Born configuration. Set the Born invariant mass squared to one, kappa=-1, and strip color and coupling factors. The normalization is that of the paper's Eq. (18). Points x=(t,a,b,r,chi) have 0<=t,a,b,r<1 and 0<chi<1; the definitions below hold at interior points, and zero faces are defined by continuity below. Set `xi1=t` and `xi2=t*y`. The seven sector maps, their absolute four-coordinate Jacobian Jmap, and the epsilon rates c=(c_t,c_a,c_b,c_r) are: I1: `eta1=a/2, eta2=a*b/4, y=r, Jmap=t*a/8, c=(4,2,1,2)`. I2: `eta1=a*b/4, eta2=b/2, y=a*r, Jmap=t*a*b/8, c=(4,3,2,2)`. I3: `eta1=a*b*r/4, eta2=b/2, y=r, Jmap=t*b*r/8, c=(4,1,2,3)`. I4: `eta1=a/2, eta2=(a/2)*(1-b/2), y=r, Jmap=t*a/8, c=(4,2,2,2)`. I5: `eta1=(b/2)*(1-a/2), eta2=b/2, y=r, Jmap=t*b/8, c=(4,2,2,2)`. II1: `eta1=a/2, eta2=1-b/2, y=a*r/2, Jmap=t*a/8, c=(4,3,1,2)`. II2: `eta1=a*min(r,1/2), eta2=1-b/2, y=r, Jmap=t*min(r,1/2)/2, c=(4,1,1,3)`. These include the alpha=0 hemisphere selectors that Section IV takes; Section 2.4 imposes the sharp cut nz1>0 in both sector types; with `eta=(1-cos(theta))/2` that cut is `eta1<1/2`. They cover xi2<xi1 and eta1<1/2 exactly once, with eta2 in (0,1). For `eta1<eta2/2` and `eta2<1/2`, write `q=2*eta1/eta2`. I2 covers `y<q`; I3 covers `q<y` through `q=a*r` and `y=r`. Thus I3 uses the ratio r in the angular map; reading the unhatted xi2 in the I3 entry of Table V as the physical xi2=t*y would leave `t*y<q<y` uncovered. The cap in II2 is that sharp cut applied to the uncapped map `eta1=etahat1*y` that Table 1 gives for the second double-collinear sector; it retains the sector's selector-cut domain and is continuous at r=1/2. The two unrepresented energy/hemisphere symmetries will be included by a factor four only after summing all seven sectors. Define `R=sqrt(eta1*(1-eta1)*eta2*(1-eta2))`, `delta=abs(eta1-eta2)`, `A=eta1+eta2-2*eta1*eta2`, `B=2*R`, `emin=A-B`, `eta3=emin+4*chi*R`, `eta12=delta**2/eta3`, `sinphi=2*delta*sqrt(chi*(1-chi))/eta3`, and `Jphi=delta/(eta3*sqrt(chi*(1-chi)))`. Because `A**2-B**2=delta**2`, evaluate `emin` as `delta**2/(A+B)`. In I4 and I5 use the exact difference `delta=a*b/4`: the order-16 rule nodes reach 1.5e-7 on each singular axis and 5e-12 in chi, where direct subtraction loses about half the digits of delta and nearly all digits of A-B. Solving Eq. (A27) for cosphi, with `cos(theta_i)=1-2*eta_i`, gives `cosphi=(2*A*chi-A+B)/eta3` with `eta3=A-B+2*B*chi`, and differentiation yields Jphi. Its epsilon-zero integral is pi; the reciprocal factor printed in Eq. (A26) would give `pi*A/delta`. Eqs. (26) and (36) print this factor with that source's eta3, which is the relative-angle variable called eta12 here. Set `L=1-xi1*xi2*(1-eta12)`, `z=(1-xi1)*(1-xi2)/L`, `rho1=xi1/(1-xi1)`, and `rho2=xi2/(1-xi2)`. The invariants are `sik=z`, `si1=z*rho1*eta1`, `si2=z*rho2*eta2`, `sk1=z*rho1*(1-eta1)`, `sk2=z*rho2*(1-eta2)`, `s12=z*rho1*rho2*eta12`, `si12=si1+si2+s12`, and `sk12=sk1+sk2+s12`. For an invariant kernel V(eps), the normalized density with respect to dt da db dr dchi is the following single product: `D_V(eps)=exp(2*EulerGamma*eps)/Gamma(1-2*eps) * 2**(-2*eps)/(4*pi) * (z*xi1*xi2)**(1-2*eps)/L**(3-2*eps) * (eta1*(1-eta1)*eta2*(1-eta2))**(-eps) * sinphi**(-2*eps) * Jphi * Jmap * V(eps)`. Epsilon is defined by the space-time dimension `4-2*eps`. EulerGamma is the Euler-Mascheroni constant. No additional solid-angle, spin-average or particle-identity factor is applied. Define the regular factor by `F_V(eps)=sqrt(chi*(1-chi))*t**(1+c_t*eps)*a**(1+c_a*eps)*b**(1+c_b*eps)*r**(1+c_r*eps)*D_V(eps)`. Stripping these regulating powers is the measure factorization of Eq. (54), whose exponents are its Table 2 entries and whose remaining regular function is expanded in a Taylor series in epsilon. Thus the angular measure for the regular jets is `dchi/sqrt(chi*(1-chi))`; the 1/pi in D_V is retained inside F_V. Return Taylor coefficients, not derivatives, in ascending epsilon powers zero through four. Values on every combination of t,a,b,r=0 are the continuous limits of these coefficients for fixed interior chi and fixed remaining coordinates. Some faces vanish because a factored pole is compensated by a positive power in F_V. Derive those finite limits algebraically; evaluating a singular invariant expression at a positive cutoff does not define a face. Rows with a coordinate equal to 1 or with chi equal to 0 or 1 are outside the domain. The two epsilon-independent kernels needed first are the scalar quark-pair radiator `Sqq=-2/(si12*sk12)*(2*sik/s12-1)+A2` from Eq. (31), and its squared polarization part `A2=((si2-si1)/(si12*s12)-(sk2-sk1)/(sk12*s12))**2`. For each input point return the two five-entry Taylor series F_Sqq and F_A2; these are the regular jets. Array arguments in every step have real numeric dtype; complex arrays, including qqbar_jets in Step 2 and rule and jets in Step 4, raise ValueError before conversion. Raise ValueError for invalid shapes, nonfinite values, an invalid sector or inputs outside the stated domain. Empty arrays with the stated column counts are accepted.

Returns
-------
jets : ndarray, shape (N,2,5) F_Sqq and F_A2 Taylor coefficients through eps degree four.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_qqbar_sector_jets(sector: str, points: np.ndarray) -> np.ndarray:
    """Construct the regular quark-pair and polarization jets in any of the seven sectors.

    Parameters
    ----------
    sector : str
        I1, I2, I3, I4, I5, II1 or II2.
    points : array_like, shape (N,5)
        Real (t,a,b,r,chi) rows, 0<=t,a,b,r<1 and 0<chi<1.

    Returns
    -------
    jets : ndarray, shape (N,2,5)
        F_Sqq and F_A2 Taylor coefficients through eps degree four.

    Raises
    ------
    ValueError
        Invalid sector, complex dtype, invalid shape, nonfinite input, or input outside the stated domain.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _c_assemble_qqbar_sector_jets__sector_components(sector, points):
    import numpy as np
    names = ('I1', 'I2', 'I3', 'I4', 'I5', 'II1', 'II2')
    if not isinstance(sector, str) or sector not in names:
        raise ValueError('one of the seven named sectors')
    x = np.asarray(points)
    if x.dtype.kind not in 'iuf':
        raise ValueError('real numeric points required')
    x = x.astype(np.result_type(x.dtype, float))
    if x.ndim != 2 or x.shape[1] != 5 or (not np.all(np.isfinite(x))) or np.any(x[:, :4] < 0) or np.any(x[:, :4] >= 1) or np.any(x[:, 4] <= 0) or np.any(x[:, 4] >= 1):
        raise ValueError('finite unit-cube points with permitted zero faces')
    t, a, b, r, chi = x.T
    result = np.zeros((len(x), 5), dtype=x.dtype)
    if sector in names[:5]:
        if sector == 'I1':
            u, v, y, sign = (a / 2, b / 2, r, 1)
            delta, sv, active = (1 - b / 2, np.sqrt(b) / np.sqrt(2.0), b > 0)
        elif sector == 'I2':
            u, v, y, sign = (b / 2, a / 2, a * r, -1)
            delta, sv, active = (1 - a / 2, np.sqrt(a) / np.sqrt(2.0), a > 0)
        elif sector == 'I3':
            u, v, y, sign = (b / 2, a * r / 2, r, -1)
            delta, sv, active = (1 - a * r / 2, np.sqrt(a) * np.sqrt(r) / np.sqrt(2.0), (a > 0) & (r > 0))
        elif sector == 'I4':
            u, v, y, sign = (a / 2, 1 - b / 2, r, 1)
            delta, sv, active = (b / 2, np.sqrt(v), np.ones(len(x), dtype=bool))
        else:
            u, v, y, sign = (b / 2, 1 - a / 2, r, -1)
            delta, sv, active = (a / 2, np.sqrt(v), np.ones(len(x), dtype=bool))
        A, B = (np.ones(len(x)), v) if sign == 1 else (v, np.ones(len(x)))
        angular = 1 + v - 2 * u * v
        remainder = (1 - u) * (1 - u * v)
        root = sv * np.sqrt(remainder)
        sg = np.hypot(delta / np.sqrt(angular + 2 * root), 2 * np.sqrt(chi) * np.sqrt(root))
        c = (delta / sg) ** 2
        d1 = 1 - t
        dy = 1 - a + a * (1 - r) if sector == 'I2' else 1 - r
        d2 = d1 + t * dy
        D = d1 * d2
        L = d1 + t * d2 + t * t * y * u * c
        z = D / L
        if sector in names[:3]:
            ell = -2 * np.log(z) + 2 * np.log(L) + 4 * np.log(sg) - np.log(remainder) - np.log(chi) - np.log1p(-chi) - np.log(2.0) - 2 * np.log(delta)
            multiplier = v / delta
        else:
            ell = -2 * np.log(z) + 2 * np.log(L) + 4 * np.log(sg) - np.log(v * remainder) - np.log(chi) - np.log1p(-chi)
            multiplier = np.ones(len(x))
        E = A * d2 + y * B * d1 + t * y * c
        F = (1 - u * A) * d2 + y * (1 - u * B) * d1 + t * y * u * c
        if sector == 'I2':
            yE = r / (d2 / 2 + r * d1 + t * r * c)
        elif sector == 'I3':
            yE = 1 / (a * d2 / 2 + d1 + t * c)
        else:
            yE = y / E
        base = 1 - a if sector in ('I1', 'I4') else 1 - b
        J = base * dy + 2 * u * delta * (d2 if sign == -1 else -y * d1)
        radial = d1 / L * (d2 / L) / np.pi
        scale = multiplier * radial
        spin = scale * (yE / F * (sg * D - t * sign * (delta / sg) * J / 2)) ** 2
        qq = scale * yE / F * (-D + y * t * t * u * c / 2) + spin
    else:
        cap = np.full(len(x), 0.5) if sector == 'II1' else np.minimum(r, 0.5)
        p1, p2, q1, q2 = (a * cap, 1 - b / 2, 1 - a * cap, b / 2)
        y = a * r / 2 if sector == 'II1' else r
        active = (a > 0) & (b > 0)
        if sector == 'II2':
            active &= r > 0
        delta = (1 - b) / 2 + (0.5 - cap) + cap * (1 - a)
        angular = p1 * q2 + p2 * q1
        small_root = np.sqrt(a) * np.sqrt(cap) * np.sqrt(b) / np.sqrt(2.0)
        root = small_root * np.sqrt(q1 * p2)
        sg = np.hypot(delta / np.sqrt(angular + 2 * root), 2 * np.sqrt(chi) * np.sqrt(root))
        h = (delta / sg) ** 2
        d1 = 1 - t
        d2 = 1 - t * y if sector == 'II1' else d1 + t * (1 - r)
        D = d1 * d2
        L = d1 + t * d2 + t * t * y * h
        z = D / L
        multiplier = small_root / delta * small_root
        ell = -2 * np.log(z) + 2 * np.log(L) + 4 * np.log(sg) - np.log(q1 * p2) - 2 * np.log(delta) - np.log(chi) - np.log1p(-chi)
        if sector == 'II1':
            yE = r / (d2 + r * p2 * d1 + t * r * h)
        else:
            ratio = np.divide(0.5, r, out=np.ones(len(x), dtype=x.dtype), where=r > 0.5)
            yE = 1 / (a * ratio * d2 + p2 * d1 + t * h)
            ell += -3 * np.log(2.0) - np.log(ratio)
        F = q1 * d2 + y * q2 * d1 + t * y * h
        J = d2 * (1 - a + a * (1 - 2 * cap)) + y * d1 * (1 - b)
        radial = d1 / L * (d2 / L) / np.pi
        scale = multiplier * radial
        spin = scale * (yE / F * (sg * D + t * (delta / sg) * J / 2)) ** 2
        qq = scale * yE / F * (-D + y * t * t * h / 2) + spin
    result[:, 4] = ell
    result[active, :2] = np.column_stack([qq, spin])[active]
    return result

def _c_assemble_qqbar_sector_jets__gamma_jet(ell):
    import numpy as np
    from scipy.special import zeta
    z2, z3, z4 = (np.pi ** 2 / 6, float(zeta(3.0, 1.0)), np.pi ** 4 / 90)
    return np.column_stack([np.ones(len(ell)), ell, ell ** 2 / 2 - 2 * z2, ell ** 3 / 6 - 2 * z2 * ell - 8 * z3 / 3, ell ** 4 / 24 - z2 * ell ** 2 - 8 * z3 * ell / 3 + z4])

def _oracle_assemble_qqbar_sector_jets(sector: str, points: np.ndarray) -> np.ndarray:
    components = _c_assemble_qqbar_sector_jets__sector_components(sector, points)
    gamma = _c_assemble_qqbar_sector_jets__gamma_jet(components[:, 4])
    return components[:, :2, None] * gamma[:, None, :]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': "sector='I1'\nimport numpy as np\npoints=np.array([[0.37, 0.61, 0.49, 0.42, 0.29], [0.72, 0.18, 0.81, 0.74, 0.73], [0.14, 0.46, 0.27, 0.5, 0.57]])\n", 'call': '#case:normal\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I1'\nimport numpy as np\npoints=np.array([[0.0, 0.61, 0.49, 0.42, 0.29], [0.37, 0.0, 0.49, 0.42, 0.29], [0.0, 0.0, 0.49, 0.42, 0.29], [0.37, 0.61, 0.0, 0.42, 0.29], [0.0, 0.61, 0.0, 0.42, 0.29], [0.37, 0.0, 0.0, 0.42, 0.29], [0.0, 0.0, 0.0, 0.42, 0.29], [0.37, 0.61, 0.49, 0.0, 0.29], [0.0, 0.61, 0.49, 0.0, 0.29], [0.37, 0.0, 0.49, 0.0, 0.29], [0.0, 0.0, 0.49, 0.0, 0.29], [0.37, 0.61, 0.0, 0.0, 0.29], [0.0, 0.61, 0.0, 0.0, 0.29], [0.37, 0.0, 0.0, 0.0, 0.29], [0.0, 0.0, 0.0, 0.0, 0.29]])\n", 'call': '#case:boundary\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I1'\nimport numpy as np\npoints=np.array([[1e-05, 0.46, 0.27, 0.83, 0.57], [0.14, 1e-05, 0.27, 0.83, 0.57], [0.14, 0.46, 1e-05, 0.83, 0.57], [0.14, 0.46, 0.27, 1e-05, 0.57]])\n", 'call': '#case:edge\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I2'\nimport numpy as np\npoints=np.array([[0.37, 0.61, 0.49, 0.42, 0.29], [0.72, 0.18, 0.81, 0.74, 0.73], [0.14, 0.46, 0.27, 0.5, 0.57]])\n", 'call': '#case:normal\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I2'\nimport numpy as np\npoints=np.array([[0.0, 0.61, 0.49, 0.42, 0.29], [0.37, 0.0, 0.49, 0.42, 0.29], [0.0, 0.0, 0.49, 0.42, 0.29], [0.37, 0.61, 0.0, 0.42, 0.29], [0.0, 0.61, 0.0, 0.42, 0.29], [0.37, 0.0, 0.0, 0.42, 0.29], [0.0, 0.0, 0.0, 0.42, 0.29], [0.37, 0.61, 0.49, 0.0, 0.29], [0.0, 0.61, 0.49, 0.0, 0.29], [0.37, 0.0, 0.49, 0.0, 0.29], [0.0, 0.0, 0.49, 0.0, 0.29], [0.37, 0.61, 0.0, 0.0, 0.29], [0.0, 0.61, 0.0, 0.0, 0.29], [0.37, 0.0, 0.0, 0.0, 0.29], [0.0, 0.0, 0.0, 0.0, 0.29]])\n", 'call': '#case:boundary\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I2'\nimport numpy as np\npoints=np.array([[1e-05, 0.46, 0.27, 0.83, 0.57], [0.14, 1e-05, 0.27, 0.83, 0.57], [0.14, 0.46, 1e-05, 0.83, 0.57], [0.14, 0.46, 0.27, 1e-05, 0.57]])\n", 'call': '#case:edge\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I3'\nimport numpy as np\npoints=np.array([[0.37, 0.61, 0.49, 0.42, 0.29], [0.72, 0.18, 0.81, 0.74, 0.73], [0.14, 0.46, 0.27, 0.5, 0.57]])\n", 'call': '#case:normal\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I3'\nimport numpy as np\npoints=np.array([[0.0, 0.61, 0.49, 0.42, 0.29], [0.37, 0.0, 0.49, 0.42, 0.29], [0.0, 0.0, 0.49, 0.42, 0.29], [0.37, 0.61, 0.0, 0.42, 0.29], [0.0, 0.61, 0.0, 0.42, 0.29], [0.37, 0.0, 0.0, 0.42, 0.29], [0.0, 0.0, 0.0, 0.42, 0.29], [0.37, 0.61, 0.49, 0.0, 0.29], [0.0, 0.61, 0.49, 0.0, 0.29], [0.37, 0.0, 0.49, 0.0, 0.29], [0.0, 0.0, 0.49, 0.0, 0.29], [0.37, 0.61, 0.0, 0.0, 0.29], [0.0, 0.61, 0.0, 0.0, 0.29], [0.37, 0.0, 0.0, 0.0, 0.29], [0.0, 0.0, 0.0, 0.0, 0.29]])\n", 'call': '#case:boundary\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I3'\nimport numpy as np\npoints=np.array([[1e-05, 0.46, 0.27, 0.83, 0.57], [0.14, 1e-05, 0.27, 0.83, 0.57], [0.14, 0.46, 1e-05, 0.83, 0.57], [0.14, 0.46, 0.27, 1e-05, 0.57]])\n", 'call': '#case:edge\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I4'\nimport numpy as np\npoints=np.array([[0.37, 0.61, 0.49, 0.42, 0.29], [0.72, 0.18, 0.81, 0.74, 0.73], [0.14, 0.46, 0.27, 0.5, 0.57]])\n", 'call': '#case:normal\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I4'\nimport numpy as np\npoints=np.array([[0.0, 0.61, 0.49, 0.42, 0.29], [0.37, 0.0, 0.49, 0.42, 0.29], [0.0, 0.0, 0.49, 0.42, 0.29], [0.37, 0.61, 0.0, 0.42, 0.29], [0.0, 0.61, 0.0, 0.42, 0.29], [0.37, 0.0, 0.0, 0.42, 0.29], [0.0, 0.0, 0.0, 0.42, 0.29], [0.37, 0.61, 0.49, 0.0, 0.29], [0.0, 0.61, 0.49, 0.0, 0.29], [0.37, 0.0, 0.49, 0.0, 0.29], [0.0, 0.0, 0.49, 0.0, 0.29], [0.37, 0.61, 0.0, 0.0, 0.29], [0.0, 0.61, 0.0, 0.0, 0.29], [0.37, 0.0, 0.0, 0.0, 0.29], [0.0, 0.0, 0.0, 0.0, 0.29]])\n", 'call': '#case:boundary\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I4'\nimport numpy as np\npoints=np.array([[1e-05, 0.46, 0.27, 0.83, 0.57], [0.14, 1e-05, 0.27, 0.83, 0.57], [0.14, 0.46, 1e-05, 0.83, 0.57], [0.14, 0.46, 0.27, 1e-05, 0.57]])\n", 'call': '#case:edge\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I5'\nimport numpy as np\npoints=np.array([[0.37, 0.61, 0.49, 0.42, 0.29], [0.72, 0.18, 0.81, 0.74, 0.73], [0.14, 0.46, 0.27, 0.5, 0.57]])\n", 'call': '#case:normal\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I5'\nimport numpy as np\npoints=np.array([[0.0, 0.61, 0.49, 0.42, 0.29], [0.37, 0.0, 0.49, 0.42, 0.29], [0.0, 0.0, 0.49, 0.42, 0.29], [0.37, 0.61, 0.0, 0.42, 0.29], [0.0, 0.61, 0.0, 0.42, 0.29], [0.37, 0.0, 0.0, 0.42, 0.29], [0.0, 0.0, 0.0, 0.42, 0.29], [0.37, 0.61, 0.49, 0.0, 0.29], [0.0, 0.61, 0.49, 0.0, 0.29], [0.37, 0.0, 0.49, 0.0, 0.29], [0.0, 0.0, 0.49, 0.0, 0.29], [0.37, 0.61, 0.0, 0.0, 0.29], [0.0, 0.61, 0.0, 0.0, 0.29], [0.37, 0.0, 0.0, 0.0, 0.29], [0.0, 0.0, 0.0, 0.0, 0.29]])\n", 'call': '#case:boundary\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='I5'\nimport numpy as np\npoints=np.array([[1e-05, 0.46, 0.27, 0.83, 0.57], [0.14, 1e-05, 0.27, 0.83, 0.57], [0.14, 0.46, 1e-05, 0.83, 0.57], [0.14, 0.46, 0.27, 1e-05, 0.57]])\n", 'call': '#case:edge\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='II1'\nimport numpy as np\npoints=np.array([[0.37, 0.61, 0.49, 0.42, 0.29], [0.72, 0.18, 0.81, 0.74, 0.73], [0.14, 0.46, 0.27, 0.5, 0.57]])\n", 'call': '#case:normal\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='II1'\nimport numpy as np\npoints=np.array([[0.0, 0.61, 0.49, 0.42, 0.29], [0.37, 0.0, 0.49, 0.42, 0.29], [0.0, 0.0, 0.49, 0.42, 0.29], [0.37, 0.61, 0.0, 0.42, 0.29], [0.0, 0.61, 0.0, 0.42, 0.29], [0.37, 0.0, 0.0, 0.42, 0.29], [0.0, 0.0, 0.0, 0.42, 0.29], [0.37, 0.61, 0.49, 0.0, 0.29], [0.0, 0.61, 0.49, 0.0, 0.29], [0.37, 0.0, 0.49, 0.0, 0.29], [0.0, 0.0, 0.49, 0.0, 0.29], [0.37, 0.61, 0.0, 0.0, 0.29], [0.0, 0.61, 0.0, 0.0, 0.29], [0.37, 0.0, 0.0, 0.0, 0.29], [0.0, 0.0, 0.0, 0.0, 0.29]])\n", 'call': '#case:boundary\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='II1'\nimport numpy as np\npoints=np.array([[1e-05, 0.46, 0.27, 0.83, 0.57], [0.14, 1e-05, 0.27, 0.83, 0.57], [0.14, 0.46, 1e-05, 0.83, 0.57], [0.14, 0.46, 0.27, 1e-05, 0.57]])\n", 'call': '#case:edge\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='II2'\nimport numpy as np\npoints=np.array([[0.37, 0.61, 0.49, 0.42, 0.29], [0.72, 0.18, 0.81, 0.74, 0.73], [0.14, 0.46, 0.27, 0.5, 0.57]])\n", 'call': '#case:normal\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='II2'\nimport numpy as np\npoints=np.array([[0.0, 0.61, 0.49, 0.42, 0.29], [0.37, 0.0, 0.49, 0.42, 0.29], [0.0, 0.0, 0.49, 0.42, 0.29], [0.37, 0.61, 0.0, 0.42, 0.29], [0.0, 0.61, 0.0, 0.42, 0.29], [0.37, 0.0, 0.0, 0.42, 0.29], [0.0, 0.0, 0.0, 0.42, 0.29], [0.37, 0.61, 0.49, 0.0, 0.29], [0.0, 0.61, 0.49, 0.0, 0.29], [0.37, 0.0, 0.49, 0.0, 0.29], [0.0, 0.0, 0.49, 0.0, 0.29], [0.37, 0.61, 0.0, 0.0, 0.29], [0.0, 0.61, 0.0, 0.0, 0.29], [0.37, 0.0, 0.0, 0.0, 0.29], [0.0, 0.0, 0.0, 0.0, 0.29]])\n", 'call': '#case:boundary\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': "sector='II2'\nimport numpy as np\npoints=np.array([[1e-05, 0.46, 0.27, 0.83, 0.57], [0.14, 1e-05, 0.27, 0.83, 0.57], [0.14, 0.46, 1e-05, 0.83, 0.57], [0.14, 0.46, 0.27, 1e-05, 0.57]])\n", 'call': '#case:edge\nassemble_qqbar_sector_jets(sector,points)', 'gold_call': '_oracle_assemble_qqbar_sector_jets(sector,points)'},
     {'setup': 'import numpy as np\npoints=np.array([[0.2, 0.3, 0.4, 0.5, 0.0]])\ndef check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\n', 'call': "#case:edge\ncheck(lambda:assemble_qqbar_sector_jets('I1',points))", 'gold_call': "check(lambda:_oracle_assemble_qqbar_sector_jets('I1',points))"},
     {'setup': 'import numpy as np\npoints=np.array([[0.37, 0.61, 0.49, 0.42, 0.29], [0.72, 0.18, 0.81, 0.74, 0.73], [0.14, 0.46, 0.27, 0.5, 0.57]])\ndef check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\n', 'call': "#case:edge\ncheck(lambda:assemble_qqbar_sector_jets('I6',points))", 'gold_call': "check(lambda:_oracle_assemble_qqbar_sector_jets('I6',points))"},
     {'setup': 'import numpy as np\ndef check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\n', 'call': "#case:edge\ncheck(lambda:assemble_qqbar_sector_jets('I1', np.array([[.3,.4,.5,.6,.7]],complex)+1j*.01))", 'gold_call': "check(lambda:_oracle_assemble_qqbar_sector_jets('I1', np.array([[.3,.4,.5,.6,.7]],complex)+1j*.01))"}]
