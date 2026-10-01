"""
Charge, dipole and traceless-quadrupole interaction blocks and nuclear derivatives.

The factorial convention for a traceless second moment fixes charge–quadrupole, dipole–quadrupole and quadrupole–quadrupole couplings. The derivative of each pair block enters the two nuclear forces with opposite signs.

Returns
-------
arrays (9N,9N) and (N,3,9N,9N)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def multipole_operator(R, u, alpha, gamma):
    """Assemble Gaussian charge, dipole and traceless-quadrupole interactions.

    Parameters
    ----------
    R : finite real (N,3) array, N>=1
    u, alpha, gamma : finite positive real (N,) arrays
        Hardness, dipole polarizability and quadrupole polarizability.
        Site parameters are independent of coordinates. All quantities
        use atomic units; boundaries are nonperiodic.

    Returns
    -------
    (G, dG) : arrays (9N,9N) and (N,3,9N,9N)
        Multipoles are ordered as all N charges, then 3N atomic dipole
        components in atomic xyz order, then 5N atomic quadrupole
        coefficients in the basis order below. For each atom,
        Q=sum_A theta_A*B_A is a traceless second-moment tensor, where
        B0=diag(1,-1,0)/sqrt(2), B1=diag(1,1,-2)/sqrt(6),
        B2=(ex*ey.T+ey*ex.T)/sqrt(2),
        B3=(ex*ez.T+ez*ex.T)/sqrt(2),
        B4=(ey*ez.T+ez*ey.T)/sqrt(2).
        The basis is orthonormal under the Frobenius inner product.
        The density convention is q-p_a*partial_a+(1/2)*Q_ab*partial_ab.

        For distinct sites i,j, use gaussian_jet at d=R_i-R_j.
        Assign rank 0 to charge, rank 1 to each dipole channel, and
        rank 2 to each quadrupole channel. Channel tensors M are 1,
        the three Cartesian unit vectors, and the five B tensors.
        The pair entry for channel A at i and channel B at j is
        (-1)**rank_B/(rank_A!*rank_B!) times the contraction of
        M_A, M_B, and the Cartesian derivative of f of order
        rank_A+rank_B, using M_A indices first and M_B indices second.
        Onsite blocks are u_i, I3/alpha_i and I5/gamma_i; onsite
        cross terms vanish. G is symmetric in its global indices.
        dG[i,k,a,b]=partial G[a,b]/partial R[i,k]. A derivative on
        the first site of a pair adds a final derivative index k;
        a derivative on its second site changes the sign. Onsite
        derivatives vanish. Coincident distinct sites use the
        analytic continuous limit. Inputs are not mutated.

    Raises
    ------
    ValueError
        If R is not finite real (N,3) with N>=1; a site-parameter
        array is not finite real (N,); or a site parameter is nonpositive.
    """
    return (np.zeros((0, 0)), np.zeros((0, 3, 0, 0)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import math
import numpy as np
from scipy.special import gammainc, gammaln
import math
import numpy as np

def _quadrupole_basis():
    basis = np.zeros((5, 3, 3), dtype=float)
    basis[0] = np.diag([1.0, -1.0, 0.0]) / math.sqrt(2)
    basis[1] = np.diag([1.0, 1.0, -2.0]) / math.sqrt(6)
    for index, (a, b) in enumerate(((0, 1), (0, 2), (1, 2)), start=2):
        basis[index, a, b] = basis[index, b, a] = 1.0 / math.sqrt(2)
    return basis
_QUADRUPOLE_BASIS = _quadrupole_basis()

def _pair_blocks(jet):
    """Return the 9-by-9 interaction and its three displacement derivatives."""
    f, gradient, hessian, third, fourth, fifth = jet
    basis = _QUADRUPOLE_BASIS
    value = np.empty((9, 9), dtype=float)
    deriv = np.empty((9, 9, 3), dtype=float)
    value[0, 0] = f
    value[1:4, 0] = gradient
    value[0, 1:4] = -gradient
    value[1:4, 1:4] = -hessian
    value[4:9, 0] = value[0, 4:9] = 0.5 * np.einsum('Aab,ab->A', basis, hessian)
    qd = 0.5 * np.einsum('Aab,abk->Ak', basis, third)
    value[4:9, 1:4] = -qd
    value[1:4, 4:9] = qd.T
    value[4:9, 4:9] = 0.25 * np.einsum('Aab,Bcd,abcd->AB', basis, basis, fourth)
    deriv[0, 0] = gradient
    deriv[1:4, 0] = hessian
    deriv[0, 1:4] = -hessian
    deriv[1:4, 1:4] = -third
    q0 = 0.5 * np.einsum('Aab,abk->Ak', basis, third)
    deriv[4:9, 0] = deriv[0, 4:9] = q0
    qd_deriv = 0.5 * np.einsum('Aab,abck->Ack', basis, fourth)
    deriv[4:9, 1:4] = -qd_deriv
    deriv[1:4, 4:9] = np.swapaxes(qd_deriv, 0, 1)
    deriv[4:9, 4:9] = 0.25 * np.einsum('Aab,Bcd,abcdk->ABk', basis, basis, fifth)
    return (value, deriv)

def _real(value, name, shape=None):
    try:
        if np.iscomplexobj(value):
            raise ValueError(name + ' must be real')
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be finite and real') from exc
    if not np.all(np.isfinite(out)) or (shape is not None and out.shape != shape):
        raise ValueError(name + ' has an invalid shape or nonfinite entry')
    return out

def _oracle_multipole_operator(R, u, alpha, gamma):
    """Assemble Gaussian charge, dipole and traceless-quadrupole interactions.

    Parameters
    ----------
    R : finite real (N,3) array, N>=1
    u, alpha, gamma : finite positive real (N,) arrays
        Hardness, dipole polarizability and quadrupole polarizability.
        Site parameters are independent of coordinates. All quantities
        use atomic units; boundaries are nonperiodic.

    Returns
    -------
    (G, dG) : arrays (9N,9N) and (N,3,9N,9N)
        Multipoles are ordered as all N charges, then 3N atomic dipole
        components in atomic xyz order, then 5N atomic quadrupole
        coefficients in the basis order below. For each atom,
        Q=sum_A theta_A*B_A is a traceless second-moment tensor, where
        B0=diag(1,-1,0)/sqrt(2), B1=diag(1,1,-2)/sqrt(6),
        B2=(ex*ey.T+ey*ex.T)/sqrt(2),
        B3=(ex*ez.T+ez*ex.T)/sqrt(2),
        B4=(ey*ez.T+ez*ey.T)/sqrt(2).
        The basis is orthonormal under the Frobenius inner product.
        The density convention is q-p_a*partial_a+(1/2)*Q_ab*partial_ab.

        For distinct sites i,j, use gaussian_jet at d=R_i-R_j.
        Assign rank 0 to charge, rank 1 to each dipole channel, and
        rank 2 to each quadrupole channel. Channel tensors M are 1,
        the three Cartesian unit vectors, and the five B tensors.
        The pair entry for channel A at i and channel B at j is
        (-1)**rank_B/(rank_A!*rank_B!) times the contraction of
        M_A, M_B, and the Cartesian derivative of f of order
        rank_A+rank_B, using M_A indices first and M_B indices second.
        Onsite blocks are u_i, I3/alpha_i and I5/gamma_i; onsite
        cross terms vanish. G is symmetric in its global indices.
        dG[i,k,a,b]=partial G[a,b]/partial R[i,k]. A derivative on
        the first site of a pair adds a final derivative index k;
        a derivative on its second site changes the sign. Onsite
        derivatives vanish. Coincident distinct sites use the
        analytic continuous limit. Inputs are not mutated.

    Raises
    ------
    ValueError
        If R is not finite real (N,3) with N>=1; a site-parameter
        array is not finite real (N,); or a site parameter is nonpositive.
    """
    positions = _real(R, 'R')
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 1:
        raise ValueError('R must have shape (N,3), N>=1')
    count = len(positions)
    hardness = _real(u, 'u', (count,))
    dipole_polarizability = _real(alpha, 'alpha', (count,))
    quadrupole_polarizability = _real(gamma, 'gamma', (count,))
    if any((np.any(parameter <= 0) for parameter in (hardness, dipole_polarizability, quadrupole_polarizability))):
        raise ValueError('Site parameters must be positive')
    operator = np.diag(np.concatenate((hardness, np.repeat(1.0 / dipole_polarizability, 3), np.repeat(1.0 / quadrupole_polarizability, 5))))
    derivative = np.zeros((count, 3, 9 * count, 9 * count), dtype=float)
    atom_indices = [np.concatenate(([i], count + 3 * i + np.arange(3), 4 * count + 5 * i + np.arange(5))).astype(int) for i in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            jet = _oracle_gaussian_jet(positions[i] - positions[j], hardness[i], hardness[j])
            pair, dpair = _pair_blocks(jet)
            forward = np.ix_(atom_indices[i], atom_indices[j])
            backward = np.ix_(atom_indices[j], atom_indices[i])
            operator[forward] = pair
            operator[backward] = pair.T
            for component in range(3):
                block = dpair[:, :, component]
                derivative[i, component][forward] = block
                derivative[i, component][backward] = block.T
                derivative[j, component][forward] = -block
                derivative[j, component][backward] = -block.T
    return (operator, derivative)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'multipole_operator(R,u,alpha,gamma)',
      'gold_call': '_oracle_multipole_operator(R,u,alpha,gamma)'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[.1,.2,-.3]])\n'
               'u=np.array([1.2])\n'
               'alpha=np.array([.25])\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'multipole_operator(R,u,alpha,gamma)',
      'gold_call': '_oracle_multipole_operator(R,u,alpha,gamma)'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'R[1]=R[0]+[1e-9,-2e-9,3e-9]\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'multipole_operator(R,u,alpha,gamma)',
      'gold_call': '_oracle_multipole_operator(R,u,alpha,gamma)'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'R[1]=R[0]\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'multipole_operator(R,u,alpha,gamma)',
      'gold_call': '_oracle_multipole_operator(R,u,alpha,gamma)'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'alpha[1]=-1\n'
               '\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        multipole_operator(R,u,alpha,gamma)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_multipole_operator(R,u,alpha,gamma)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'R=np.array([[0.,0.,0.],[.8,-.3,.6],[-.7,1.1,.4]])\n'
               'u=np.array([1.7,.9,1.3])\n'
               'alpha=np.array([.2,.3,.25])\n'
               'alpha[1]=.3\n'
               '\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        multipole_operator(R,u,alpha,gamma)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_multipole_operator(R,u,alpha,gamma)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'gamma=np.full(len(u),0.07)\n'
               '\n'
               'gamma[1]=0.0\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
