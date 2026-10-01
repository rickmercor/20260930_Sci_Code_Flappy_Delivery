#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import itertools
import math
import numpy as np
from scipy.special import gammainc, gammaln
import math
import numpy as np

def _positive_scalar(value, name):
    result = float(_real(value, name, ()))
    if result <= 0:
        raise ValueError(name + ' must be positive')
    return result

def _derivative_specification():
    """Group Cartesian tensor entries by their derivative multi-index."""
    all_orders = []
    for order in range(6):
        grouped = {}
        for flat, index in enumerate(itertools.product(range(3), repeat=order)):
            counts = tuple((index.count(axis) for axis in range(3)))
            grouped.setdefault(counts, []).append(flat)
        entries = []
        for counts, positions in grouped.items():
            terms = []
            for pairs in itertools.product(*(range(n // 2 + 1) for n in counts)):
                powers = tuple((n - 2 * p for n, p in zip(counts, pairs)))
                coefficient = 1
                for n, p, power in zip(counts, pairs, powers):
                    coefficient *= math.factorial(n) // (2 ** p * math.factorial(p) * math.factorial(power))
                terms.append((order - sum(pairs), powers, coefficient))
            entries.append((np.array(positions, dtype=int), tuple(terms)))
        all_orders.append(tuple(entries))
    return tuple(all_orders)
_DERIVATIVE_SPEC = _derivative_specification()

def _gaussian_moments(argument):
    """Return integral_0^1 t**(2*n) exp(-argument*t*t) dt, n=0,...,5."""
    indices = np.arange(6, dtype=float)
    if argument < 0.5:
        values = 1.0 / (2 * indices + 1)
        power_over_factorial = 1.0
        for j in range(1, 80):
            power_over_factorial *= -argument / j
            increment = power_over_factorial / (2 * indices + 2 * j + 1)
            values += increment
            if np.max(np.abs(increment)) <= 2e-17 * np.max(np.abs(values)):
                break
        return values
    exponents = indices + 0.5
    return 0.5 * np.exp(gammaln(exponents) - exponents * math.log(argument)) * gammainc(exponents, argument)

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

def gaussian_jet(d, ui, uj):
    """Return Cartesian derivatives of the screened Coulomb kernel through order five.

    Parameters
    ----------
    d : finite real array (3,)
        Pair displacement R_i-R_j in atomic units.
    ui, uj : finite positive real scalars
        Site hardnesses in atomic units.

    Returns
    -------
    (f, D1, D2, D3, D4, D5) : tuple
        Let u=2*ui*uj/(ui+uj), a=sqrt(pi)*u/2 and r=||d||_2.
        The scalar is f(d)=erf(a*r)/r, continued to f(0)=u.
        Dk[i1,...,ik] is the k-th partial derivative of f with respect
        to d[i1],...,d[ik]. Dk has shape (3,)*k, for k=1,...,5.
        All odd-order tensors vanish at d=0. The even-order tensors
        there follow from the analytic power series
        f(d)=u*sum_{j>=0} (-a*a*dot(d,d))**j/(j!*(2*j+1)).
        Use analytic derivatives, including the continuous coincident
        limit. Finite differences are not the defined calculation.
        Return Cartesian tensors with no factorial or symmetry scaling.
        Inputs are not mutated.

    Raises
    ------
    ValueError
        If d is not a finite real (3,) array, or either hardness is not
        a finite positive real scalar.
    """
    displacement = _real(d, 'd', (3,))
    hardness_i = _positive_scalar(ui, 'ui')
    hardness_j = _positive_scalar(uj, 'uj')
    smaller, larger = sorted((hardness_i, hardness_j))
    hardness = smaller * (2.0 / (1.0 + smaller / larger))
    a2 = math.pi / 4.0 * hardness * hardness
    argument = a2 * float(displacement @ displacement)
    moments = _gaussian_moments(argument)
    radial = hardness * (-2.0 * a2) ** np.arange(6) * moments
    powers = displacement[:, None] ** np.arange(6)[None, :]
    tensors = []
    for order, specification in enumerate(_DERIVATIVE_SPEC):
        entries = np.zeros(3 ** order, dtype=float)
        for positions, terms in specification:
            value = 0.0
            for moment_order, exponents, coefficient in terms:
                value += coefficient * radial[moment_order] * powers[0, exponents[0]] * powers[1, exponents[1]] * powers[2, exponents[2]]
            entries[positions] = value
        tensors.append(float(entries[0]) if order == 0 else entries.reshape((3,) * order))
    return tuple(tensors)

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
            jet = gaussian_jet(positions[i] - positions[j], hardness[i], hardness[j])
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

import itertools
import math
import numpy as np
from scipy.special import gammainc, gammaln
import math
import numpy as np

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

def _scalar(value, name, positive=False, nonnegative=False):
    out = _real(value, name, ())
    x = float(out)
    if positive and x <= 0 or (nonnegative and x < 0):
        raise ValueError(name + ' is outside its domain')
    return x

def _matrix_inputs(G, chi, charge, x, groups):
    chi = _real(chi, 'chi')
    if chi.ndim != 1 or len(chi) < 1:
        raise ValueError('chi must be a nonempty vector')
    n = len(chi)
    G = _real(G, 'G', (9 * n, 9 * n))
    x = _real(x, 'x', (9 * n,))
    charge = _scalar(charge, 'charge')
    if not np.allclose(G, G.T, atol=1e-12, rtol=0):
        raise ValueError('G must be symmetric within absolute tolerance 1e-12')
    groups = _real(groups, 'groups', (n,))
    if np.any(groups < 0) or np.any(groups != np.floor(groups)):
        raise ValueError('groups must contain nonnegative integer-valued labels')
    labels = np.concatenate((groups, np.repeat(groups, 3), np.repeat(groups, 5)))
    mask = labels[:, None] == labels[None, :]
    S = G * mask
    return (G, chi, charge, x, labels, mask, S)

def shadow_response(G, chi, charge, x, groups):
    """Equilibrate a partially linearized multipole energy and return its residual Jacobian.

Parameters
----------
G : finite real symmetric array of shape (9N,9N)
    Symmetry tolerance is absolute 1e-12, relative zero; N>=1.
chi : finite real array (N,)
charge : finite real scalar
x : finite real array (9N,)
    Extended multipoles ordered as q[N], p[3N], theta[5N]; p and theta
    are atom-major. Theta uses the traceless basis of multipole_operator. Its charge
    sum need not equal charge.
groups : length-N array of nonnegative integer-valued labels
    Equal labels identify one fragment; labels need not be consecutive.

Returns
-------
(c, J) : arrays (9N,) and (9N,9N)
    Retain in S every G entry connecting DOFs on atoms in the SAME
    fragment, including all charge, dipole and quadrupole couplings; set all other entries
    of S to zero. L=G-S. For e=[chi,0_(8N)] define
      E(c,x)=e.T c + (1/2)c.T S c + (c-x/2).T L x.
    c is the unique minimizer over w.T c=charge, where w=[ones(N),
    zeros(8N)]. Each retained fragment block of S is positive definite.
    J is d(c(x)-x)/dx at fixed G, chi, charge and groups.
    The constraint applies only to charges, and globally across all
    fragments. Do not impose separate fragment charge constraints.
    Do not include the Lagrange multiplier in either returned array.
    Do not mutate inputs.

Raises
------
ValueError
    If an input is not finite and real; shapes or symmetry violate the
    conditions above; a label is negative/nonintegral; or a retained
    fragment block is not positive definite."""
    G, chi, charge, x, labels, mask, S = _matrix_inputs(G, chi, charge, x, groups)
    n = len(chi)
    e = np.concatenate((chi, np.zeros(8 * n)))
    w = np.concatenate((np.ones(n), np.zeros(8 * n)))
    inv_s = np.zeros_like(G)
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        block = S[np.ix_(ids, ids)]
        try:
            np.linalg.cholesky(block)
            inv_s[np.ix_(ids, ids)] = np.linalg.solve(block, np.eye(len(ids)))
        except np.linalg.LinAlgError as exc:
            raise ValueError('Each retained fragment block must be positive definite') from exc
    v = inv_s @ w
    denom = float(w @ v)
    P = inv_s - np.outer(v, v) / denom
    L = G - S
    c = -P @ (e + L @ x) + v * (charge / denom)
    J = -P @ L - np.eye(9 * n)
    return (c, J)

import itertools
import math
import numpy as np
from scipy.special import gammainc, gammaln
import math
import numpy as np

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

def _scalar(value, name, positive=False, nonnegative=False):
    out = _real(value, name, ())
    x = float(out)
    if positive and x <= 0 or (nonnegative and x < 0):
        raise ValueError(name + ' is outside its domain')
    return x

def _matrix_inputs(G, chi, charge, x, groups):
    chi = _real(chi, 'chi')
    if chi.ndim != 1 or len(chi) < 1:
        raise ValueError('chi must be a nonempty vector')
    n = len(chi)
    G = _real(G, 'G', (9 * n, 9 * n))
    x = _real(x, 'x', (9 * n,))
    charge = _scalar(charge, 'charge')
    if not np.allclose(G, G.T, atol=1e-12, rtol=0):
        raise ValueError('G must be symmetric within absolute tolerance 1e-12')
    groups = _real(groups, 'groups', (n,))
    if np.any(groups < 0) or np.any(groups != np.floor(groups)):
        raise ValueError('groups must contain nonnegative integer-valued labels')
    labels = np.concatenate((groups, np.repeat(groups, 3), np.repeat(groups, 5)))
    mask = labels[:, None] == labels[None, :]
    S = G * mask
    return (G, chi, charge, x, labels, mask, S)

def shadow_state(R, u, alpha, gamma, chi, charge, x, groups):
    """Evaluate the relaxed shadow potential, nuclear forces and electronic response.

Parameters
----------
R is a finite real (N,3) array, N>=1. u, alpha and gamma are finite
positive (N,) arrays of charge hardness, dipole polarizability and
quadrupole polarizability. The 9N ordering is charges, atom-major
Cartesian dipoles, then atom-major five-component quadrupoles in the
orthonormal traceless basis specified by multipole_operator.
All units are atomic units. Boundaries are nonperiodic. Site parameters
do not depend on coordinates. Inputs are never mutated.
chi is finite real (N,), charge is a finite real scalar, x is finite
real (9N,); groups obeys shadow_response's fragment-label contract.

Returns
-------
(energy, forces, c, J) : numerical tuple
    Shapes (), (N,3), (9N,), (9N,9N). Assemble G from the Gaussian
    multipole model; obtain S,L,c and J using shadow_response's
    definition. energy=E(c(x),x) with e=[chi,zeros(8N)]. There is
    no charge-independent potential. forces[i,k] is minus the partial
    derivative of this relaxed energy with respect to R[i,k] at FIXED x.
    Account for coordinate dependence of BOTH S and L. Fragment
    membership remains fixed under differentiation. These are forces
    of the shadow potential, not forces of the fully equilibrated
    regular Born-Oppenheimer potential. Return J for residual c(x)-x.

Raises
------
ValueError
    For any invalid shape, nonfinite/nonreal input, nonpositive u/alpha/gamma,
    invalid group label, or non-positive-definite retained block, as
    specified by multipole_operator and shadow_response."""
    G, dG = multipole_operator(R, u, alpha, gamma)
    G, chi, charge, x, labels, mask, S = _matrix_inputs(G, chi, charge, x, groups)
    c, J = shadow_response(G, chi, charge, x, groups)
    n = len(chi)
    L = G - S
    energy = float(chi @ c[:n] + 0.5 * c @ S @ c + (c - 0.5 * x) @ L @ x)
    dS = dG * mask
    dL = dG - dS
    forces = -0.5 * np.einsum('a,ijab,b->ij', c, dS, c)
    forces -= np.einsum('a,ijab,b->ij', c - 0.5 * x, dL, x)
    return (energy, forces, c, J)

import itertools
import math
import numpy as np
from scipy.special import gammainc, gammaln
import math
import numpy as np

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

def _scalar(value, name, positive=False, nonnegative=False):
    out = _real(value, name, ())
    x = float(out)
    if positive and x <= 0 or (nonnegative and x < 0):
        raise ValueError(name + ' is outside its domain')
    return x

def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if value < minimum:
        raise ValueError(name + ' is too small')
    return int(value)

def krylov_action(J, K0, residual, tol, max_rank):
    """Approximate an inverse-residual-Jacobian action with a preconditioned Krylov space.

Parameters
----------
J, K0 : finite real arrays (D,D), D>=1
    J is the residual Jacobian; K0 is a fixed LEFT preconditioner.
    Both matrices have smallest/largest singular-value ratio >1e-14.
residual : finite real vector (D,)
tol : finite scalar, 0<=tol<1
max_rank : integer (not bool), 1<=max_rank<=D

Returns
-------
(z, rank, errors) : numerical tuple
    Set A=K0@J, b=K0@residual and beta=||b||_2. If beta<=1e-14,
    return a zero vector, rank 0, and an empty float array.
    Otherwise start v_1=b/beta. At rank m, V has the m orthonormal
    Arnoldi columns and W=A@V. Find the minimum-norm least-squares
    minimizer y of ||W*y-b||_2, with relative SVD cutoff 1e-14.
    Set z=V*y and append ||W*y-b||_2/beta to errors. Stop at the first
    error<=tol or at max_rank; otherwise generate the next v from
    A@v_m by TWO passes of modified Gram-Schmidt against ALL existing
    columns in creation order. Stop on breakdown if its remaining norm
    is <=1e-13*max(1,||A@v_m||_2); else normalize it and continue.
    z has shape (D,), rank is the number of used columns, and errors
    has shape (rank,). On a rank cap return the current approximation;
    a full inverse action is a different result and must not replace it.
    Do not mutate inputs. Do not assume J, K0 or A are symmetric.

Raises
------
ValueError
    If inputs are not finite/real, their shapes disagree, a singular-value
    ratio is <=1e-14, tol is outside [0,1), or max_rank is not an integer
    in [1,D]."""
    f = _real(residual, 'residual')
    if f.ndim != 1 or len(f) < 1:
        raise ValueError('residual must be a nonempty vector')
    n = len(f)
    J, K0 = (_real(J, 'J', (n, n)), _real(K0, 'K0', (n, n)))
    tol = _scalar(tol, 'tol', nonnegative=True)
    max_rank = _integer(max_rank, 'max_rank', 1)
    if tol >= 1 or max_rank > n:
        raise ValueError('Require 0<=tol<1 and max_rank<=dimension')
    for A in (J, K0):
        singular = np.linalg.svd(A, compute_uv=False)
        if singular[-1] <= 1e-14 * singular[0]:
            raise ValueError('J and K0 must have singular-value ratio above 1e-14')
    b = K0 @ f
    beta = float(np.linalg.norm(b))
    if beta <= 1e-14:
        return (np.zeros(n), 0, np.zeros(0))
    A = K0 @ J
    V, W, errors = ([], [], [])
    v = b / beta
    for _ in range(max_rank):
        V.append(v.copy())
        w = A @ v
        W.append(w.copy())
        Vm, Wm = (np.column_stack(V), np.column_stack(W))
        y = np.linalg.lstsq(Wm, b, rcond=1e-14)[0]
        z = Vm @ y
        error = float(np.linalg.norm(Wm @ y - b) / beta)
        errors.append(error)
        if error <= tol:
            break
        candidate = w.copy()
        for _pass in range(2):
            for basis in V:
                candidate -= np.dot(basis, candidate) * basis
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-13 * max(1.0, float(np.linalg.norm(w))):
            break
        v = candidate / norm
    return (z, len(V), np.asarray(errors))

import itertools
import math
import numpy as np
from scipy.special import gammainc, gammaln
import math
import numpy as np

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

def _scalar(value, name, positive=False, nonnegative=False):
    out = _real(value, name, ())
    x = float(out)
    if positive and x <= 0 or (nonnegative and x < 0):
        raise ValueError(name + ' is outside its domain')
    return x

def _sites(R, u, alpha, gamma):
    R = _real(R, 'R')
    if R.ndim != 2 or R.shape[1] != 3 or R.shape[0] < 1:
        raise ValueError('R must have shape (N,3), N>=1')
    n = len(R)
    u, alpha = (_real(u, 'u', (n,)), _real(alpha, 'alpha', (n,)))
    gamma = _real(gamma, 'gamma', (n,))
    if np.any(u <= 0) or np.any(alpha <= 0) or np.any(gamma <= 0):
        raise ValueError('u, alpha and gamma must be positive')
    return (R, u, alpha, gamma)

def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if value < minimum:
        raise ValueError(name + ' is too small')
    return int(value)

def _controls(dt, kappa, diss, coeff, tol, max_rank, dimension):
    dt = _scalar(dt, 'dt', positive=True)
    kappa = _scalar(kappa, 'kappa', positive=True)
    if kappa >= 4:
        raise ValueError('kappa must be less than 4')
    diss = _scalar(diss, 'diss', nonnegative=True)
    coeff = _real(coeff, 'coeff')
    if coeff.ndim != 1 or len(coeff) < 2:
        raise ValueError('coeff must have length at least 2')
    if abs(float(np.sum(coeff))) > 1e-12:
        raise ValueError('coeff must sum to zero within 1e-12')
    tol = _scalar(tol, 'tol', nonnegative=True)
    if tol >= 1:
        raise ValueError('tol must be less than 1')
    max_rank = _integer(max_rank, 'max_rank', 1)
    if max_rank > dimension:
        raise ValueError('max_rank exceeds the multipole dimension')
    return (dt, kappa, diss, coeff, tol, max_rank)

def shadow_step(R, velocity, history, K0, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, tol, max_rank):
    """Advance nuclear and extended multipole states by one coupled shadow-MD step.

Parameters
----------
R is a finite real (N,3) array, N>=1. u, alpha and gamma are finite
positive (N,) arrays of charge hardness, dipole polarizability and
quadrupole polarizability. The 9N ordering is charges, atom-major
Cartesian dipoles, then atom-major five-component quadrupoles in the
orthonormal traceless basis specified by multipole_operator.
All units are atomic units. Boundaries are nonperiodic. Site parameters
do not depend on coordinates. Inputs are never mutated.
velocity is finite (N,3); chi and masses are finite (N,),
with masses strictly positive. charge is a finite scalar. groups is a
length-N array of nonnegative integer-valued fragment labels (equal
labels, including nonconsecutive labels, identify the same fragment).
dt>0, 0<kappa<4, diss>=0 are finite scalars. coeff is a finite vector of
length H>=2 whose sum is zero within absolute tolerance 1e-12.
tol is finite, 0<=tol<1; max_rank is an integer in [1,9N].
At each kernel call use the noise, SVD, rank, and breakdown conventions
of krylov_action; do not replace the truncated action by a full solve.
history has shape (H,9N), newest first: [x(t),x(t-dt),...].
K0 is a finite real (9N,9N) matrix satisfying krylov_action's
singular-value condition. The history entries are independent extended
states, not relaxed multipoles; their charge sums need not equal charge.

Returns
-------
(R_new, velocity_new, history_new, energy_new, c_new, rank, errors)
    Evaluate shadow_state at OLD R and history[0] and obtain force,
    c and J. Let z be krylov_action(J,K0,c-history[0],tol,max_rank).
    Perform the nuclear velocity half kick and position drift with
    old forces and masses. At the same old state use
      x_new=2*history[0]-history[1]-kappa*z+diss*(coeff@history).
    Place x_new at the front of history_new and drop the oldest row.
    Evaluate shadow_state at R_new and x_new, then complete the velocity
    half kick using NEW forces. Keep history_new[0]=x_new; never replace
    it with c_new. Return new-state energy and c, but OLD-state kernel
    rank and error history. kappa already equals dt^2*omega^2.
    Shapes: (N,3),(N,3),(H,9N),(),(9N,),(),(rank,).

Raises
------
ValueError
    For invalid shapes, nonfinite/nonreal entries, nonpositive masses,
    u, alpha, gamma or dt, kappa outside (0,4), diss<0, H<2, coefficient sum
    outside absolute 1e-12 of zero, invalid tol/max_rank/groups, or any
    retained block/J/K0 violating the conditions of shadow_state or
    krylov_action."""
    R, u, alpha, gamma = _sites(R, u, alpha, gamma)
    n = len(R)
    velocity = _real(velocity, 'velocity', (n, 3))
    masses = _real(masses, 'masses', (n,))
    chi = _real(chi, 'chi', (n,))
    if np.any(masses <= 0):
        raise ValueError('masses must be positive')
    dt, kappa, diss, coeff, tol, max_rank = _controls(dt, kappa, diss, coeff, tol, max_rank, 9 * n)
    history = _real(history, 'history', (len(coeff), 9 * n))
    K0 = _real(K0, 'K0', (9 * n, 9 * n))
    charge = _scalar(charge, 'charge')
    _, force, c, J = shadow_state(R, u, alpha, gamma, chi, charge, history[0], groups)
    z, rank, errors = krylov_action(J, K0, c - history[0], tol, max_rank)
    half_velocity = velocity + 0.5 * dt * force / masses[:, None]
    new_R = R + dt * half_velocity
    new_x = 2 * history[0] - history[1] - kappa * z + diss * (coeff @ history)
    new_history = np.concatenate((new_x[None, :], history[:-1]), axis=0)
    energy, new_force, new_c, _ = shadow_state(new_R, u, alpha, gamma, chi, charge, new_x, groups)
    new_velocity = half_velocity + 0.5 * dt * new_force / masses[:, None]
    return (new_R, new_velocity, new_history, energy, new_c, rank, errors)

import itertools
import math
import numpy as np
from scipy.special import gammainc, gammaln
import math
import numpy as np

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

def _scalar(value, name, positive=False, nonnegative=False):
    out = _real(value, name, ())
    x = float(out)
    if positive and x <= 0 or (nonnegative and x < 0):
        raise ValueError(name + ' is outside its domain')
    return x

def _sites(R, u, alpha, gamma):
    R = _real(R, 'R')
    if R.ndim != 2 or R.shape[1] != 3 or R.shape[0] < 1:
        raise ValueError('R must have shape (N,3), N>=1')
    n = len(R)
    u, alpha = (_real(u, 'u', (n,)), _real(alpha, 'alpha', (n,)))
    gamma = _real(gamma, 'gamma', (n,))
    if np.any(u <= 0) or np.any(alpha <= 0) or np.any(gamma <= 0):
        raise ValueError('u, alpha and gamma must be positive')
    return (R, u, alpha, gamma)

def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if value < minimum:
        raise ValueError(name + ' is too small')
    return int(value)

def _controls(dt, kappa, diss, coeff, tol, max_rank, dimension):
    dt = _scalar(dt, 'dt', positive=True)
    kappa = _scalar(kappa, 'kappa', positive=True)
    if kappa >= 4:
        raise ValueError('kappa must be less than 4')
    diss = _scalar(diss, 'diss', nonnegative=True)
    coeff = _real(coeff, 'coeff')
    if coeff.ndim != 1 or len(coeff) < 2:
        raise ValueError('coeff must have length at least 2')
    if abs(float(np.sum(coeff))) > 1e-12:
        raise ValueError('coeff must sum to zero within 1e-12')
    tol = _scalar(tol, 'tol', nonnegative=True)
    if tol >= 1:
        raise ValueError('tol must be less than 1')
    max_rank = _integer(max_rank, 'max_rank', 1)
    if max_rank > dimension:
        raise ValueError('max_rank exceeds the multipole dimension')
    return (dt, kappa, diss, coeff, tol, max_rank)

def shadow_trajectory(R, velocity, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, steps, tol, max_rank):
    """Initialize and propagate a finite deterministic block-retaining shadow trajectory.

Parameters
----------
R is a finite real (N,3) array, N>=1. u, alpha and gamma are finite
positive (N,) arrays of charge hardness, dipole polarizability and
quadrupole polarizability. The 9N ordering is charges, atom-major
Cartesian dipoles, then atom-major five-component quadrupoles in the
orthonormal traceless basis specified by multipole_operator.
All units are atomic units. Boundaries are nonperiodic. Site parameters
do not depend on coordinates. Inputs are never mutated.
velocity is finite (N,3); chi and masses are finite (N,),
with masses strictly positive. charge is a finite scalar. groups is a
length-N array of nonnegative integer-valued fragment labels (equal
labels, including nonconsecutive labels, identify the same fragment).
dt>0, 0<kappa<4, diss>=0 are finite scalars. coeff is a finite vector of
length H>=2 whose sum is zero within absolute tolerance 1e-12.
tol is finite, 0<=tol<1; max_rank is an integer in [1,9N].
At each kernel call use the noise, SVD, rank, and breakdown conventions
of krylov_action; do not replace the truncated action by a full solve.
steps is an integer (not bool) >=0. Initial G must be positive definite.

Returns
-------
record : float array (steps+1,12)
    At initial R, minimize the REGULAR energy e.T c+0.5*c.T G*c
    with global charge constraint. Fill ALL H history rows with this
    initial c0. Calculate J0 from shadow_response at this state and
    retain K0=J0^(-1) for the WHOLE trajectory; never refresh it.
    Call shadow_step exactly steps times. Record the initial state and
    each completed step, using the RELAXED c at that state's R and x.
    Columns: [time, shadow_energy, nuclear_kinetic_energy,
              total_energy, ||c-x||_2, sum(q), D_x, D_y, D_z,
              preceding_step_kernel_rank, preceding_step_final_error,
              ||theta||_2].
    D=sum_i(q_i*R_i+p_i) is the total physical dipole about the fixed
    coordinate origin; intrinsic quadrupoles have zero charge and dipole
    moments and do not enter D directly. Kinetic energy is sum_i masses_i*|velocity_i|^2/2.
    At row 0, rank and final_error are 0; a rank-zero step also has
    final_error 0. Extended DOFs have no kinetic contribution in this
    mass-zero model. No equilibration iterations or thermostat are added.
    steps=0 returns only the initialized row. Do not mutate inputs.

Raises
------
ValueError
    For any invalid dynamics parameter described by shadow_step; if steps
    is not an integer >=0; if initial G is not positive definite/J0 is
    singular; or if a later retained block/J/K0 violates the stated
    positive-definiteness or singular-value conditions."""
    R, u, alpha, gamma = _sites(R, u, alpha, gamma)
    n = len(R)
    R = R.copy()
    velocity = _real(velocity, 'velocity', (n, 3)).copy()
    chi, masses = (_real(chi, 'chi', (n,)), _real(masses, 'masses', (n,)))
    if np.any(masses <= 0):
        raise ValueError('masses must be positive')
    charge = _scalar(charge, 'charge')
    dt, kappa, diss, coeff, tol, max_rank = _controls(dt, kappa, diss, coeff, tol, max_rank, 9 * n)
    steps = _integer(steps, 'steps', 0)
    G, _ = multipole_operator(R, u, alpha, gamma)
    w = np.concatenate((np.ones(n), np.zeros(8 * n)))
    b = np.concatenate((-chi, np.zeros(8 * n), [charge]))
    augmented = np.zeros((9 * n + 1, 9 * n + 1))
    augmented[:-1, :-1] = G
    augmented[:-1, -1] = augmented[-1, :-1] = w
    try:
        np.linalg.cholesky(G)
        initial_c = np.linalg.solve(augmented, b)[:-1]
        _, J0 = shadow_response(G, chi, charge, initial_c, groups)
        K0 = np.linalg.solve(J0, np.eye(9 * n))
    except np.linalg.LinAlgError as exc:
        raise ValueError('Initial G must be positive definite and J0 nonsingular') from exc
    history = np.repeat(initial_c[None, :], len(coeff), axis=0)
    record = np.empty((steps + 1, 12))
    rank, errors = (0, np.zeros(0))
    for k in range(steps + 1):
        energy, _, c, _ = shadow_state(R, u, alpha, gamma, chi, charge, history[0], groups)
        kinetic = float(0.5 * np.sum(masses[:, None] * velocity ** 2))
        dipole = c[:n] @ R + c[n:4 * n].reshape(n, 3).sum(axis=0)
        record[k] = np.r_[k * dt, energy, kinetic, energy + kinetic, np.linalg.norm(c - history[0]), np.sum(c[:n]), dipole, rank, errors[-1] if len(errors) else 0.0, np.linalg.norm(c[4 * n:])]
        if k < steps:
            R, velocity, history, _, _, rank, errors = shadow_step(R, velocity, history, K0, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, tol, max_rank)
    return record

import itertools
import math
import numpy as np
from scipy.special import gammainc, gammaln
import math
import numpy as np

def solve(R, velocity, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, steps, tol, max_rank):
    """Return the finite-window, time-averaged initial-dipole correlation.

Parameters
----------
Same arguments and validation domain as shadow_trajectory, including
groups, steps>=0, finite dt>0 and fixed K0 initialization.
R is a finite real (N,3) array, N>=1. u, alpha and gamma are finite
positive (N,) arrays of charge hardness, dipole polarizability and
quadrupole polarizability. The 9N ordering is charges, atom-major
Cartesian dipoles, then atom-major five-component quadrupoles in the
orthonormal traceless basis specified by multipole_operator.
All units are atomic units. Boundaries are nonperiodic. Site parameters
do not depend on coordinates. Inputs are never mutated.
velocity is finite (N,3); chi and masses are finite (N,),
with masses strictly positive. charge is a finite scalar. groups is a
length-N array of nonnegative integer-valued fragment labels (equal
labels, including nonconsecutive labels, identify the same fragment).
dt>0, 0<kappa<4, diss>=0 are finite scalars. coeff is a finite vector of
length H>=2 whose sum is zero within absolute tolerance 1e-12.
tol is finite, 0<=tol<1; max_rank is an integer in [1,9N].
At each kernel call use the noise, SVD, rank, and breakdown conventions
of krylov_action; do not replace the truncated action by a full solve.

Returns
-------
answer : native Python float
    Obtain shadow_trajectory and take C_k=D(k*dt).dot(D(0)) from its
    RELAXED physical dipoles (columns 6:9). For steps M>0 return the
    composite trapezoidal integral of C(t), divided by M*dt:
      (C_0/2 + sum(C_1,...,C_(M-1)) + C_M/2)/M.
    For M=0 return C_0, the zero-window limit. Do not normalize by
    |D(0)|^2, discard the initial sample, or use extended x as c.
    Units are squared atomic dipole units. No input is mutated.

Raises
------
ValueError
    If steps is not an integer >=0; any array has an invalid shape or
    nonfinite/nonreal entry; a group label is negative/nonintegral;
    masses/u/alpha/gamma/dt are nonpositive; kappa is outside (0,4); diss<0;
    coeff has length <2 or sum outside absolute 1e-12 of zero; tol is
    outside [0,1); max_rank is not an integer in [1,9N]; initial G is
    not positive definite or J0 is singular; a retained fragment block
    is not positive definite; or a J/K0 passed to krylov_action has
    smallest/largest singular-value ratio <=1e-14."""
    record = shadow_trajectory(R, velocity, u, alpha, gamma, chi, masses, charge, groups, dt, kappa, diss, coeff, steps, tol, max_rank)
    correlation = record[:, 6:9] @ record[0, 6:9]
    if len(correlation) == 1:
        return float(correlation[0])
    return float((0.5 * correlation[0] + np.sum(correlation[1:-1]) + 0.5 * correlation[-1]) / (len(correlation) - 1))
SCICODE_GOLD_EOF
