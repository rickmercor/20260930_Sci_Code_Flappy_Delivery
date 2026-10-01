# Chemistry-Computational_Chemistry-13

## Background

This is a reduced vibronic-spectroscopy model. The full polynomial potential drives the classical centre. A fixed reference Hessian governs the Gaussian width. This separation follows single-Hessian Gaussian wave packet dynamics and permits exact width propagation alongside a geometric centre integrator. The cubic transition dipole prepares non-Gaussian nuclear states. They are finite superpositions of Hagedorn functions whose coefficients remain constant under the chosen time-dependent quadratic Hamiltonian. Their autocorrelation requires overlaps between different moving bases.  The orientational average is one third of the trace of the Cartesian dipole-correlation tensor. Higher polynomial terms are explicit benchmark assumptions; they are not called the linear Herzberg–Teller approximation. The matrices, polynomial coefficients, time grid and spectral ordinate are synthetic benchmark data. They are not experimental parameters or reported results from these papers. The generating-function calculation below is derived for this benchmark from normalized Hagedorn functions; it is equivalent to cross-basis overlap recurrences, not a claimed equation from the main paper

## Problem

Compute the normalized reduced absorption ordinate at $$\omega=2.2$$ for the three-coordinate molecular model below, at zero temperature and with $$\hbar=0.7$$.
The initial nuclear state is the ground state of $$H_g=\tfrac12p^TM^{-1}p+\tfrac12(q-q_i)^TK_g(q-q_i)$$; the excited-state potential and vector transition dipole are the monomial polynomials in the tables, with no factorial factors in their coefficients.
Approximate excited-state dynamics by one moving quadratic potential that retains the full potential value and gradient at its classical centre and uses the Hessian evaluated once at $$q_r$$; the initial centre is $$q_i$$ with zero momentum and zero action.
Propagate the three dipole-prepared nuclear states in the Hagedorn basis guided by that same moving Gaussian, retaining the complete cubic polynomial; the polynomial components do not change the guiding trajectory.
Use the initial frame gauge $$Q_0=A_g^{-1/2},\ P_0=iA_g^{1/2}$$, where $$A_g$$ is the SPD solution of $$A_gM^{-1}A_g=K_g$$, and continue the phase of the Gaussian normalization from time zero.
For the centre and action, use the fourth-order composition of exact potential-half/kinetic/potential-half flows with substep sizes $$(\gamma\Delta t,(1-2\gamma)\Delta t,\gamma\Delta t)$$ in that order, where $$\gamma=(2-2^{1/3})^{-1}$$; propagate the width exactly with the fixed Hessian.
For the isotropic ensemble, let $$C(t)$$ be the orientationally averaged autocorrelation of the unnormalized dipole-prepared states, and evaluate the finite-grid quantity

$$
J=\frac{\Delta t}{3\pi C(0)}\operatorname{Re}
\sum_{j=0}^{300}w_j C(t_j)
\exp\!\left[i\left(\omega+\frac{E_{g,0}}{\hbar}\right)t_j-\eta t_j^2\right],
\qquad t_j=j\Delta t.
$$

Use $$\Delta t=0.04$$, $$\eta=0.045$$, endpoint weights $$w_0=w_{300}=1$$, odd interior weights 4, and even interior weights 2; this finite-grid expression defines the requested answer, with no frequency prefactor or tail correction.
Document the reference Hessian, initial frame and zero-point energy, dipole coefficients $$c_{000}$$ and $$c_{100}$$, initial norm, terminal centre/action/determinant phase, the overlaps $$\langle\phi_{000}(0)|\phi_{000}(3)\rangle$$ and $$\langle\phi_{002}(0)|\phi_{101}(3)\rangle$$, and $$C(t)$$ at $$t=1,3,6,12$$; return $$J$$ to six decimal places.

## Model data

All values use consistent reduced units. The mass matrix is full.

$$
M=\begin{pmatrix}1.2&0.12&-0.08\\0.12&1.5&0.10\\-0.08&0.10&0.9\end{pmatrix},\qquad
K_g=\begin{pmatrix}1.4&0.23&-0.14\\0.23&2.1&0.17\\-0.14&0.17&0.95\end{pmatrix}.
$$

$$
q_i=(-0.35,0.22,0.18)^T,\quad q_e=(0.10,-0.15,0.08)^T,\quad
q_r=(0.06,-0.04,0.12)^T.
$$

The potential uses $$x=q-q_e$$. A row $$(a,b,c),v$$ contributes $$v x_1^a x_2^b x_3^c$$.

| Powers of x | Coefficient |
| --- | --- |
| (0, 0, 0) | 1.6 |
| (2, 0, 0) | 0.5 |
| (0, 2, 0) | 0.925 |
| (0, 0, 2) | 0.65 |
| (1, 1, 0) | 0.21 |
| (1, 0, 1) | -0.16 |
| (0, 1, 1) | 0.27 |
| (3, 0, 0) | 0.06 |
| (0, 3, 0) | -0.04 |
| (0, 0, 3) | 0.03 |
| (2, 1, 0) | 0.08 |
| (1, 1, 1) | -0.055 |
| (4, 0, 0) | 0.018 |
| (0, 4, 0) | 0.024 |
| (0, 0, 4) | 0.02 |
| (2, 2, 0) | 0.012 |
| (0, 2, 2) | 0.016 |
| (2, 0, 2) | 0.01 |

The transition dipole uses $$y=q-q_i$$. Each row contributes the stated vector times $$y_1^a y_2^b y_3^c$$.

| Powers of y | mu_x | mu_y | mu_z |
| --- | --- | --- | --- |
| (0, 0, 0) | 0.32 | -0.18 | 0.11 |
| (1, 0, 0) | 0.75 | 0.12 | -0.24 |
| (0, 1, 0) | -0.28 | 0.6 | 0.16 |
| (0, 0, 1) | 0.18 | -0.32 | 0.55 |
| (2, 0, 0) | 0.2 | -0.1 | 0.08 |
| (1, 1, 0) | -0.17 | 0.22 | -0.12 |
| (0, 1, 1) | 0.09 | -0.16 | 0.13 |
| (0, 0, 2) | -0.11 | 0.07 | 0.2 |
| (3, 0, 0) | 0.065 | -0.04 | 0.015 |
| (1, 1, 1) | -0.045 | 0.035 | 0.055 |
| (0, 2, 1) | 0.03 | -0.05 | 0.025 |

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_potential_jet

Goal
----
Evaluate the supplied polynomial potential, its gradient, and its Hessian at one geometry.

```python
def potential_jet(q, origin, powers, coefficients):
    r"""Evaluate V(q)=sum_r coefficients[r]*product_j (q[j]-origin[j])**powers[r,j].
    Coefficients are ordinary monomial coefficients, with no factorial convention.
    Differentiate analytically; coordinates equal to the expansion origin are valid.

    Parameters
    ----------
    q, origin : real arrays, shape (D,), D >= 1
    powers : numeric integer-valued array, shape (R,D), entries >= 0
    coefficients : real array, shape (R,)
        R=0 is allowed and means the zero potential. Repeated powers add.

    Returns
    -------
    value : float
    gradient : real array, shape (D,)
    hessian : real array, shape (D,D)

    Raises
    ------
    ValueError
        For incompatible shapes, empty q, nonfinite data, or powers that are
        negative or not integer-valued.
    """
    return value, gradient, hessian
```

### Step 2

02_initial_frame

Goal
----
Construct a fixed gauge for the ground-state Gaussian with a full mass matrix.

```python
def initial_frame(mass, ground_hessian, hbar):
    r"""Use the ground vibrational state of H_g=p.T M^{-1} p/2 + y.T K_g y/2.
    Let A be the symmetric positive-definite solution of A M^{-1} A=K_g.
    Fix the frame gauge Q0=A^{-1/2}, P0=i A^{1/2}; use principal SPD matrix powers.
    The initial Gaussian is (pi*hbar)^(-D/4)*det(Q0)^(-1/2)
    times exp(-y.T A y/(2*hbar)). Return its zero-point energy E0.

    Parameters
    ----------
    mass, ground_hessian : real symmetric positive-definite arrays, shape (D,D)
    hbar : finite positive float

    Returns
    -------
    Q0 : real SPD array, shape (D,D)
    P0 : complex array, shape (D,D)
    E0 : float
        hbar/2 times the sum of the positive generalized normal-mode frequencies.

    Raises
    ------
    ValueError
        For non-square, empty, unequal or nonfinite matrices; lack of symmetry
        within absolute tolerance 1e-12; non-positive-definite matrices; or hbar<=0
        or nonfinite hbar.
    """
    return Q0, P0, E0
```

### Step 3

03_centre_action

Goal
----
Propagate the centre and classical action with a specified fourth-order composition.

```python
def centre_action(mass, q0, p0, origin, powers, coefficients, dt, nsteps):
    r"""Propagate dq/dt=M^{-1}p, dp/dt=-grad V(q), dS/dt=p.T M^{-1}p/2-V(q).
    Start S=0. V is the monomial polynomial defined in potential_jet.
    One VTV(h) step applies the exact potential flow for h/2, the exact kinetic
    flow for h, and the potential flow for h/2, updating S in every flow.
    One requested step is VTV(gamma*dt), VTV((1-2*gamma)*dt), VTV(gamma*dt),
    in that chronological order, where gamma=1/(2-2**(1/3)).
    Return initial values and values after each full requested step.
    Earlier dependency: potential_jet(q, origin, powers, coefficients).

    Parameters
    ----------
    mass : real SPD (D,D) array
    q0, p0, origin : real (D,) arrays
    powers, coefficients : polynomial arrays with the contract of potential_jet
    dt : finite nonzero float; negative dt is allowed
    nsteps : nonnegative integer

    Returns
    -------
    positions, momenta : real arrays, shape (nsteps+1,D)
    actions : real array, shape (nsteps+1,)

    Raises
    ------
    ValueError
        For invalid polynomial data as defined in potential_jet; invalid shapes,
        empty or nonfinite state; mass not symmetric within 1e-12 or not SPD;
        zero or nonfinite dt; or nsteps not a nonnegative integer.
        The supported data have finite trajectories over the requested interval.
    """
    return positions, momenta, actions
```

### Step 4

04_width_path

Goal
----
Evolve the width with the constant reference Hessian and retain the determinant phase.

```python
def width_path(mass, reference_hessian, q0, p0, times):
    r"""Solve dQ/dt=M^{-1}P and dP/dt=-K_ref Q exactly at the supplied times.
    K_ref is constant; it may be singular or indefinite. Q0 is real SPD and P0
    is complex. Both initial canonical identities must hold:
    Q0.T P0-P0.T Q0=0 and Q0.conj().T P0-P0.conj().T Q0=2i I.
    Return logdetQ=log(abs(det(Q)))+i*theta. Set theta[0]=0 and choose at each
    successive sample the 2*pi lift of arg(det(Q)) nearest the preceding theta.
    All supported grids resolve the true phase: each true increment has magnitude
    less than pi, with no ties. Do not reset to the principal determinant phase.

    Parameters
    ----------
    mass : real SPD (D,D) array
    reference_hessian : real symmetric (D,D) array
    q0 : real SPD (D,D) initial Q matrix
    p0 : complex (D,D) initial P matrix
    times : finite (N,) array, N>=1, starting at zero
        If N>1, entries are strictly increasing or strictly decreasing.

    Returns
    -------
    Q_path, P_path : complex arrays, shape (N,D,D)
    logdetQ : complex array, shape (N,)

    Raises
    ------
    ValueError
        For invalid or nonfinite shapes; mass not SPD; symmetry errors above
        1e-12 in mass, reference_hessian or q0; q0 imaginary part above 1e-12
        or q0 not SPD; initial canonical errors above 1e-9; or invalid times.
    """
    return Q_path, P_path, logdetQ
```

### Step 5

05_dipole_coefficients

Goal
----
Expand all three Cartesian transition-dipole components in the initial Hagedorn basis.

```python
def dipole_coefficients(q0, powers, coefficients, hbar):
    r"""For the real SPD initial Q0 and P0=i*inv(Q0), expand
    mu_a(y)*phi_0 = sum_k c[k,a]*phi_k for a=x,y,z, y=q-q_initial.
    The basis is orthonormal, with coordinate operator
    y_i=sqrt(hbar/2)*sum_j Q0[i,j]*(a_j + a_j^dagger),
    a_j|k>=sqrt(k_j)|k-e_j>, a_j^dagger|k>=sqrt(k_j+1)|k+e_j>.
    mu_a(y)=sum_r coefficients[r,a]*product_i y_i**powers[r,i].
    There are no hidden factorials. Repeated terms add. Include every occupation
    label with total degree <= max_r sum_i powers[r,i], including zero coefficients.
    Sort labels first by total degree, then by the tuple in ascending lexicographic
    order. This order is (0,0), (0,1), (1,0), (0,2), (1,1), (2,0) for D=2, degree=2.

    Parameters
    ----------
    q0 : real SPD array, shape (D,D)
    powers : nonnegative integer-valued array, shape (R,D), R>=1, degree<=6
    coefficients : finite real or complex array, shape (R,3)
    hbar : finite positive float

    Returns
    -------
    labels : integer array, shape (L,D)
    c : complex array, shape (L,3), unnormalized

    Raises
    ------
    ValueError
        For invalid or nonfinite shapes; q0 not SPD or symmetry error above 1e-12;
        negative, noninteger or degree>6 powers; or nonfinite/nonpositive hbar.
    """
    return labels, c
```

### Step 6

06_overlap_generator

Goal
----
Construct the Gaussian generating function for overlaps of two Hagedorn bases.

```python
def overlap_generator(q_bra, p_bra, Q_bra, P_bra, S_bra, logdet_bra,
                              q_ket, p_ket, Q_ket, P_ket, S_ket, logdet_ket, hbar):
    r"""Return B, b, G such that
    F(u,v)=sum_{k,l} <phi_bra,k|phi_ket,l>*u**k*v**l/sqrt(k!*l!)
          =G*exp(z.T B z/2+b.T z), z=(u,v), with bra variables first.
    Variables u,v are formal and are not conjugated. B is complex symmetric.
    This definition fully fixes the signs, ordering and factorial convention.

    For either frame define y=x-q, A=P@inv(Q), and
    phi_0(x)=(pi*hbar)^(-D/4)*exp(-logdetQ/2)
             *exp(i*(y.T A y/2+p.T y+S)/hbar).
    Its basis generating function is
    sum_k phi_k(x)*v**k/sqrt(k!)
     =phi_0(x)*exp(sqrt(2/hbar)*v.T inv(Q)y-v.T inv(Q)conj(Q)v/2).
    Conjugate the bra coefficients before carrying out the integral over real x.
    In the Gaussian integral use the analytic determinant square root for complex
    symmetric precision W with positive-definite real part: its log determinant
    is the sum of principal logarithms of W's eigenvalues, not necessarily the
    principal scalar logarithm of det(W). The supplied lifted logdetQ values
    are part of the state and may differ by multiples of 2*pi*i for identical Q.

    Parameters
    ----------
    q_bra, p_bra, q_ket, p_ket : finite real arrays, shape (D,), D>=1
    Q_bra, P_bra, Q_ket, P_ket : finite complex arrays, shape (D,D)
        Each frame obeys Q.T P-P.T Q=0 and Q.conj().T P-P.conj().T Q=2i I.
    S_bra, S_ket : finite real scalars
    logdet_bra, logdet_ket : finite complex scalars, exp(logdet)=det(Q)
    hbar : finite positive float

    Returns
    -------
    B : complex symmetric array, shape (2D,2D)
    b : complex array, shape (2D,)
    G : complex scalar, the vacuum overlap

    Raises
    ------
    ValueError
        For invalid or nonfinite shapes/data, nonpositive hbar, canonical defects
        above 1e-8, or determinant mismatch exceeding 1e-8*max(1,abs(det(Q))).
    """
    return B, b, G
```

### Step 7

07_fock_overlap

Goal
----
Extract normalized overlap coefficients for repeated vibrational occupations.

```python
def fock_overlap(B, b, vacuum_overlap, bra_labels, ket_labels):
    r"""For F(z)=G*exp(z.T B z/2+b.T z), return the matrix
    O[k,l]=G*[partial_z**(k,l) exp(z.T B z/2+b.T z)] at z=0 / sqrt(k!*l!).
    Here z consists of D bra variables followed by D ket variables. Multi-index
    factorials mean products of ordinary factorials. Do not take absolute values.
    The zero-occupation derivative is one. Label order and duplicates are preserved.

    Parameters
    ----------
    B : finite complex symmetric array, shape (2D,2D), D>=1
    b : finite complex array, shape (2D,)
    vacuum_overlap : finite complex scalar G
    bra_labels, ket_labels : nonnegative integer-valued arrays, shapes (L,D),(R,D)
        Each row has total occupation <=8. Empty arrays of shape (0,D) are allowed.

    Returns
    -------
    overlaps : complex array, shape (L,R)

    Raises
    ------
    ValueError
        For incompatible or nonfinite data; B symmetry error above 1e-10;
        negative/noninteger occupations; or a row total above eight.
    """
    return overlaps
```

### Step 8

08_correlation_path

Goal
----
Contract the basis overlaps with coherent dipole coefficients at each time.

```python
def correlation_path(qs, ps, actions, Qs, Ps, logdets, labels, coefficients, hbar):
    r"""Evaluate C(t)=sum_{a=x,y,z} sum_{k,l} conj(c[k,a])*O[k,l](0,t)*c[l,a]/3.
    Coefficients are constant in the moving Hagedorn basis. Do not normalize C
    and do not delete off-diagonal occupation pairs. Initial state is path entry 0.
    The wavefunction/basis and determinant lifts have the overlap_generator contract.
    Use earlier overlap_generator and fock_overlap to construct O at each time.

    Parameters
    ----------
    qs, ps : finite real arrays, shape (N,D), N,D>=1
    actions : finite real array, shape (N,)
    Qs, Ps : finite canonical complex frames, shape (N,D,D)
    logdets : finite lifted log determinants, shape (N,)
    labels : nonnegative integer-valued array, shape (L,D), row totals<=8
    coefficients : finite complex array, shape (L,3)
    hbar : finite positive float

    Returns
    -------
    correlation : complex array, shape (N,)
        C(0)=sum(abs(coefficients)**2)/3. A zero coefficient matrix returns zeros.

    Raises
    ------
    ValueError
        For invalid or nonfinite shapes/data or nonpositive hbar; invalid canonical
        frames/determinant lifts under overlap_generator; or invalid labels under
        fock_overlap.
    """
    return correlation
```

### Step 9

09_solve

Goal
----
Integrate the complete single-Hessian dipole correlation on the stated grid.

```python
def solve(data):
    r"""Return the normalized finite-grid spectral ordinate

    J = dt/(3*pi*C(0)) * Re sum_j w_j*C(t_j)
        * exp(i*(omega + E0/hbar)*t_j - eta*t_j**2),
    where t_j = j*dt, j=0,...,nsteps.

    Simpson weights: 1 at endpoints, 4 at odd interior indices,
    2 at even interior indices. No additional frequency prefactor,
    tail correction, or clipping.

    Use earlier steps in this order:
    initial_frame; potential_jet at q_reference; centre_action with
    zero initial momentum and action; width_path with the constant
    reference Hessian; dipole_coefficients; correlation_path.
    correlation_path calls overlap_generator and fock_overlap.

    The potential uses coordinates q-origin.
    The dipole uses coordinates q-q_initial.

    Parameters
    ----------
    data : dict
        Required fields:
        mass, ground_hessian: real SPD arrays, shape (D,D).
        q_initial, origin, q_reference: real arrays, shape (D,).
        potential_powers: nonnegative integer-valued array, shape (R,D).
        potential_coefficients: real monomial coefficients, shape (R,).
        dipole_powers: nonnegative integer-valued array, shape (U,D),
            U>=1, total degree of each row <=6.
        dipole_coefficients: real or complex array, shape (U,3).
        hbar, dt: finite positive scalars.
        eta: finite nonnegative scalar.
        omega: finite real scalar.
        nsteps: even integer >=2.

        All arrays are finite. Dimensions and canonical conditions
        follow the earlier step contracts. The time grid must resolve
        the determinant phase as specified in width_path.

    Returns
    -------
    ordinate : float
        Normalized spectral ordinate as a native Python float.

    Raises
    ------
    ValueError
        For a non-dict or missing fields; invalid spectral scalars
        or grid; invalid arrays under the earlier step contracts;
        or zero initial dipole-state norm.
    """
    return ordinate
```
