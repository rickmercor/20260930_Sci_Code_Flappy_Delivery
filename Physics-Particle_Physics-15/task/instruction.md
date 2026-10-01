# Physics-Particle_Physics-15

## Background

The constructed point uses GeV and the charged-lepton flavor basis \((e,\mu,\tau)\):
\[
m_D^{(0)}=\begin{pmatrix}14+2i&4-3i&2+i\\3+2i&19-i&5+4i\\2-i&6+3i&24+2i\end{pmatrix},\qquad
m_R=\begin{pmatrix}135+7i&11-4i&7+3i\\11-4i&193-9i&13+5i\\7+3i&13+5i&271+12i\end{pmatrix}.
\]
The pole reference data are \(m_\ell=(0.000511,0.10566,1.77686)\), \(m_W=80.379\), \(m_Z=91.1876\), \(m_h=125.1\), \(m_t=172.76\), \(\alpha^{-1}=137.035999084\), and \(\alpha_s=0.1181\).
The two gauges are \((\xi_W^a,\xi_Z^a)=(0.37,2.4)\) and \((\xi_W^b,\xi_Z^b)=(3.1,0.64)\).
These matrices define a constructed numerical test point; the gauge contrast and its normalization are task-defined diagnostics.

The mass convention is \(\mathcal M=\begin{psmallmatrix}0&m_D\\m_D^T&m_R\end{psmallmatrix}=V\widehat m V^T\), with positive ascending Takagi masses and \(B\) the first three rows of \(V\); \(C=B^\dagger B\), \(s_W^2=1-m_W^2/m_Z^2\), \(\Delta=1/\epsilon-\gamma_E+\ln4\pi\).

Repeated neutrino indices run over all six states; charged-lepton indices run over three flavors; a star means elementwise conjugation.
All scalar loop integrals are dispersive real parts. With nonnegative masses \(a,b\),
\[
B_0(p^2;a,b)=\Delta-\int_0^1\ln\left|\frac{xa^2+(1-x)b^2-x(1-x)p^2}{\mu^2}\right|\,dx .
\]
Interior logarithmic singularities have their integrable limiting meaning. Couplings remain complex.

The phase family and its coincidence susceptibility are constructed diagnostics of the source prescription.

## Problem

Determine the local susceptibility of an accidental gauge-scheme coincidence in the constructed type-I seesaw family \(m_D(\theta,\eta)=m_D^{(0)}\operatorname{diag}(e^{i\theta},e^{i\eta},1)\), with fixed \(m_R\).
For \(f=e,\mu\), use the Majorana invariant \(I_f=\operatorname{Im}[(B_{f1}B_{f2}^{*})^2]\): let \(d_f\) be the difference between its first-order shifts under \(\widetilde{\delta B}(\xi^a)-\delta B\) and \(\widetilde{\delta B}(\xi^b)-\delta B\), where the tilde denotes the full anti-Hermitian on-shell prescription and \(\delta B\) denotes the source's gauge-independent Majorana prescription.
Let \(u_f\) be the invariant's first-order response to the coefficient of \(\Delta\) retained by that prescription from the sector carrying gauge parameters.
Among all isolated solutions of \(d_e=d_\mu=0\) in \((\theta,\eta)\in[-0.8,0]\times[-1.5,0.7]\) radians, select the one maximizing \(\mathcal U=\sqrt{u_e^2+u_\mu^2}\).
At the selected point compute \(S=\sigma_{\max}(J)\), where \(J=\partial(d_e/u_e,d_\mu/u_\mu)/\partial(\theta,\eta)\), and report the four-value certificate \((\theta,\eta,\mathcal U,\sigma_{\min}(J))\) with a compact justification of the Majorana prescription, branch selection and normalization.
Use \(\mu=m_Z\), six significant figures for \(S\), and eight for the certificate; phase tolerance is \(10^{-7}\) radians and the other quantities have tolerance \(10^{-13}+5\times10^{-5}|q|\).
The coding deliverable composes every preceding public subproblem; NumPy and SciPy are available, with a 120-second limit per subproblem.

Output Format Requirements:

Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.

You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.

Rules:

- The tags are required. Do not omit them or leave them empty.

- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.

- Put only that one number between the tags. No units, no words, no extra lines.

Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.

Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

seesaw_basis

Goal
----
Exact Majorana mass basis

```python
def seesaw_basis(md: 'np.ndarray', mr: 'np.ndarray') -> 'np.ndarray':
    """Exact Majorana mass basis.

    Parameters
    ----------
    md : complex ndarray, shape (n,n)
        Dirac mass matrix in GeV, 1 <= n <= 3.
    mr : complex ndarray, shape (n,n)
        Symmetric sterile Majorana mass matrix in GeV. The assembled mass
        matrix has distinct positive Takagi masses and a nonsingular md.

    Returns
    -------
    result : complex ndarray, shape (2*n+1,2*n)
        Row 0 contains positive ascending masses (GeV) with zero imaginary
        part. Rows 1 through 2*n contain V, ordered as active then sterile
        rows, with M=V diag(m) V.T. Largest-entry column signs follow the
        step background. The first n rows of V form B.
    """
    return result
```

### Step 2

pv_laurent

Goal
----
Dispersive two-point Laurent data

```python
def pv_laurent(queries: 'np.ndarray', scale: float) -> 'np.ndarray':
    """Dispersive two-point Laurent data.

    Parameters
    ----------
    queries : float ndarray, shape (q,3)
        Rows [p_squared,a,b], respectively in GeV^2, GeV, GeV; all are
        nonnegative and each row has p_squared+a+b > 0.
    scale : float
        Positive subtraction scale in GeV.

    Returns
    -------
    result : float ndarray, shape (q,2)
        In input row order, columns are the Delta coefficient and dispersive
        finite part of B0. Both columns are dimensionless.
    """
    return result
```

### Step 3

charged_gauge

Goal
----
Charged-lepton gauge sector

```python
def charged_gauge(b: 'np.ndarray', m: 'np.ndarray', ml: 'np.ndarray', mw: float, alpha: float, sw2: float, xi_w: float, loops: 'np.ndarray') -> 'np.ndarray':
    """Charged-lepton gauge sector.

    Parameters
    ----------
    b : complex ndarray, shape (n,N), N=2*n, 1 <= n <= 3
        Charged-current mixing, b b.H=I and b diag(m) b.T=0.
    m : float ndarray, shape (N,)
        Positive Majorana masses in GeV, ordered like b columns.
    ml : float ndarray, shape (n,)
        Nonnegative charged-lepton masses in GeV, in b row order.
    mw : float
        Positive W mass in GeV.
    alpha, sw2 : float
        Positive alpha and 0 < sin(theta_W)^2 < 1.
    xi_w : float
        Nonnegative W gauge parameter.
    loops : float ndarray, shape (n,N,2)
        Indices: charged-lepton external row, internal neutrino, Laurent column.
        Final-axis order [Delta coefficient, finite part], as defined for
        L in the common background; entries are dimensionless.

    Returns
    -------
    result : complex ndarray, shape (n,N,2)
        Dimensionless G^ell, active row then mass column then Laurent
        axis [Delta coefficient, finite part].
    """
    return result
```

### Step 4

neutrino_w_gauge

Goal
----
Majorana charged-current gauge sector

```python
def neutrino_w_gauge(b: 'np.ndarray', m: 'np.ndarray', ml: 'np.ndarray', mw: float, alpha: float, sw2: float, xi_w: float, loops: 'np.ndarray') -> 'np.ndarray':
    """Majorana charged-current gauge sector.

    Parameters
    ----------
    b : complex ndarray, shape (n,N), N=2*n, 1 <= n <= 3
        Charged-current mixing, b b.H=I and b diag(m) b.T=0.
    m : float ndarray, shape (N,)
        Positive Majorana masses in GeV, ordered like b columns.
    ml : float ndarray, shape (n,)
        Nonnegative charged-lepton masses in GeV, in b row order.
    mw : float
        Positive W mass in GeV.
    alpha, sw2 : float
        Positive alpha and 0 < sin(theta_W)^2 < 1.
    xi_w : float
        Nonnegative W gauge parameter.
    loops : float ndarray, shape (N,n,2)
        Indices: external neutrino, internal charged lepton, Laurent column.
        Final-axis order [Delta coefficient, finite part], as defined for
        W in the common background; entries are dimensionless.

    Returns
    -------
    result : complex ndarray, shape (n,N,2)
        Dimensionless G^W, active row then mass column then Laurent
        axis [Delta coefficient, finite part].
    """
    return result
```

### Step 5

neutrino_z_gauge

Goal
----
Off-diagonal neutral-current gauge sector

```python
def neutrino_z_gauge(b: 'np.ndarray', m: 'np.ndarray', mw: float, mz: float, alpha: float, sw2: float, xi_z: float, loops: 'np.ndarray') -> 'np.ndarray':
    """Off-diagonal neutral-current gauge sector.

    Parameters
    ----------
    b : complex ndarray, shape (n,N), N=2*n, 1 <= n <= 3
        Charged-current mixing, b b.H=I and b diag(m) b.T=0.
    m : float ndarray, shape (N,)
        Positive Majorana masses in GeV, ordered like b columns.
    mw : float
        Positive W mass in GeV.
    mz : float
        Positive Z mass in GeV.
    alpha, sw2 : float
        Positive alpha and 0 < sin(theta_W)^2 < 1.
    xi_z : float
        Nonnegative Z gauge parameter.
    loops : float ndarray, shape (N,N,2)
        Indices: external neutrino, internal neutrino, Laurent column.
        Final-axis order [Delta coefficient, finite part], as defined for
        Z in the common background; entries are dimensionless.

    Returns
    -------
    result : complex ndarray, shape (n,N,2)
        Dimensionless G^Z, active row then mass column then Laurent
        axis [Delta coefficient, finite part].
    """
    return result
```

### Step 6

majorana_diagonal_gauge

Goal
----
Diagonal Majorana gauge sector

```python
def majorana_diagonal_gauge(b: 'np.ndarray', m: 'np.ndarray', mw: float, mz: float, alpha: float, sw2: float, xi_z: float, loops: 'np.ndarray') -> 'np.ndarray':
    """Diagonal Majorana gauge sector.

    Parameters
    ----------
    b : complex ndarray, shape (n,N), N=2*n, 1 <= n <= 3
        Charged-current mixing, b b.H=I and b diag(m) b.T=0.
    m : float ndarray, shape (N,)
        Positive Majorana masses in GeV, ordered like b columns.
    mw : float
        Positive W mass in GeV.
    mz : float
        Positive Z mass in GeV.
    alpha, sw2 : float
        Positive alpha and 0 < sin(theta_W)^2 < 1.
    xi_z : float
        Nonnegative Z gauge parameter.
    loops : float ndarray, shape (N,N,2)
        Indices: external neutrino, internal neutrino, Laurent column.
        Final-axis order [Delta coefficient, finite part], as defined for
        Z in the common background; entries are dimensionless.

    Returns
    -------
    result : complex ndarray, shape (n,N,2)
        Dimensionless G^d, active row then mass column then Laurent
        axis [Delta coefficient, finite part].
    """
    return result
```

### Step 7

restored_uv

Goal
----
Gauge-independent ultraviolet completion

```python
def restored_uv(b: 'np.ndarray', m: 'np.ndarray', mw: float, alpha: float, sw2: float) -> 'np.ndarray':
    """Gauge-independent ultraviolet completion.

    Parameters
    ----------
    b : complex ndarray, shape (n,N), N=2*n, 1 <= n <= 3
        Charged-current mixing matrix with b b.H=I and b diag(m) b.T=0.
    m : float ndarray, shape (N,)
        Positive Majorana pole masses in GeV, ordered like b columns.
    mw : float
        Positive W mass in GeV.
    alpha, sw2 : float
        Positive fine-structure constant and 0 < sin(theta_W)^2 < 1.

    Returns
    -------
    result : complex ndarray, shape (n,N)
        Dimensionless coefficient U of Delta retained from the complete
        gauge-parameter sector, in b's row and column order.
    """
    return result
```

### Step 8

scheme_response

Goal
----
Majorana-invariant scheme response

```python
def scheme_response(b: 'np.ndarray', finite_a: 'np.ndarray', finite_b: 'np.ndarray', restored: 'np.ndarray', diagonal_a: 'np.ndarray', diagonal_b: 'np.ndarray', flavor: int, pair: 'Sequence[int]') -> 'np.ndarray':
    """Majorana-invariant scheme response.

    Parameters
    ----------
    b : complex ndarray, shape (n,N)
        Dimensionless reference mixing matrix, active flavors by mass states.
    finite_a, finite_b : complex ndarray, shape (n,N)
        Dimensionless finite scheme-difference matrices at gauges a and b.
    restored : complex ndarray, shape (n,N)
        Dimensionless retained ultraviolet coefficient U.
    diagonal_a, diagonal_b : complex ndarray, shape (n,N)
        Dimensionless diagonal-neutrino portions of the two finite matrices.
    flavor : int
        Zero-based active row, 0 <= flavor < n.
    pair : sequence of two int
        Distinct zero-based mass-column indices (i,j), each in [0,N).
        The supplied matrices have a nonzero D_I[restored].

    Returns
    -------
    result : float ndarray, shape (5,)
        Dimensionless entries ordered [R, D_I[finite_a], D_I[finite_b],
        D_I[restored], D_I[diagonal_a-diagonal_b]], for
        I=Im[(b[flavor,i]*conj(b[flavor,j]))^2].
        Here D_I[H] = d/d epsilon I(b+epsilon H) at epsilon=0 for real
        epsilon, and R=(D_I[finite_a]-D_I[finite_b])/D_I[restored].
    """
    return result
```

### Step 9

gauge_scheme_benchmark

Goal
----
Complete gauge-scheme benchmark

```python
def gauge_scheme_benchmark(md: 'np.ndarray', mr: 'np.ndarray', ml: 'np.ndarray', ew: 'np.ndarray', scale: float, gauges: 'np.ndarray', flavor: int=0, pair: 'Sequence[int]'=(0, 1)) -> 'np.ndarray':
    """Complete gauge-scheme benchmark.

    Parameters
    ----------
    md, mr : complex ndarray, shape (n,n), n=2 or 3
        Dirac and symmetric sterile mass matrices in GeV, defining distinct
        positive Takagi masses in the convention of seesaw_basis.
    ml : float ndarray, shape (n,)
        Positive charged-lepton masses in GeV, in md's active row order.
    ew : float ndarray, shape (3,)
        Entries [mW,mZ,alpha], masses in GeV, 0 < mW < mZ and alpha > 0.
    scale : float
        Positive subtraction scale in GeV.
    gauges : float ndarray, shape (2,2)
        Rows a,b; columns xiW,xiZ; all entries nonnegative.
    flavor : int, default 0
        Zero-based active row defining I.
    pair : sequence of two int, default (0,1)
        Distinct zero-based Takagi columns defining I. Inputs have D_I[U]!=0.

    Returns
    -------
    result : float ndarray, shape (5,)
        Dimensionless [R, D_I[F(a)], D_I[F(b)], D_I[U],
        D_I[F_diag(a)-F_diag(b)]]. These are the per-flavor scheme responses.
        Here F(a) and F(b) are the finite scheme-difference matrices,
        U is the retained ultraviolet coefficient, and D_I[H] is the
        first-order shift of I under B -> B+epsilon H for real epsilon.
        R=(D_I[F(a)]-D_I[F(b)])/D_I[U].
    """
    return result
```

### Step 10

coincidence_susceptibility

Goal
----
Phase-space coincidence and local susceptibility

```python
def coincidence_susceptibility(md: 'np.ndarray', mr: 'np.ndarray', ml: 'np.ndarray', ew: 'np.ndarray', scale: float, gauges: 'np.ndarray', box: 'np.ndarray', flavors: 'Sequence[int]'=(0, 1), pair: 'Sequence[int]'=(0, 1)) -> 'np.ndarray':
    """Gauge-coincidence susceptibility.

    Parameters
    ----------
    md, mr : complex ndarray, shape (3,3)
        Base Dirac matrix and symmetric sterile matrix, in GeV. Phase theta
        multiplies column 0 of md and eta multiplies column 1. mr stays fixed.
    ml : float ndarray, shape (3,)
        Positive charged-lepton masses in GeV, in active row order.
    ew : float ndarray, shape (3,)
        [mW,mZ,alpha], with 0 < mW < mZ (GeV) and alpha > 0.
    scale : float
        Positive subtraction scale in GeV.
    gauges : float ndarray, shape (2,2)
        Rows a,b and columns xiW,xiZ, nonnegative.
    box : float ndarray, shape (2,2)
        Rows theta,eta and columns lower,upper, in radians within [-pi,pi].
        The rectangle contains isolated coincidence points and a unique
        maximum of their UV-response norms. Takagi masses are distinct and
        positive throughout; both flavor responses u are nonzero at the
        selected coincidence.
    flavors : sequence of two int, default (0,1)
        Distinct active rows whose gauge contrasts vanish, in Jacobian row order.
    pair : sequence of two int, default (0,1)
        Distinct zero-based ascending Takagi columns defining each invariant.

    Returns
    -------
    result : float ndarray, shape (5,)
        [largest Jacobian singular value, theta, eta, UV-response norm,
        smallest Jacobian singular value]. Phases are in radians; the norm
        is dimensionless, and singular values are per radian. The Jacobian
        differentiates both normalized contrasts with respect to theta,eta.
        Compare numerical components with absolute tolerance 1e-6.
    """
    return result
```
