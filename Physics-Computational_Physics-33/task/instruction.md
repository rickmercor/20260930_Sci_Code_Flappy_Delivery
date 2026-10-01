# Crystal-cut advantage of cubic magneto-optic anisotropy

## Background

The source expands the cubic-crystal permittivity as epsilon_ij=epsilon_d delta_ij+K_ijk M_k+G_ijkl M_k M_l+H_ijklm M_k M_l M_m. The lab axes are M=(M_T,M_L,M_P)=(M_x,M_y,M_z). K is the isotropic Levi-Civita tensor times the complex coefficient K. The observable quadratic parameters are G_s=G_11-G_12 and 2G_44, with Delta_G=G_s-2G_44. The cubic tensor has the two independent complex parameters H_123 and H_125, with Delta_H=H_123-3H_125.

Steps 01 and 02 use the paper's source-frame tensor layouts, all required permutation copies, its rank-five/rank-four tensor rotation laws, the printed (001)-to-(111) reference matrix, and Appendix A's z-rotation matrix. A crystallographically equivalent but differently phased (111) basis changes the signed threefold coefficients on the specified alpha grid, so use the printed layouts and transformations exactly.

All complex packets use pair order `(23,31,12,32,13,21)` and final axis `[real,imag]`. Step 03 preserves separate order-1, order-2, and order-3 packets. Step 04 uses the paper's thin-film approximation

`Phi_s=A_s*(epsilon_yx-epsilon_yz*epsilon_zx/epsilon_d)+B_s*epsilon_zx`

`Phi_p=-A_p*(epsilon_xy-epsilon_zy*epsilon_xz/epsilon_d)+B_p*epsilon_xz`.

The cross-products are truncated through total magnetization order three: retain `(1,1)`, `(1,2)`, and `(2,1)` only. This retains the mixed cubic term proportional to `K*Delta_G/epsilon_d` for the (111) response while excluding fourth- and higher-order products.

The eight directions are `mu=(0,45,...,315)` degrees with `M_T=cos(mu)`, `M_L=sin(mu)`, `M_P=0`. The four separated channels are `(M_L+M_L^3)`, `M_L*M_T`, `(M_T^2-M_L^2)`, and `M_T^3`, in that order. The first and fourth carry the cubic anisotropy; the middle two carry the quadratic anisotropy. The expected angular harmonic is fourfold for (001) and threefold for (111). Uniform sampling with n divisible by 12 resolves both exactly.

For each cut, form the stated weighted complex Fourier coefficient for every channel and polarization on the deterministic warped angle grid. The warp remains monotone because its derivative is bounded below by `1-0.07*5-0.03*7=0.44`, and the acquisition weights are positive because they are at least `1-0.2-0.1=0.7`. Euclidean norms include the two selected channels and both s/p polarizations. The difference of log1p-transformed per-cut ratios measures the orientation-dependent balance while retaining sensitivity to both absolute cut normalizations. The warped scan, weights, aggregation, and fixed coefficients define a new computational benchmark. The paper supplies the magneto-optic model.

## Problem

Implement the seven ordered Python functions and use their composed result to compare cubic-in-magnetization and quadratic magneto-optic anisotropies in (001)- and (111)-cut cubic crystals. The source paper defines the fifth-rank cubic tensor, fourth-rank quadratic tensor, crystal-frame transformations, perturbative Kerr relation, and eight-directional measurement channels. The requested scalar is a new deterministic benchmark and is not reported in the paper.

Use the paper's coordinate and sign conventions exactly: magnetization is ordered `[M_T,M_L,M_P]=[M_x,M_y,M_z]`; the six off-diagonal entries are ordered `(23,31,12,32,13,21)`; positive sample rotation is the clockwise-from-above convention in Appendix A; and the (111) reference orientation is the paper's `[-211] || x`, `[111] || z` orientation. Complex arrays are transported as a final `[real,imag]` axis. Apply the printed frame transformations once in this specified reference basis. Steps 01-02 must retain the signed tensor responses for general out-of-plane magnetization including the terms absent from in-plane-only formulas.

For the final benchmark use, without intermediate rounding,

`epsilon_d=-9.6+14.2j`, `K=0.034+0.021j`, `G_s=0.018-0.011j`, `2G_44=-0.007+0.016j`, `H_123=0.0042+0.0028j`, `H_125=-0.0011+0.0007j`, `A_s=0.73-0.18j`, `B_s=0.11+0.06j`, `A_p=-0.61+0.27j`, `B_p=0.16-0.09j`, and `n_angles=48`.

For `j=0,...,n_angles-1`, define `beta_j=2*pi*j/n_angles`, the monotone warped sample angle `alpha_j=beta_j+0.07*sin(5*beta_j+0.17)+0.03*cos(7*beta_j-0.31)`, and the positive acquisition weight `w_j=1+0.2*cos(3*beta_j+0.4)+0.1*sin(8*beta_j-0.2)`. At each `alpha_j`, evaluate all eight in-plane magnetization directions and separate the four paper channels. Use harmonic 4 for (001) and harmonic 3 for (111). For each channel and polarization separately, the weighted coefficient convention is `(2/sum_j w_j)*sum_j w_j*channel_j*exp(-1j*m*alpha_j)`. The CMOKE norm combines channels 0 and 3 over both polarizations; the QMOKE norm combines channels 1 and 2. The per-cut ratio is `CMOKE_norm/QMOKE_norm`. The reported logarithmic crystal advantage is `numpy.log1p(ratio_111)-numpy.log1p(ratio_001)`, after one final `numpy.round(value,10)`.

In `<reasoning>`, briefly explain the seven functions. Use Section II A of the paper to give the source-frame cubic polynomial for epsilon_yz, the quadratic difference epsilon_yy-epsilon_zz, and the quadratic epsilon_yz term. These establish the tensor contractions before rotation. Report Delta_G and Delta_H; the two complex (111) weighted coefficients C[0,s] and C[1,s]; and, for each cut, the CMOKE norm, QMOKE norm, and their ratio. Report intermediate numerical values to six significant digits (relative tolerance 5e-6 per nonzero real or imaginary component); keep full precision during computation. Put the result of the single final rounding to 10 decimal places in <final_answer>.

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_evaluate_rotated_cmoke_cubic_response

Goal
----
Evaluate the paper's rotated fifth-rank cubic magneto-optic response.

```python
import numpy as np

def evaluate_rotated_cmoke_cubic_response(magnetization: np.ndarray, h123: complex, h125: complex, crystal_cut: str, alpha: float) -> np.ndarray:
    '''Return the rotated cubic-H off-diagonal packet.

    Parameters
    ----------
    magnetization : np.ndarray, shape (N,3)
        Nonempty finite rows in M_T, M_L, M_P order.
    h123, h125 : complex
        Finite scalar cubic tensor coefficients.
    crystal_cut : str
        Reference cut, '001' or '111'.
    alpha : float
        Finite non-Boolean scalar sample angle in radians.

    Returns
    -------
    response : np.ndarray, shape (N,6,2), float
        Pair order (23,31,12,32,13,21), final axis real/imag.

    Raises
    ------
    ValueError
        An input has an invalid shape, is nonfinite, or violates a scalar or cut constraint.'''
    return np.empty((0, 6, 2), dtype=float)
```

### Step 2

02_evaluate_rotated_qmoke_quadratic_response

Goal
----
Evaluate the rotated cubic-crystal quadratic magneto-optic response.

```python
import numpy as np

def evaluate_rotated_qmoke_quadratic_response(magnetization: np.ndarray, g_s: complex, two_g44: complex, crystal_cut: str, alpha: float) -> np.ndarray:
    '''Return the rotated quadratic-G off-diagonal packet.

    Parameters
    ----------
    magnetization : np.ndarray, shape (N,3)
        Nonempty finite rows in M_T, M_L, M_P order.
    g_s, two_g44 : complex
        Finite scalars G_11-G_12 and 2G_44.
    crystal_cut : str
        Reference cut, '001' or '111'.
    alpha : float
        Finite non-Boolean scalar sample angle in radians.

    Returns
    -------
    response : np.ndarray, shape (N,6,2), float
        Pair order (23,31,12,32,13,21), final axis real/imag.

    Raises
    ------
    ValueError
        An input has an invalid shape, is nonfinite, or violates a scalar or cut constraint.'''
    return np.empty((0, 6, 2), dtype=float)
```

### Step 3

03_pack_magnetooptic_orders

Goal
----
Pack the linear, quadratic, and cubic magneto-optic orders.

```python
import numpy as np

def pack_magnetooptic_orders(magnetization: np.ndarray, k_linear: complex, quadratic_response: np.ndarray, cubic_response: np.ndarray) -> np.ndarray:
    '''Pack responses by magnetization order.

    Parameters
    ----------
    magnetization : np.ndarray, shape (N,3)
        Nonempty finite rows in M_T, M_L, M_P order.
    k_linear : complex
        Finite scalar linear magneto-optic coefficient.
    quadratic_response, cubic_response : np.ndarray, shape (N,6,2)
        Finite packets with matching event counts and final real/imag axis.

    Returns
    -------
    orders : np.ndarray, shape (N,3,6,2), float
        Axes event, order 1/2/3, pair (23,31,12,32,13,21), real/imag.

    Raises
    ------
    ValueError
        An input has an invalid shape, is nonfinite, or event counts do not match.'''
    return np.empty((0, 3, 6, 2), dtype=float)
```

### Step 4

04_compute_third_order_kerr_angles

Goal
----
Evaluate the perturbative s- and p-polarized Kerr angles through cubic order.

```python
import numpy as np

def compute_third_order_kerr_angles(order_packets: np.ndarray, epsilon_d: complex, a_s: complex, b_s: complex, a_p: complex, b_p: complex) -> np.ndarray:
    '''Return perturbative Kerr angles.

    Parameters
    ----------
    order_packets : np.ndarray, shape (N,3,6,2)
        Nonempty finite packets in order 1/2/3 and the specified pair order.
    epsilon_d : complex
        Finite nonzero scalar diagonal permittivity.
    a_s, b_s, a_p, b_p : complex
        Finite scalar optical weights for s and p polarization.

    Returns
    -------
    kerr : np.ndarray, shape (N,2,2), float
        Polarization order s,p, final axis real/imag.

    Raises
    ------
    ValueError
        An input has an invalid shape, is nonfinite, or epsilon_d is zero.'''
    return np.empty((0, 2, 2), dtype=float)
```

### Step 5

05_build_eight_directional_design

Goal
----
Build the paper's eight-directional magnetization design.

```python
import numpy as np

def build_eight_directional_design(sample_angles: np.ndarray) -> np.ndarray:
    '''Return sample angles and eight in-plane directions.

    Parameters
    ----------
    sample_angles : np.ndarray, shape (A,)
        Nonempty finite sample angles in radians; preserve their order.

    Returns
    -------
    design : np.ndarray, shape (A,8,4), float
        Columns alpha,M_T,M_L,M_P in fixed direction order.

    Raises
    ------
    ValueError
        Sample angles are empty, nonfinite, or not one-dimensional.'''
    return np.empty((0, 8, 4), dtype=float)
```

### Step 6

06_separate_eight_directional_channels

Goal
----
Separate the four eight-directional Kerr channels.

```python
import numpy as np

def separate_eight_directional_channels(kerr_scan: np.ndarray) -> np.ndarray:
    '''Return the four separated channel packets.

    Parameters
    ----------
    kerr_scan : np.ndarray, shape (A,8,2,2)
        Nonempty finite scan in the prescribed direction order, with s/p
        polarization and final real/imag axes.

    Returns
    -------
    channels : np.ndarray, shape (A,4,2,2), float
        Channel, polarization s/p, and real/imag axes.

    Raises
    ------
    ValueError
        The scan is empty, nonfinite, or has an invalid shape.'''
    return np.empty((0, 4, 2, 2), dtype=float)
```

### Step 7

07_compute_cmoke_crystal_advantage

Goal
----
Compose the logarithmic (111)-versus-(001) CMOKE anisotropy advantage.

```python
import numpy as np

def compute_cmoke_crystal_advantage(epsilon_d: complex, k_linear: complex, g_s: complex, two_g44: complex, h123: complex, h125: complex, a_s: complex, b_s: complex, a_p: complex, b_p: complex, n_angles: int) -> float:
    '''Return the rounded crystal-cut anisotropy advantage.

    Parameters
    ----------
    epsilon_d : complex
        Finite nonzero scalar diagonal permittivity.
    k_linear, g_s, two_g44, h123, h125 : complex
        Finite scalar magneto-optic coefficients in the stated convention.
    a_s, b_s, a_p, b_p : complex
        Finite scalar optical weights for s and p polarization.
    n_angles : int
        Non-Boolean grid size, at least 24 and divisible by 12.

    Returns
    -------
    advantage : float
        Log1p contrast of the (111) and (001) cubic-to-quadratic harmonic ratios.

    Raises
    ------
    ValueError
        An input violates its contract, a quadratic norm is numerically zero,
        or a computed ratio is nonfinite. The background specifies the norm threshold.'''
    return 0.0
```
