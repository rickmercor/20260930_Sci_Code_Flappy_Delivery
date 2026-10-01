# Mathematics-Computational_Mechanics-31

## Background

A finite pulse produces a phase-dependent superposition of elastic modes at a receptor. When a short record does not resolve neighbouring resonances, agreement in peak frequencies need not imply agreement in measured amplitudes. The full anisotropic grain response and the traction-free approximation determine the excitation and observation of those modes.

## Problem

Resonance frequencies alone can conceal the effect of grain structure on a finite specimen, because the measured amplitudes also depend on excitation, receptor position and interference between unresolved elastic modes. For the titanium specimen specified below, determine the ratio of the band-summed squared vector Fourier amplitude of its complete elastic receptor displacement to the sum of those powers when each distinct elastic eigenspace acts alone. Use the complete finite record in both quantities, so this is a ratio of measured displacement powers rather than mechanical energies.

The displacement has three components and depends only on the specimen coordinate $x_1$; retain the full rotated single-crystal constitutive law in every grain, and measure elastic displacement relative to the specimen centre of mass. The specimen is represented on an odd periodic voxel grid with the massless compliant exterior defined below; exterior displacements remain unconstrained and satisfy mechanical equilibrium. The prescribed discretisation defines the model for this task, including its small exterior stiffness, and all elastic modes of that model contribute to the response.

| Quantity | Value or convention |
| --- | --- |
| Specimen length and density | $l=10^{-3}$ m; $\rho=4506.3$ kg per cubic metre |
| Grain voxel counts, from the loaded face | $(7,5,8,6,7,4,9,5,5,4)$; $60$ specimen voxels in total |
| Grid | $67$ periodic voxels of size $h=l/60$; specimen indices $0$ through $59$, followed by seven exterior voxels |
| Crystal stiffness | In GPa, $c_{11}=c_{22}=162.4$, $c_{33}=180.7$, $c_{12}=92.0$, $c_{13}=c_{23}=69.0$, $c_{44}=c_{55}=46.7$, $c_{66}=35.2$, with all other engineering-Voigt entries zero except symmetry-related entries |
| Tensor convention | Engineering Voigt ordering $(11,22,33,23,13,12)$; sixfold crystal axis along crystal axis 3 |
| Orientation convention | Crystal axes expressed in the specimen frame are the columns of $R=Z(\varphi_1)X(\Phi)Z(\varphi_2)$; $X$ and $Z$ are right-handed active rotations about axes 1 and 3 |
| Spatial discretisation | Forward voxel difference for displacement gradients and its adjoint for positive stiffness; periodic on the complete grid |
| Exterior properties | Zero density and acoustic tensor $10^{-7}\operatorname{diag}(\bar A_{11},(\bar A_{22}+\bar A_{33})/2,(\bar A_{22}+\bar A_{33})/2)$, where $A_{ik}=C'_{i1k1}$ is the rotated grain acoustic tensor and $\bar A$ is its specimen voxel average |
| Surface traction | At voxel 0, $T(t)(1,2,-1)/\sqrt{6}$, with $T(t)=10^6[\exp(-(t-t_0)^2/(2s^2))-\exp(-(t-t_0-s)^2/(2s^2))]$ Pa, $t_0=80$ ns and $s=20$ ns |
| Receptor | All three displacement components at voxel 59 |
| Time record | Average-acceleration Newmark convention, $\beta=1/4$, $\gamma=1/2$, $\Delta t=2$ ns and $384$ steps; zero initial displacement, velocity and acceleration; traction evaluated at each step's end time; all $385$ samples retained, including time zero |
| Spectral band | Closed interval $[2,9]$ MHz; retain its non-negative DFT bin centres, with no window and no zero padding |
| Numerically repeated frequencies | Starting from the lowest unused positive frequency, one eigenspace cluster contains consecutive frequencies at most $(1+10^{-8})$ times that first frequency; use the arithmetic-mean cluster frequency |

The ten grain orientations $(\varphi_1,\Phi,\varphi_2)$, in degrees and in spatial order, are $(359.6,99.9,146.4)$, $(167.4,116.6,342.2)$, $(37.2,79.9,195.1)$, $(95.9,79.7,130.4)$, $(274.0,64.8,46.0)$, $(49.3,102.1,97.3)$, $(89.2,98.0,58.3)$, $(18.3,72.8,349.8)$, $(204.3,83.1,116.8)$ and $(17.9,52.6,33.5)$. Report the dimensionless ratio as a decimal, with an absolute acceptance tolerance of $10^{-5}$.

## Output Format Requirements

Emit one finite decimal inside `<final_answer>...</final_answer>` and a concise explanation inside `<reasoning>...</reasoning>`. Explain the treatment of exterior inertia and modal multiplicity, and how the finite record determines the reported ratio; intermediate numerical diagnostics are not required.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

step_01_crystal_acoustic_tensors

Goal
----
The one-dimensional displacement field retains three coupled components. Each hexagonal grain retains its full anisotropic stiffness. Crystal axes in the specimen frame are the columns of the active rotation $R=Z(\varphi_1)X(\Phi)Z(\varphi_2)$, with right-handed rotations and angles in degrees. The acoustic tensor $A$ is defined by $\sigma_{i1}=A_{ik}\partial_1 u_k$. The engineering Voigt ordering is (11, 22, 33, 23, 13, 12); the sixfold axis is the third crystal axis. The independent constants are c11, c33, c12, c13 and c44; c22=c11, c23=c13, c55=c44 and c66=(c11-c12)/2.

```python
def crystal_acoustic_tensors(euler_angles, constants) -> np.ndarray:
    r"""The one-dimensional displacement field retains three coupled components. Each hexagonal grain retains its full anisotropic stiffness. Crystal axes in the specimen frame are the columns of the active rotation $R=Z(\varphi_1)X(\Phi)Z(\varphi_2)$, with right-handed rotations and angles in degrees. The acoustic tensor $A$ is defined by $\sigma_{i1}=A_{ik}\partial_1 u_k$. The engineering Voigt ordering is (11, 22, 33, 23, 13, 12); the sixfold axis is the third crystal axis. The independent constants are c11, c33, c12, c13 and c44; c22=c11, c23=c13, c55=c44 and c66=(c11-c12)/2.

    Parameters
    ----------
    euler_angles
        Finite array (n_grains, 3) of Euler angles in degrees.
    constants
        Mapping c11, c33, c12, c13, c44 in pascal, defining a positive definite hexagonal stiffness.

    Returns
    -------
    np.ndarray, acoustic tensors of shape (n_grains, 3, 3), in pascal.

    Raises
    ------
    ValueError
        If input dimensions or stated physical constraints are violated.
    """
    return
```

### Step 2

step_02_buffered_stiffness

Goal
----
The specimen occupies the first n_specimen voxels of an odd periodic grid, with three displacement components stored consecutively per voxel. A forward voxel difference defines strain and its adjoint defines the positive stiffness operator. Buffer voxels have zero density and acoustic tensor buffer_ratio times diag(mean_A11, mean_Atransverse, mean_Atransverse), where the means are voxel-weighted specimen averages and mean_Atransverse is half the sum of the transverse diagonal means. Construct the real stiffness matrix of this Fourier discretisation and its diagonal inertia coefficients. Neither matrix includes a voxel-volume factor.

```python
def buffered_stiffness(acoustic, grain_voxels, density, length, n_buffer, buffer_ratio) -> dict:
    r"""The specimen occupies the first n_specimen voxels of an odd periodic grid, with three displacement components stored consecutively per voxel. A forward voxel difference defines strain and its adjoint defines the positive stiffness operator. Buffer voxels have zero density and acoustic tensor buffer_ratio times diag(mean_A11, mean_Atransverse, mean_Atransverse), where the means are voxel-weighted specimen averages and mean_Atransverse is half the sum of the transverse diagonal means. Construct the real stiffness matrix of this Fourier discretisation and its diagonal inertia coefficients. Neither matrix includes a voxel-volume factor.

    Parameters
    ----------
    acoustic
        Finite symmetric positive definite tensors (n_grains, 3, 3), in pascal.
    grain_voxels
        Positive integer array (n_grains,), in spatial order.
    density
        Positive specimen density in kg per cubic metre.
    length
        Positive specimen length in metre.
    n_buffer
        Positive integer buffer voxel count; total grid length is odd.
    buffer_ratio
        Positive buffer stiffness multiplier.

    Returns
    -------
    dict, keys stiffness (3*n_total, 3*n_total), mass (3*n_total,), n_specimen and voxel_size.

    Raises
    ------
    ValueError
        If input dimensions or stated physical constraints are violated.
    """
    return
```

### Step 3

step_03_elastic_modes

Goal
----
Find all elastic natural modes of a specimen with positive specimen inertia and exactly zero buffer inertia. Buffer displacements are unconstrained and satisfy instantaneous mechanical equilibrium. Elastic displacement is measured relative to the specimen centre of mass; exclude the three rigid translations. Return specimen mode columns normalised in the specimen mass inner product, with positive frequencies in ascending order. Signs and orthonormal bases within repeated eigenspaces are unrestricted. The condensed stiffness maps specimen displacements to specimen forces when the buffer is in equilibrium.

```python
def elastic_modes(stiffness, mass, n_specimen) -> dict:
    r"""Find all elastic natural modes of a specimen with positive specimen inertia and exactly zero buffer inertia. Buffer displacements are unconstrained and satisfy instantaneous mechanical equilibrium. Elastic displacement is measured relative to the specimen centre of mass; exclude the three rigid translations. Return specimen mode columns normalised in the specimen mass inner product, with positive frequencies in ascending order. Signs and orthonormal bases within repeated eigenspaces are unrestricted. The condensed stiffness maps specimen displacements to specimen forces when the buffer is in equilibrium.

    Parameters
    ----------
    stiffness
        Finite real symmetric stiffness matrix, ordered voxel then component.
    mass
        Diagonal inertia vector, positive on specimen entries and exactly zero on buffer entries.
    n_specimen
        Number of specimen voxels, at least two and smaller than the total.

    Returns
    -------
    dict, keys frequencies (3*n_specimen-3,) in hertz, modes (3*n_specimen, 3*n_specimen-3) and condensed_stiffness (3*n_specimen, 3*n_specimen).

    Raises
    ------
    ValueError
        If input dimensions or stated physical constraints are violated.
    """
    return
```

### Step 4

step_04_modal_residues

Goal
----
Determine the vector receptor displacement residue for unit scalar traction at the source in each distinct elastic eigenspace. Normalise the prescribed traction direction and represent surface traction as a voxel force density. Mode columns contain three consecutive components per specimen voxel and are mass-normalised. Starting at the lowest unused frequency, a cluster contains consecutive frequencies no greater than (1 + cluster_rtol) times that first frequency; represent it by the arithmetic-mean frequency. Residues must be invariant under sign changes and orthogonal basis changes within repeated eigenspaces.

```python
def modal_residues(frequencies, modes, source, receptor, direction, voxel_size, cluster_rtol) -> dict:
    r"""Determine the vector receptor displacement residue for unit scalar traction at the source in each distinct elastic eigenspace. Normalise the prescribed traction direction and represent surface traction as a voxel force density. Mode columns contain three consecutive components per specimen voxel and are mass-normalised. Starting at the lowest unused frequency, a cluster contains consecutive frequencies no greater than (1 + cluster_rtol) times that first frequency; represent it by the arithmetic-mean frequency. Residues must be invariant under sign changes and orthogonal basis changes within repeated eigenspaces.

    Parameters
    ----------
    frequencies
        Finite positive sorted frequencies in hertz.
    modes
        Mass-normalised specimen mode columns.
    source
        Integer source voxel index inside the specimen.
    receptor
        Integer receptor voxel index inside the specimen.
    direction
        Finite nonzero vector (3,), normalised internally.
    voxel_size
        Positive voxel size in metre.
    cluster_rtol
        Relative clustering threshold in [0, 1), measured from the first frequency of each cluster.

    Returns
    -------
    dict, keys frequencies (n_clusters,) in hertz, residues (n_clusters, 3) and integer sizes (n_clusters,).

    Raises
    ------
    ValueError
        If input dimensions or stated physical constraints are violated.
    """
    return
```

### Step 5

step_05_modal_impact_record

Goal
----
Compute each elastic eigenspace's receptor history under the difference of two Gaussian traction pulses of common amplitude and width, centred at pulse_centre and pulse_centre + pulse_width. Use average-acceleration Newmark integration with beta=1/4 and gamma=1/2, zero initial displacement, velocity and acceleration, and traction evaluated at each step's end time. Include the initial zero sample. Each residue maps its unit-inertia oscillator response into the three receptor components. Retain the finite-record phase of every cluster.

```python
def modal_impact_record(frequencies, residues, dt, n_steps, pulse_amplitude, pulse_width, pulse_centre) -> np.ndarray:
    r"""Compute each elastic eigenspace's receptor history under the difference of two Gaussian traction pulses of common amplitude and width, centred at pulse_centre and pulse_centre + pulse_width. Use average-acceleration Newmark integration with beta=1/4 and gamma=1/2, zero initial displacement, velocity and acceleration, and traction evaluated at each step's end time. Include the initial zero sample. Each residue maps its unit-inertia oscillator response into the three receptor components. Retain the finite-record phase of every cluster.

    Parameters
    ----------
    frequencies
        Finite positive sorted frequencies in hertz.
    residues
        Finite vector residues (n_clusters, 3) for unit scalar traction.
    dt
        Positive time increment in second.
    n_steps
        Positive integer number of time increments.
    pulse_amplitude
        Finite scalar traction amplitude in pascal.
    pulse_width
        Positive Gaussian width in second.
    pulse_centre
        Finite first-pulse centre in second.

    Returns
    -------
    np.ndarray, receptor histories of shape (n_steps+1, n_clusters, 3), in metre.

    Raises
    ------
    ValueError
        If input dimensions or stated physical constraints are violated.
    """
    return
```

### Step 6

step_06_finite_record_coherence

Goal
----
Measure interference between elastic eigenspaces in a frequency band. The numerator is the band sum of squared vector Fourier amplitudes of the total receptor displacement. The denominator is the sum of the corresponding powers of the individual eigenspace histories. Use the complete record, NumPy's default Fourier normalisation, no window and no zero padding, retaining non-negative bin centres in the closed band. The powers are unnormalised DFT sums. Their dimensionless ratio can exceed one.

```python
def finite_record_coherence(history, dt, frequency_band) -> dict:
    r"""Measure interference between elastic eigenspaces in a frequency band. The numerator is the band sum of squared vector Fourier amplitudes of the total receptor displacement. The denominator is the sum of the corresponding powers of the individual eigenspace histories. Use the complete record, NumPy's default Fourier normalisation, no window and no zero padding, retaining non-negative bin centres in the closed band. The powers are unnormalised DFT sums. Their dimensionless ratio can exceed one.

    Parameters
    ----------
    history
        Finite displacement array (n_samples, n_clusters, 3), including the initial sample.
    dt
        Positive time increment in second.
    frequency_band
        Pair (low, high) in hertz, with 0 < low < high <= Nyquist and at least one retained bin.

    Returns
    -------
    dict, keys ratio, coherent_power, incoherent_power and integer n_bins.

    Raises
    ------
    ValueError
        If input dimensions or stated physical constraints are violated.
    """
    return
```

### Step 7

step_07_polycrystal_coherence

Goal
----
Compute the finite-record interference ratio from the grain orientations and physical configuration. Retain the rotated anisotropic acoustic tensors, the odd periodic grid, the massless compliant buffer, every elastic eigenspace and the prescribed finite-duration pulse experiment. The source is voxel zero. Compare the total receptor vector displacement power with the sum of isolated-eigenspace powers in the band. Recompute every intermediate quantity from the top-level inputs.

```python
def polycrystal_coherence(euler_angles, constants, grain_voxels, density, length, n_buffer, buffer_ratio, direction, receptor, dt, n_steps, pulse_amplitude, pulse_width, pulse_centre, frequency_band, cluster_rtol) -> dict:
    r"""Compute the finite-record interference ratio from the grain orientations and physical configuration. Retain the rotated anisotropic acoustic tensors, the odd periodic grid, the massless compliant buffer, every elastic eigenspace and the prescribed finite-duration pulse experiment. The source is voxel zero. Compare the total receptor vector displacement power with the sum of isolated-eigenspace powers in the band. Recompute every intermediate quantity from the top-level inputs.

    Parameters
    ----------
    euler_angles
        Finite array (n_grains, 3) of Euler angles in degrees.
    constants
        Mapping c11, c33, c12, c13, c44 in pascal, defining a positive definite hexagonal stiffness.
    grain_voxels
        Positive integer array (n_grains,), in spatial order.
    density
        Positive specimen density in kg per cubic metre.
    length
        Positive specimen length in metre.
    n_buffer
        Positive integer buffer voxel count; total grid length is odd.
    buffer_ratio
        Positive buffer stiffness multiplier.
    direction
        Finite nonzero vector (3,), normalised internally.
    receptor
        Integer receptor voxel index inside the specimen.
    dt
        Positive time increment in second.
    n_steps
        Positive integer number of time increments.
    pulse_amplitude
        Finite scalar traction amplitude in pascal.
    pulse_width
        Positive Gaussian width in second.
    pulse_centre
        Finite first-pulse centre in second.
    frequency_band
        Pair (low, high) in hertz, with 0 < low < high <= Nyquist and at least one retained bin.
    cluster_rtol
        Relative clustering threshold in [0, 1), measured from the first frequency of each cluster.

    Returns
    -------
    dict, keys ratio, coherent_power, incoherent_power and integer n_bins.

    Raises
    ------
    ValueError
        If input dimensions or stated physical constraints are violated.
    """
    return
```
