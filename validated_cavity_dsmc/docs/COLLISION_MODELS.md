# Collision-model implementation notes

## Shared physics

For every trial, the code evaluates the VHS rate coefficient

\[
(\sigma g)_{ij}=
\frac{\pi d_{\mathrm{ref}}^2}{\Gamma(2.5-\omega)}
\left(\frac{2kT_{\mathrm{ref}}}{m_r g_{ij}^2}\right)^{\omega-1/2}g_{ij},
\qquad m_r=m/2.
\]

An accepted equal-mass pair is scattered isotropically in the center-of-mass
frame. This preserves the pair momentum and kinetic energy to floating-point
roundoff.

## NTC and NTC-PreScan

For cell occupancy `N`, volume `V`, particle weight `F_N`, and majorant
`(sigma g)_max`, NTC selects

\[
N_{\rm sel}=\operatorname{floor}\left[
\frac{N(N-1)}{2}\frac{F_N(\sigma g)_{\max}\Delta t}{V}+R
\right]
\]

candidate pairs and accepts each with `(sigma g)_ij/(sigma g)_max`.
`ntc-prescan` samples up to 32 random cell-local pairs before selection and
raises the persistent majorant with a 5% guard. A post-trial update prevents a
later step from retaining a stale smaller majorant.

## SBT

After a cell-local shuffle, particle `i` is paired with one uniformly sampled
particle from `i+1,...,N-1`. The Bernoulli probability is

\[
W_i=(N-i-1)\frac{F_N(\sigma g)_{ij}\Delta t}{V},
\]

using zero-based `i`. There are `N-1` trials. This reproduces the selection
structure in the supplied DSMC2 Fortran `COLLSBT` routine.

## GBT

GBT performs `N_s < N-1` trials. Its correction is

\[
C=\frac{N(N-1)}{N_s(2N-N_s-1)},\qquad
W_i=C(N-i-1)\frac{F_N(\sigma g)_{ij}\Delta t}{V}.
\]

When `N_s >= N-1`, the code falls back to SBT rather than applying a singular
or redundant generalized formula.

## SSBT and SGBT

SSBT selects each first particle once and samples its second partner from the
full cell excluding itself. The trial multiplier is `(N-1)/2`, which accounts
for the two orientations of each unordered pair.

SGBT selects only `N_s` first particles. The multiplier is

\[
\frac{N(N-1)}{2N_s}.
\]

The implementation rejects duplicate unordered pairs, including reversed
duplicates. This addresses the failure mode noted in the supplied development
codes.

## TAS variants

TAS creates local adaptive subcells with a target number of particles and
performs an unstaggered plus half-subcell-shifted pass. Edge cells in the
staggered pass use half width and corner cells use quarter area, matching the
environmental-subcell correction in the supplied Fortran code. Each pass
represents half a collision step. Probabilities use the actual subcell volume. This is why
the TAS Bernoulli probability limit can be smaller than the parent SBT/GBT
limit.

The implementation is intentionally explicit and auditable. It is not claimed
to reproduce every optimization in the legacy Fortran storage layout.
