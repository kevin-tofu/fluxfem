# Material Models Status

Current status after the structural element and coupling work on `main`.

Implemented:

- Small-strain isotropic linear elasticity through `isotropic_3d_D(...)` and
  `linear_elasticity_form`.
- Compressible Neo-Hookean hyperelasticity helpers:
  `deformation_gradient`, `right_cauchy_green`, `green_lagrange_strain`,
  `pk2_neo_hookean`, and `neo_hookean_residual_form`.
- Neo-Hookean residual/Jacobian assembly through JAX AD.
- Neo-Hookean tutorials and nonlinear solve examples:
  `tutorials/nonlinear/neo_hookean_cantilever.py` and
  `tutorials/nonlinear/linear_material_geo_nonlinear.py`.
- Basic structural damping/spring/dashpot helpers for lumped DOFs. These are
  not continuum viscoelastic material models.

Not implemented:

- J2 plasticity.
- Continuum viscoelasticity with internal variables, such as Maxwell,
  generalized Maxwell, Kelvin-Voigt, or standard linear solid models.
- Continuum damage models.
- Consistent algorithmic tangents for history-dependent material updates.
- Quadrature-point material state storage/update APIs for production plasticity,
  viscoelasticity, or damage.

Important limits:

- The current Neo-Hookean model is compressible and Total-Lagrangian. It is not
  a mixed nearly-incompressible formulation.
- Tangents for the current hyperelastic residual are obtained by JAX AD. There
  is no hand-coded constitutive tangent layer yet.
- History-dependent materials need an explicit design for quadrature-point
  state, load stepping, state commit/rollback, and restart/output. That design
  should happen before adding J2 plasticity or damage.
- Damage with softening needs regularization or nonlocal/gradient treatment to
  avoid mesh-dependent localization; a simple local scalar damage model should
  be marked as a demonstration only.

Recommended next steps:

1. Strengthen hyperelastic verification.
   - Add uniaxial and simple-shear checks against closed-form Neo-Hookean
     stresses.
   - Clarify compressible versus nearly-incompressible limits.
   - Add a small nonlinear tutorial/test that checks residual monotonicity or
     energy consistency.

2. Add a small-strain linear viscoelastic prototype.
   - Start with a 1D or scalar generalized Maxwell/Kelvin-Voigt tutorial.
   - Use this to design internal state update and time stepping before moving
     to full 3D tensor models.

3. Design J2 plasticity before implementation.
   - Required pieces: return mapping, plastic strain state, equivalent plastic
     strain, yield stress/hardening law, consistent tangent, load stepping, and
     state commit/rollback.
   - Keep a verification path from the start: uniaxial tension, unload/reload,
     and patch tests.

4. Defer damage until the state/update infrastructure is in place.
   - Start with a clearly labeled local scalar damage demo only if needed.
   - Do not present local softening as mesh-objective without regularization.

Main checks for the current implemented nonlinear material path:

- `PYTHONPATH=src pytest -q src/tests/test_neo_hookean.py src/tests/test_weakform_nonlinear.py`
- `PYTHONPATH=src pytest -q src/tests/test_kernel_assembly.py -k neo_hookean`
- `PYTHONPATH=src JAX_PLATFORMS=cpu python tutorials/nonlinear/neo_hookean_cantilever.py --nx 2 --ny 1 --nz 1 --nstep 2 --no-output --linear-solver spsolve`
