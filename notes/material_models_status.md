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
- Small-strain J2 plasticity material-point update:
  `J2Plasticity`, `J2PlasticityState`, `make_j2_plasticity_state`, and
  `j2_return_mapping`.
  - Uses 6-component Voigt strain/stress with engineering shear components.
  - Supports isotropic linear hardening and JAX pytree/JIT material-point use.
  - Covered by elastic, hydrostatic, pure-shear return, unload, and JIT tests.
- J2 FE integration entry points:
  `J2PlasticityQuadratureState`, `make_j2_quadrature_state`,
  `j2_plasticity_residual_form`, `update_j2_quadrature_state`, and
  `solve_j2_plasticity_load_steps`.
  - Stores history as element/quadrature arrays with shapes `(n_elem, n_q, 6)`
    and `(n_elem, n_q)`.
  - Assembles a frozen-state small-strain internal-force residual.
  - Provides an explicit post-convergence state update helper and a basic
    load-step wrapper with trial/commit behavior.
  - Exposes `evaluate_j2_quadrature_strain` and
    `evaluate_j2_quadrature_stress` for diagnostics and verification.
  - Exposes `make_j2_cell_data`, `make_j2_point_and_cell_data`, and
    `write_j2_vtu` for element-averaged VTU cell data.
  - Covered by elastic residual equality against linear elasticity and plastic
    quadrature-state commit tests, including displacement-controlled stepping,
    material-point reference matching for uniaxial extension, and elastic
    unload response after committed plasticity.
- J2 visualization tutorial:
  `tutorials/nonlinear/j2_uniaxial_tension.py` writes displacement point data,
  element-averaged plastic strain/stress VTU cell data, and a CSV load history.
- Basic structural damping/spring/dashpot helpers for lumped DOFs. These are
  not continuum viscoelastic material models.

Not implemented:

- Continuum viscoelasticity with internal variables, such as Maxwell,
  generalized Maxwell, Kelvin-Voigt, or standard linear solid models.
- Continuum damage models.
- Consistent algorithmic tangents for history-dependent material updates.
- Production-grade quadrature-state lifecycle for plasticity, viscoelasticity,
  or damage. J2 now has FE-facing frozen-state residual/update helpers and a
  basic load-step wrapper, but not full restart/output or general analysis
  orchestration.

Important limits:

- The current Neo-Hookean model is compressible and Total-Lagrangian. It is not
  a mixed nearly-incompressible formulation.
- Tangents for the current hyperelastic residual are obtained by JAX AD. There
  is no hand-coded constitutive tangent layer yet.
- History-dependent FE materials still need restart/output support and a broader
  analysis lifecycle before J2 is promoted to production FE material
  integration.
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

3. Harden J2 FE integration.
   - Add restart/output support for committed quadrature histories.
   - Exercise the load-step wrapper on force-controlled and mixed BC examples.
   - Add consistent algorithmic tangents or a clearly documented AD/tangent
     strategy.
   - Add patch tests beyond the current one-element homogeneous extension
     checks.

4. Defer damage until the state/update infrastructure is in place.
   - Start with a clearly labeled local scalar damage demo only if needed.
   - Do not present local softening as mesh-objective without regularization.

Main checks for the current implemented nonlinear material path:

- `PYTHONPATH=src pytest -q src/tests/test_neo_hookean.py src/tests/test_weakform_nonlinear.py`
- `PYTHONPATH=src pytest -q src/tests/test_kernel_assembly.py -k neo_hookean`
- `PYTHONPATH=src pytest -q src/tests/test_j2_plasticity.py`
- `PYTHONPATH=src pytest -q src/tests/test_j2_fe_integration.py`
- `PYTHONPATH=src python tutorials/nonlinear/j2_uniaxial_tension.py --steps 2 --nx 2`
- `PYTHONPATH=src JAX_PLATFORMS=cpu python tutorials/nonlinear/neo_hookean_cantilever.py --nx 2 --ny 1 --nz 1 --nstep 2 --no-output --linear-solver spsolve`
