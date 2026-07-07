# Tutorials

Tutorial scripts are grouped by theme:

- `common/`: shared helpers used by tutorials.
- `contact/`: contact, mortar, Nitsche, and fixture-workpiece examples.
- `craig_bampton/`: Craig-Bampton ROM and reduced-equation examples.
- `diffusion/`: diffusion, reaction-diffusion, and heat examples.
- `dynamics/`: transient dynamics examples.
- `elasticity/`: linear elasticity examples.
- `nonlinear/`: full-order geometric/material nonlinear examples.
- `nonlinear_rom/`: nonlinear full-vs-ROM comparison examples.
- `petsc/`: PETSc backend examples.
- `remote_constraints/`: RBE3/reference-point constraint examples.
- `thermoelastic/`: thermoelastic coupling examples.

Generated files should stay under `tutorials/{theme}/results/` when a tutorial
has persistent outputs such as JSON metrics, plots, or VTU files.

## Selected Scripts

Elasticity and structural helpers:

- `elasticity/linearelastic_tensile_bar.py`: linear elasticity weak-form assembly.
- `elasticity/beam_cantilever.py`: 3D Euler-Bernoulli cantilever beam with selectable matrix format.
- `elasticity/beam_point_load.py`: beam tip force and moment loading.
- `elasticity/beam_uniform_load.py`: equivalent nodal loads for uniform beam loading.
- `elasticity/beam_cantilever_modes.py`: first bending frequency check for a beam cantilever.
- `elasticity/truss_bar_cantilever.py`: 3D truss/bar cantilever with selectable matrix format.
- `elasticity/truss_uniform_load.py`: equivalent nodal loads for uniform truss/bar loading.

Dynamics:

- `dynamics/spring_mass_dashpot.py`: single-DOF mass-spring-dashpot Newmark integration.
- `dynamics/beam_tip_spring_dashpot.py`: beam plus lumped tip spring/dashpot.
- `dynamics/beam_rayleigh_damping.py`: Rayleigh damping from modal damping targets.

Contact and constraints:

- `contact/two_body_contact_displacement_vtu.py`: two-body contact solve exported as a combined VTU.
- `contact/curved_surface_contact_vtu_demo.py`: visualization-oriented curved contact VTU sequence.
- `contact/contact_mortar_builder_methods.py`: mortar multiplier choices through explicit and builder APIs.
- `remote_constraints/remote_rbe3_spring_compliance.py`: reference-point constraint compliance example.
