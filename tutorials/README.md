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
