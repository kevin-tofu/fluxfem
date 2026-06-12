# Contact ROM API guide

This note summarizes the current user-facing path for Craig-Bampton reduced
contact dynamics. It is intentionally API-oriented; implementation history and
validation details live in `notes/craig_bampton_rom.md` and
`notes/contact_rom_api_audit.md`.

## Recommended workflow

1. Build full-space structural operators.

   ```python
   K = ...
   M = ...
   C = 0.02 * M
   ```

2. Retain contact-surface DOFs, then modalize the remaining interior DOFs.

   ```python
   contact_nodes = np.unique(np.concatenate([slave.conn.reshape(-1), master.conn.reshape(-1)]))
   retained = ff.vector_dofs_from_nodes(contact_nodes, dim)
   cb = ff.make_craig_bampton_basis(K, M, retained_dofs=retained, n_modes=n_modes)
   ```

   For active or moving contact, retain every surface that may enter the contact
   candidate set. If the active patch moves outside the retained band, the ROM
   basis no longer gives the contact residual enough physical coordinates.

   For larger internal blocks, the static constraint-mode solve can be swapped:

   ```python
   cb = ff.make_craig_bampton_basis(
       K,
       M,
       retained_dofs=retained,
       n_modes=n_modes,
       constraint_solver="cg",
       modal_solver="subspace",
       modal_linear_solver="cg",
       cg_tol=1e-8,
       cg_maxiter=200,
   )
   ```

   The subspace modal solver is an in-tree block inverse iteration path. For
   SciPy-backed modal extraction, use `modal_solver="eigsh"` when SciPy is
   installed. For project-specific sparse backends, pass a custom
   `modal_solver` callable.

   If `K` and `M` are `FluxSparseMatrix` or SciPy sparse matrices, the CB block
   partition keeps internal blocks sparse for sparse-aware solver paths:

   ```python
   cb = ff.make_craig_bampton_basis(
       K_sparse,
       M_sparse,
       retained_dofs=retained,
       n_modes=n_modes,
       constraint_solver="spsolve",
       modal_solver="eigsh",
   )
   ```

3. Choose a search manager.

   Node-to-surface:

   ```python
   search_manager = ff.make_node_surface_contact_search_manager(
       slave,
       master,
       dim=dim,
       n_total_nodes=n_nodes,
       search_radius=search_radius,
       skin=skin,
       penalty=penalty,
       normal=normal,
       cell_size=cell_size,
   )
   ```

   Surface-quadrature:

   ```python
   search_manager = ff.make_surface_quadrature_contact_search_manager(
       slave,
       master,
       dim=dim,
       n_total_nodes=n_nodes,
       search_radius=search_radius,
       skin=skin,
       penalty=penalty,
       normal=normal,
       quadrature_rule="vertices",
       cell_size=cell_size,
   )
   ```

   The manager owns broad-phase AABB state, neighbor-list refresh policy, and
   frozen exact pairing caches.

4. Add friction history only when needed.

   ```python
   friction_manager = ff.TangentialPenaltyFrictionManager(
       mu=mu,
       tangential_penalty=tangential_penalty,
       previous_displacement=cb.expand(q0),
   )
   ```

   Friction history is explicit. A solve uses a frozen snapshot; after
   convergence, `ReducedContactDynamics` advances the stored history by default.

5. Solve through the facade.

   ```python
   dynamics = ff.ReducedContactDynamics(
       cb=cb,
       stiffness=K,
       mass=M,
       damping=C,
       search_manager=search_manager,
       friction_manager=friction_manager,
   )

   next_state, info = dynamics.active_newmark_step(
       f_full,
       state,
       ff.NewmarkConfig(dt=dt, tol=tol, atol=atol, maxiter=maxiter),
       max_active_updates=max_active_updates,
   )
   ```

   Pass full-space forces by default. Use `force_is_reduced=True` only if the
   caller already projected the force with `cb.project_vector(...)`.

## Interface roles

- `CraigBamptonBasis`: owns `expand(q)`, `project_vector(f)`, and
  `project_matrix(A)`.
- `ContactSearchManagerLike`: must implement `build_contact(displacement)`.
- `FrictionManagerLike`: must implement `snapshot(contact, u)` and
  `advance(contact, u)`.
- `ReducedContactDynamics`: composes the basis, structural operators, contact
  search, optional friction history, and active Newmark solve.
- `ContactUpdateSnapshot`: freezes the active contact state for the inner
  Newton solve.
- `FrictionalContactUpdateSnapshot`: extends the frozen normal-contact snapshot
  with frozen tangential friction residuals.

## AD behavior

The reduced residual is still differentiated through full-space residuals:

```python
u = cb.expand(q)
Rr(q) = cb.basis.T @ R_full(u)
```

This keeps the contact residual and its active/frozen state visible to JAX while
avoiding a separate ROM-only contact kernel. Frozen friction residuals are
constant during an inner solve, so their Jacobian is zero until history is
advanced explicitly.

## Active contact applicability

The current API supports active contact workflows when these assumptions hold:

- The potential contact surface DOFs are retained in the CB basis.
- Search managers are refreshed when surface motion exceeds the neighbor-list
  skin threshold.
- Exact pairing and active state are frozen inside each Newton solve.
- Friction history is advanced outside the frozen inner solve.

This is suitable for small-to-medium active contact experiments, dynamic contact
steps, and ROM interface validation. Large sliding contact with a wide moving
active region will need a larger retained band, basis enrichment, or online
remeshing/reduction updates.

## Validated entry points

- `tutorials/craig_bampton_surface_quadrature_friction_active_newmark.py`
- `tutorials/craig_bampton_node_surface_friction_active_newmark.py`
- `tutorials/craig_bampton_surface_contact_reference.py`
- `src/tests/test_rom.py -k "reduced_contact_dynamics"`
- `src/tests/test_contact.py -k "independent_reference or independent_penalty_form or independent_weighted_penalty_form"`

## Practical defaults

- Start with `quadrature_rule="vertices"` for surface-quadrature contact.
- Use a positive `skin` so repeated steps can reuse search state.
- Keep `search_radius + skin` large enough to cover expected motion per step.
- Use all contact-surface DOFs as retained DOFs before tuning `n_modes`.
- Compare at least one all-mode ROM run against full-order for each new model.
