import numpy as np

import fluxfem as ff


def test_named_space_stores_symbol_and_space():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    named = ff.NamedSpace("U", space)
    assert named.name == "U"
    assert named.space is space


def test_build_form_contexts_pair_separates_test_and_trial():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    test_space = ff.make_hex_space(mesh, dim=1, intorder=2)
    trial_space = ff.make_hex_space(mesh, dim=1, intorder=2)

    ctx = ff.build_form_contexts_pair(test_space, trial_space, test_name="V", trial_name="U")

    assert ctx.default_space == "default"
    assert ctx.spaces is not None
    assert "V" in ctx.spaces
    assert "U" in ctx.spaces
    assert ctx.test is ctx.spaces["V"].test
    assert ctx.trial is ctx.spaces["U"].trial
    assert ctx.spaces["default"].test is ctx.test
    assert ctx.spaces["default"].trial is ctx.trial
    assert np.asarray(ctx.x_q).shape[0] == test_space.elem_dofs.shape[0]
