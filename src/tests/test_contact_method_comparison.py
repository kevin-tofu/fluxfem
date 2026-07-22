import numpy as np

import fluxfem as ff


def test_compare_primal_solutions_separates_incomparable_vectors():
    rows = ff.compare_primal_solutions(
        {
            "reference": np.array([1.0, 2.0]),
            "same_layout": np.array([1.0, 3.0]),
            "different_layout": np.array([1.0, 2.0, 3.0]),
            "missing": None,
        },
        reference="reference",
    )

    by_name = {row.method: row for row in rows}
    assert by_name["reference"].comparable
    assert by_name["same_layout"].comparable
    np.testing.assert_allclose(by_name["same_layout"].abs_l2, 1.0)
    assert not by_name["different_layout"].comparable
    assert "shape mismatch" in str(by_name["different_layout"].reason)
    assert not by_name["missing"].comparable
    assert by_name["missing"].reason == "solution is unavailable"


def test_contact_method_metric_is_flat_serializable_row():
    metric = ff.ContactMethodMetric(
        method="mortar",
        enforcement="mortar",
        formulation="multiplier",
        ok=True,
        elapsed_seconds=0.1,
        operator_shape=(2, 3),
        operator_nnz=4,
    )

    row = metric.to_dict()
    assert row["method"] == "mortar"
    assert row["operator_shape"] == (2, 3)
    assert row["operator_nnz"] == 4

