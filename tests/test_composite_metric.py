"""
Tests for CompositeMetric and oscillation index (from test_sgd_smo.py).
"""

from harness.metrics import CompositeMetric, ast_distance


def test_ast_distance_same():
    code = "def f():\n    return 1"
    assert ast_distance(code, code) == 0.0


def test_ast_distance_different():
    initial = "def f():\n    return 1"
    different = "def g():\n    return 2"
    assert ast_distance(initial, different) > 0.0


def test_composite_metric():
    initial = "def f():\n    return 1"
    same = "def f():\n    return 1"
    different = "def g():\n    return 2"

    metric = CompositeMetric(initial_code=initial, lambda_ast=0.2)

    r1 = metric.compute(same, test_loss=0.5, step=0)
    assert r1["L_test"] == 0.5
    assert r1["L_ast"] == 0.0
    assert r1["L_total"] == 0.5

    r2 = metric.compute(different, test_loss=0.5, step=1)
    assert r2["L_test"] == 0.5
    assert r2["L_ast"] > 0.0
    assert r2["L_total"] > 0.5


def test_oscillation_index():
    initial = "def f():\n    return 1"
    code_a = "def f():\n    return 1"
    code_b = "def f():\n    return 2"

    # Oscillation: A → B → A → B
    metric_osc = CompositeMetric(initial_code=initial, lambda_ast=0.2)
    metric_osc.compute(code_a, 0.5, 0)
    metric_osc.compute(code_b, 0.3, 1)
    metric_osc.compute(code_a, 0.5, 2)
    metric_osc.compute(code_b, 0.3, 3)

    oi_osc = metric_osc.oscillation_index()
    assert oi_osc > 0

    # Monotonic convergence: A → A' → A''
    code_a1 = "def f():\n    return 1 + 0"
    code_a2 = "def f():\n    return 1 + 0 + 0"

    metric_conv = CompositeMetric(initial_code=initial, lambda_ast=0.2)
    metric_conv.compute(code_a, 0.5, 0)
    metric_conv.compute(code_a1, 0.3, 1)
    metric_conv.compute(code_a2, 0.1, 2)

    oi_conv = metric_conv.oscillation_index()
    assert oi_conv <= oi_osc
