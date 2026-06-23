"""
Tests for SGD evaluator (from test_sgd_smo.py).
"""

from environments.evaluator import SGDEvaluator


def test_blind_box_pass_all():
    test_cases = [
        ("1+1", 2), ("2*3", 6), ("10-5", 5),
        ("2+3*4", 14), ("(2+3)*4", 20),
    ]
    evaluator = SGDEvaluator(test_cases=test_cases, entry_point="SimpleInterpreter", seed=42)

    code_passes_all = """
class SimpleInterpreter:
    def evaluate(self, expr):
        return eval(expr)
"""
    result = evaluator.evaluate(code_passes_all)
    assert result["passed_tests"] == 5
    assert result["error_msg"] == ""


def test_blind_box_fail_returns_single_error():
    test_cases = [
        ("1+1", 2), ("2*3", 6), ("10-5", 5),
        ("2+3*4", 14), ("(2+3)*4", 20),
    ]
    evaluator = SGDEvaluator(test_cases=test_cases, entry_point="SimpleInterpreter", seed=42)

    code_fails_all = """
class SimpleInterpreter:
    def evaluate(self, expr):
        return -1
"""
    result = evaluator.evaluate(code_fails_all)
    assert result["passed_tests"] == 0
    assert result["error_msg"] != ""
    assert "mismatch" in result["error_msg"].lower() or "expected" in result["error_msg"].lower()
    assert result["error_msg"].count("mismatch") <= 1


def test_randomness_across_seeds():
    test_cases = [("1+1", 2), ("2*3", 6), ("10-5", 5), ("100+200", 300)]
    bad_code = """
class SimpleInterpreter:
    def evaluate(self, expr):
        return -1
    """
    e1 = SGDEvaluator(test_cases, "SimpleInterpreter", seed=1)
    e2 = SGDEvaluator(test_cases, "SimpleInterpreter", seed=2)

    r1 = e1.evaluate(bad_code)
    r2 = e2.evaluate(bad_code)

    assert r1["passed_tests"] == 0
    assert r2["passed_tests"] == 0
