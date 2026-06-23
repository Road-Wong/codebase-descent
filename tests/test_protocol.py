"""
Tests for SEARCH/REPLACE protocol and CodeEnv (from root test_protocol.py).
"""

import re
from core.protocol import parse_action, apply_patch, make_action, validate_action, ProtocolError
from core.env_base import CodeEnv
from core.types import Action, ActionFormat


class TestProtocolParser:
    def test_valid_single_block(self):
        action = """<<<< SEARCH
def foo():
    return 1
====
def foo():
    return 2
>>>>"""
        blocks = parse_action(action)
        assert len(blocks) == 1
        assert "return 1" in blocks[0].search
        assert "return 2" in blocks[0].replace

    def test_valid_multi_block(self):
        action = """<<<< SEARCH
a = 1
====
a = 2
>>>>
<<<< SEARCH
b = 3
====
b = 4
>>>>"""
        blocks = parse_action(action)
        assert len(blocks) == 2

    def test_invalid_no_markers(self):
        try:
            parse_action("just some random text")
            assert False, "Should have raised"
        except ProtocolError:
            pass

    def test_invalid_unterminated(self):
        try:
            parse_action("<<<< SEARCH\nfoo\n====\nbar")
            assert False, "Should have raised"
        except ProtocolError:
            pass


class TestProtocolApplier:
    def test_exact_match(self):
        code = """def greet():
    print("hello")
    return 42"""
        action = make_action('    print("hello")', '    print("world")')
        result = apply_patch(code, action)
        assert result.success is True
        assert result.applied == 1
        assert 'print("world")' in result.code
        assert 'print("hello")' not in result.code

    def test_fuzzy_match(self):
        code = "def foo():\n    return 1   \n"
        action = make_action("    return 1", "    return 2")
        result = apply_patch(code, action)
        assert result.success is True
        assert "return 2" in result.code

    def test_search_not_found(self):
        code = "def greet():\n    print('hello')\n    return 42"
        action = make_action("nonexistent_code()", "replacement()")
        result = apply_patch(code, action)
        assert result.success is False
        assert result.applied == 0
        assert len(result.errors) == 1

    def test_empty_search(self):
        code = "def greet():\n    print('hello')\n    return 42"
        action = make_action("", "# header\n")
        result = apply_patch(code, action)
        assert result.success is True
        assert result.code.startswith("# header\n")


class TestCodeEnv:
    def test_load(self):
        env = CodeEnv(asset_name="original_code", obfuscation_level=0, max_steps=20)
        assert "SimpleInterpreter" in env.ground_truth
        assert "tokenize" in env.ground_truth
        assert "_eval_expr" in env.ground_truth

    def test_reset(self):
        env = CodeEnv(asset_name="original_code", obfuscation_level=0, max_steps=20)
        obs = env.reset()
        assert obs.loss == 0.0
        assert obs.passed_tests == 26
        assert obs.total_tests == 26
        assert obs.step == 0
        assert "SimpleInterpreter" in obs.current_code

    def test_patch(self):
        env = CodeEnv(asset_name="original_code", obfuscation_level=1, max_steps=20)
        obs = env.reset()
        initial_code = obs.current_code

        if "SimpleInterpreter" not in initial_code:
            match = re.search(r"class (\w+):", initial_code)
            if match:
                old_name = match.group(1)
                action_text = make_action(f"class {old_name}:", "class SimpleInterpreter:")
                action = Action(patch=action_text, format=ActionFormat.SEARCH_REPLACE)
                obs2, reward, done, truncated, info = env.step(action)
                assert obs2.step == 1

    def test_full_recovery(self):
        env = CodeEnv(asset_name="original_code", obfuscation_level=0, max_steps=20)
        broken_code = env.ground_truth.replace("left = left + right", "left = left - right", 1)
        obs = env.reset(initial_code=broken_code)
        assert obs.loss > 0

        action_text = make_action("left = left - right", "left = left + right")
        action = Action(patch=action_text, format=ActionFormat.SEARCH_REPLACE)
        obs2, reward, done, truncated, info = env.step(action)

        assert done is True
        assert obs2.loss == 0.0
        assert obs2.passed_tests == 26

    def test_bad_patch_no_crash(self):
        env = CodeEnv(asset_name="original_code", obfuscation_level=0, max_steps=20)
        obs = env.reset()

        action = Action(
            patch="<<<< SEARCH\nnonexistent\n====\nreplacement\n>>>>",
            format=ActionFormat.SEARCH_REPLACE,
        )
        obs2, reward, done, truncated, info = env.step(action)

        assert done is False
        assert obs2.current_code == obs.current_code
        assert "patch_error" in info or obs2.loss == obs.loss
