"""
Order Discount Environment — Complex business logic task.

15 test cases covering: simple order, member discount, bulk discount,
threshold discount, coupon rules, edge cases, multi-item scenarios.
"""

from typing import Dict, Any

from core.env_base import AbstractEnvironment


class OrderDiscountEnvironment(AbstractEnvironment):
    """Environment for implementing an e-commerce order discount calculator."""

    def __init__(self, obfuscation_level: int = 0, max_steps: int = 20):
        ground_truth = """class OrderSystem:
    def __init__(self):
        self.items = {}
        self.member_level = 0
        self.coupon = None

    def add_item(self, item_id, price, quantity):
        if item_id in self.items:
            self.items[item_id]['quantity'] += quantity
        else:
            self.items[item_id] = {'price': price, 'quantity': quantity}

    def set_member_level(self, level):
        self.member_level = level

    def apply_coupon(self, coupon_type, value):
        self.coupon = (coupon_type, value)

    def calculate_total(self):
        subtotal = 0
        for item_id, item_data in self.items.items():
            price = item_data['price']
            quantity = item_data['quantity']
            if quantity >= 10:
                price *= 0.8
            elif quantity >= 5:
                price *= 0.9
            subtotal += price * quantity
        member_discounts = {0: 1.0, 1: 0.95, 2: 0.9, 3: 0.85}
        subtotal *= member_discounts.get(self.member_level, 1.0)
        if self.coupon:
            coupon_type, value = self.coupon
            if coupon_type == 'percentage':
                discount_rate = min(value, 30) / 100
                subtotal *= (1 - discount_rate)
            elif coupon_type == 'fixed':
                discount_amount = min(value, 50)
                subtotal -= discount_amount
        if subtotal >= 500:
            subtotal *= 0.85
        elif subtotal >= 200:
            subtotal *= 0.9
        elif subtotal >= 100:
            subtotal *= 0.95
        return max(subtotal, 0.0)
"""
        super().__init__(
            ground_truth_code=ground_truth,
            max_steps=max_steps,
            obfuscation_level=obfuscation_level,
        )
        self.test_cases = [
            {"items": [(1,50.0,1)], "member_level": 0, "coupon": None, "expected": 50.0},
            {"items": [(1,100.0,1)], "member_level": 2, "coupon": None, "expected": 90.0},
            {"items": [(1,10.0,6)], "member_level": 1, "coupon": None, "expected": 51.3},
            {"items": [(1,100.0,1)], "member_level": 0, "coupon": None, "expected": 95.0},
            {"items": [(1,100.0,2)], "member_level": 2, "coupon": ("percentage",10), "expected": 153.9},
            {"items": [(1,10.0,12)], "member_level": 0, "coupon": None, "expected": 96.0},
            {"items": [(1,80.0,1)], "member_level": 1, "coupon": ("fixed",20), "expected": 56.0},
            {"items": [(1,30.0,3),(2,40.0,2)], "member_level": 3, "coupon": ("percentage",15), "expected": 116.68375},
            {"items": [(1,100.0,1)], "member_level": 0, "coupon": ("percentage",50), "expected": 70.0},
            {"items": [(1,100.0,1)], "member_level": 0, "coupon": ("fixed",80), "expected": 50.0},
            {"items": [(1,100.0,6)], "member_level": 2, "coupon": None, "expected": 437.4},
            {"items": [(1,50.0,1)], "member_level": 3, "coupon": ("fixed",50), "expected": 0.0},
            {"items": [(1,20.0,8),(2,15.0,3)], "member_level": 1, "coupon": ("percentage",10), "expected": 153.51525},
            {"items": [(1,150.0,4)], "member_level": 3, "coupon": ("percentage",20), "expected": 367.2},
            {"items": [(1,25.0,11),(2,30.0,7),(3,20.0,4)], "member_level": 2, "coupon": ("fixed",40), "expected": 360.09},
        ]

    def _get_task_description(self) -> str:
        return (
            "Implement an OrderSystem class with:\n"
            "- __init__(): Initialize the system\n"
            "- add_item(item_id, price, quantity): Add item to cart\n"
            "- set_member_level(level): Set membership (0=regular, 1=silver, 2=gold, 3=platinum)\n"
            "- apply_coupon(coupon_type, value): Apply coupon ('percentage' or 'fixed')\n"
            "- calculate_total(): Calculate final price after all discounts\n"
            "Rules: bulk discount (5+: 10%, 10+: 20%), member discount, "
            "threshold (>=100: 5%, >=200: 10%, >=500: 15%), "
            "coupon (max 30% or $50), final >= 0"
        )

    def _evaluate(self, code: str) -> Dict[str, Any]:
        passed_tests = 0
        total_tests = len(self.test_cases)
        errors = []
        has_syntax_error = False

        try:
            namespace = {}
            exec(code, namespace)

            if "OrderSystem" not in namespace:
                errors.append("OrderSystem class not found")
                return {
                    "loss": 1.0, "passed_tests": 0, "total_tests": total_tests,
                    "errors": errors, "has_syntax_error": True,
                }

            OrderSystem = namespace["OrderSystem"]
            for i, test_case in enumerate(self.test_cases):
                try:
                    system = OrderSystem()
                    for item_id, price, quantity in test_case["items"]:
                        system.add_item(item_id, price, quantity)
                    system.set_member_level(test_case["member_level"])
                    if test_case["coupon"]:
                        coupon_type, value = test_case["coupon"]
                        system.apply_coupon(coupon_type, value)
                    result = system.calculate_total()
                    expected = test_case["expected"]
                    if abs(result - expected) < 0.01:
                        passed_tests += 1
                    else:
                        errors.append(f"Test {i+1} failed: expected {expected:.2f}, got {result:.2f}")
                except Exception as e:
                    errors.append(f"Test {i+1} error: {str(e)}")

        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")
            has_syntax_error = True
        except Exception as e:
            errors.append(f"Execution error: {str(e)}")
            has_syntax_error = True

        loss = 1.0 - (passed_tests / total_tests) if total_tests > 0 else 1.0
        return {
            "loss": loss, "passed_tests": passed_tests, "total_tests": total_tests,
            "errors": errors[:3], "has_syntax_error": has_syntax_error,
        }
