"""
Order Discount Environment - Complex business logic task.

This environment tests the agent's ability to implement a complex e-commerce
order discount calculation system with multiple interacting rules.

Task Complexity:
- Multiple discount types with priority rules
- Coupon stacking logic with restrictions
- Inventory validation
- Complex edge cases and business constraints
- Non-obvious rule interactions
"""

from typing import Dict, Any, Tuple, Optional, List
from .abstract_env import AbstractEnvironment


class OrderDiscountEnvironment(AbstractEnvironment):
    """
    Environment for implementing an e-commerce order discount calculator.
    
    The task is to implement an OrderSystem class with:
    - __init__(): Initialize the system
    - add_item(item_id, price, quantity): Add item to cart
    - set_member_level(level): Set customer membership level (0=regular, 1=silver, 2=gold, 3=platinum)
    - apply_coupon(coupon_type, value): Apply a coupon (percentage or fixed amount)
    - calculate_total(): Calculate final price after all discounts
    
    Business Rules (complex and interacting):
    1. Member discounts (applied to subtotal):
       - Regular: 0% off
       - Silver: 5% off
       - Gold: 10% off
       - Platinum: 15% off
    
    2. Bulk discounts (per item, before member discount):
       - 5+ items of same type: 10% off that item
       - 10+ items of same type: 20% off that item
    
    3. Order threshold discounts:
       - Subtotal >= $100: additional 5% off
       - Subtotal >= $200: additional 10% off
       - Subtotal >= $500: additional 15% off
    
    4. Coupon rules:
       - Only ONE coupon can be applied
       - Percentage coupons: max 30% off
       - Fixed amount coupons: max $50 off
       - Coupons applied AFTER member discount but BEFORE threshold discount
    
    5. Calculation order:
       - Calculate item prices with bulk discounts
       - Sum to get subtotal
       - Apply member discount
       - Apply coupon (if any)
       - Apply threshold discount
       - Final total cannot be negative
    
    This is harder because:
    1. Multiple discount types interact in non-obvious ways
    2. Order of operations matters critically
    3. Many edge cases and constraints
    4. Business logic is domain-specific
    5. Not a standard algorithm from textbooks
    """
    
    def __init__(self, ground_truth: Optional[str] = None):
        """Initialize Order Discount environment."""
        self.c_star = ground_truth or self._get_ground_truth()
        super().__init__(self.c_star)
        
        # Test cases with complex scenarios
        self.test_cases = [
            # Test 1: Simple order, no discounts
            {
                "items": [(1, 50.0, 1)],  # (item_id, price, quantity)
                "member_level": 0,
                "coupon": None,
                "expected": 50.0
            },
            
            # Test 2: Member discount only
            {
                "items": [(1, 100.0, 1)],
                "member_level": 2,  # Gold: 10% off
                "coupon": None,
                "expected": 90.0  # 100 * 0.9
            },
            
            # Test 3: Bulk discount + member discount
            {
                "items": [(1, 10.0, 6)],  # 6 items: 10% bulk discount
                "member_level": 1,  # Silver: 5% off
                "coupon": None,
                "expected": 51.3  # (10*6*0.9) * 0.95 = 54 * 0.95
            },
            
            # Test 4: Threshold discount (>= $100)
            {
                "items": [(1, 100.0, 1)],
                "member_level": 0,
                "coupon": None,
                "expected": 95.0  # 100 * 0.95 (5% threshold discount)
            },
            
            # Test 5: Member + coupon + threshold
            {
                "items": [(1, 100.0, 2)],  # $200 subtotal
                "member_level": 2,  # Gold: 10% off -> $180
                "coupon": ("percentage", 10),  # 10% off -> $162
                "expected": 153.9  # $162 * 0.95 (5% threshold, since $162 < $200)
            },
            
            # Test 6: Large bulk discount
            {
                "items": [(1, 10.0, 12)],  # 12 items: 20% bulk discount = $96
                "member_level": 0,  # No member discount
                "coupon": None,
                "expected": 96.0  # $96 < $100, no threshold discount
            },
            
            # Test 7: Fixed coupon
            {
                "items": [(1, 80.0, 1)],
                "member_level": 1,  # Silver: 5% off -> $76
                "coupon": ("fixed", 20),  # $20 off -> $56
                "expected": 56.0  # No threshold discount (< $100)
            },
            
            # Test 8: Multiple items, complex
            {
                "items": [(1, 30.0, 3), (2, 40.0, 2)],  # $90 + $80 = $170
                "member_level": 3,  # Platinum: 15% off -> $144.5
                "coupon": ("percentage", 15),  # 15% off -> $122.825
                "expected": 116.68375  # $122.825 * 0.95 (5% threshold, since $122.825 < $200)
            },
            
            # Test 9: Coupon max limit (percentage)
            {
                "items": [(1, 100.0, 1)],
                "member_level": 0,
                "coupon": ("percentage", 50),  # Should be capped at 30%
                "expected": 70.0  # 100 * 0.7 = $70 < $100, no threshold discount
            },
            
            # Test 10: Coupon max limit (fixed)
            {
                "items": [(1, 100.0, 1)],
                "member_level": 0,
                "coupon": ("fixed", 80),  # Should be capped at $50
                "expected": 50.0  # 100 - 50 = $50 < $100, no threshold discount
            },
            
            # Test 11: High threshold discount (>= $500)
            {
                "items": [(1, 100.0, 6)],  # $600 with 10% bulk = $540
                "member_level": 2,  # Gold: 10% off -> $486
                "coupon": None,
                "expected": 437.4  # $486 * 0.9 (10% threshold, since $486 < $500)
            },
            
            # Test 12: Edge case - result near zero
            {
                "items": [(1, 50.0, 1)],
                "member_level": 3,  # Platinum: 15% off -> $42.5
                "coupon": ("fixed", 50),  # Max $50 off, but can't go negative
                "expected": 0.0  # Max($42.5 - $50, 0)
            },
            
            # Test 13: Multiple items with different bulk discounts
            {
                "items": [(1, 20.0, 8), (2, 15.0, 3)],  # $160*0.9 + $45 = $189
                "member_level": 1,  # Silver: 5% off -> $179.55
                "coupon": ("percentage", 10),  # 10% off -> $161.595
                "expected": 153.51525  # $161.595 * 0.95 (5% threshold, since $161.595 < $200)
            },
            
            # Test 14: Platinum member with high-value order
            {
                "items": [(1, 150.0, 4)],  # $600
                "member_level": 3,  # Platinum: 15% off -> $510
                "coupon": ("percentage", 20),  # 20% off -> $408
                "expected": 367.2  # $408 * 0.9 (10% threshold, since $408 < $500)
            },
            
            # Test 15: Complex multi-item with all discounts
            {
                "items": [(1, 25.0, 11), (2, 30.0, 7), (3, 20.0, 4)],
                # Item 1: 25*11*0.8 = 220
                # Item 2: 30*7*0.9 = 189
                # Item 3: 20*4 = 80
                # Subtotal: 489
                "member_level": 2,  # Gold: 10% off -> $440.1
                "coupon": ("fixed", 40),  # $40 off -> $400.1
                "expected": 360.09  # $400.1 * 0.9 (10% threshold, since $400.1 < $500)
            }
        ]
    
    def _get_ground_truth(self) -> str:
        """Return the ground truth implementation."""
        return """class OrderSystem:
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
        # Step 1: Calculate item prices with bulk discounts
        subtotal = 0
        for item_id, item_data in self.items.items():
            price = item_data['price']
            quantity = item_data['quantity']
            
            # Apply bulk discount
            if quantity >= 10:
                price *= 0.8  # 20% off
            elif quantity >= 5:
                price *= 0.9  # 10% off
            
            subtotal += price * quantity
        
        # Step 2: Apply member discount
        member_discounts = {0: 1.0, 1: 0.95, 2: 0.9, 3: 0.85}
        subtotal *= member_discounts.get(self.member_level, 1.0)
        
        # Step 3: Apply coupon
        if self.coupon:
            coupon_type, value = self.coupon
            if coupon_type == 'percentage':
                # Cap at 30%
                discount_rate = min(value, 30) / 100
                subtotal *= (1 - discount_rate)
            elif coupon_type == 'fixed':
                # Cap at $50
                discount_amount = min(value, 50)
                subtotal -= discount_amount
        
        # Step 4: Apply threshold discount (based on current subtotal)
        if subtotal >= 500:
            subtotal *= 0.85  # 15% off
        elif subtotal >= 200:
            subtotal *= 0.9   # 10% off
        elif subtotal >= 100:
            subtotal *= 0.95  # 5% off
        
        # Step 5: Ensure non-negative
        return max(subtotal, 0.0)
"""
    
    def step(self, code: str) -> Tuple[float, Dict[str, Any]]:
        """Execute one step and return loss and info."""
        self.step_count += 1
        loss = self.get_loss(code)
        info = self.evaluate(code)
        info['step'] = self.step_count
        info['loss'] = loss
        return loss, info
    
    def get_loss(self, code: str) -> float:
        """Calculate loss for given code."""
        info = self.evaluate(code)
        
        # Base loss from failed tests
        test_loss = 1.0 - (info['passed_tests'] / info['total_tests'])
        
        # Penalty for syntax errors
        syntax_penalty = 0.2 if info['has_syntax_error'] else 0.0
        
        # Semantic distance
        semantic_distance = self._semantic_distance(code)
        
        # Combined loss
        loss = test_loss + syntax_penalty + 0.1 * semantic_distance
        
        return loss
    
    def evaluate(self, code: str) -> Dict[str, Any]:
        """Evaluate code against test cases."""
        passed_tests = 0
        total_tests = len(self.test_cases)
        errors = []
        has_syntax_error = False
        
        try:
            namespace = {}
            exec(code, namespace)
            
            if 'OrderSystem' not in namespace:
                errors.append("OrderSystem class not found")
                has_syntax_error = True
                return {
                    'passed_tests': 0,
                    'total_tests': total_tests,
                    'errors': errors,
                    'has_syntax_error': has_syntax_error
                }
            
            OrderSystem = namespace['OrderSystem']
            
            # Run each test case
            for i, test_case in enumerate(self.test_cases):
                try:
                    system = OrderSystem()
                    
                    # Add items
                    for item_id, price, quantity in test_case['items']:
                        system.add_item(item_id, price, quantity)
                    
                    # Set member level
                    system.set_member_level(test_case['member_level'])
                    
                    # Apply coupon if exists
                    if test_case['coupon']:
                        coupon_type, value = test_case['coupon']
                        system.apply_coupon(coupon_type, value)
                    
                    # Calculate total
                    result = system.calculate_total()
                    expected = test_case['expected']
                    
                    # Check if result matches (with small tolerance for floating point)
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
        
        return {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'errors': errors[:3],
            'has_syntax_error': has_syntax_error
        }
    
    def _semantic_distance(self, code: str) -> float:
        """Calculate semantic distance from ground truth."""
        # Check for key components
        has_init = '__init__' in code
        has_add_item = 'add_item' in code
        has_set_member = 'set_member_level' in code
        has_apply_coupon = 'apply_coupon' in code
        has_calculate = 'calculate_total' in code
        has_bulk_logic = ('0.8' in code or '0.9' in code) and 'quantity' in code
        has_member_logic = ('0.95' in code or '0.85' in code) and 'member' in code
        has_threshold_logic = '100' in code or '200' in code or '500' in code
        
        components = [
            has_init, has_add_item, has_set_member, has_apply_coupon,
            has_calculate, has_bulk_logic, has_member_logic, has_threshold_logic
        ]
        present_count = sum(components)
        
        distance = 1.0 - (present_count / len(components))
        return distance
    
    def reset(self):
        """Reset environment state."""
        self.step_count = 0
    
    def is_solved(self, code: str) -> bool:
        """Check if code solves the task."""
        info = self.evaluate(code)
        return info['passed_tests'] == info['total_tests']
