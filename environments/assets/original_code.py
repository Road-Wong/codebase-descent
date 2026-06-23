"""
Saddle Point Task — Expression Parser & Calculator (Ground Truth c*)

This is the "global optimum" that the Agent must recover from
obfuscated/degraded initial states.  It implements a simple
arithmetic interpreter with:

  - Lexer  (tokenize): digits, + - * /, parentheses
  - Parser (evaluate): recursive descent, correct precedence
  - Coupled components: modifying lexer breaks parser and vice versa

Test suite: 26 expressions covering precedence, nesting, edge cases.
Loss function: L(c) = 1 - passed/26
"""


class SimpleInterpreter:
    def __init__(self):
        self.tokens = []

    def tokenize(self, expr):
        """Tokenize an arithmetic expression into a list of (type, value) pairs."""
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i].isspace():
                i += 1
                continue
            if expr[i].isdigit():
                num = ''
                while i < len(expr) and expr[i].isdigit():
                    num += expr[i]
                    i += 1
                tokens.append(('NUM', int(num)))
            elif expr[i] in '+-*/':
                tokens.append(('OP', expr[i]))
                i += 1
            elif expr[i] in '()':
                tokens.append(('PAREN', expr[i]))
                i += 1
            else:
                i += 1
        return tokens

    def evaluate(self, expr):
        """Evaluate an arithmetic expression and return the result."""
        tokens = self.tokenize(expr)
        return self._eval_expr(tokens, 0)[0]

    # ------------------------------------------------------------------
    # Recursive descent parser
    # ------------------------------------------------------------------

    def _eval_expr(self, tokens, pos):
        """expr = term (('+' | '-') term)*"""
        left, pos = self._eval_term(tokens, pos)
        while pos < len(tokens) and tokens[pos][0] == 'OP' and tokens[pos][1] in '+-':
            op = tokens[pos][1]
            pos += 1
            right, pos = self._eval_term(tokens, pos)
            if op == '+':
                left = left + right
            else:
                left = left - right
        return left, pos

    def _eval_term(self, tokens, pos):
        """term = factor (('*' | '/') factor)*"""
        left, pos = self._eval_factor(tokens, pos)
        while pos < len(tokens) and tokens[pos][0] == 'OP' and tokens[pos][1] in '*/':
            op = tokens[pos][1]
            pos += 1
            right, pos = self._eval_factor(tokens, pos)
            if op == '*':
                left = left * right
            else:
                left = left // right
        return left, pos

    def _eval_factor(self, tokens, pos):
        """factor = NUMBER | '(' expr ')'"""
        if tokens[pos][0] == 'NUM':
            return tokens[pos][1], pos + 1
        elif tokens[pos][0] == 'PAREN' and tokens[pos][1] == '(':
            pos += 1
            result, pos = self._eval_expr(tokens, pos)
            pos += 1  # skip ')'
            return result, pos
        return 0, pos
