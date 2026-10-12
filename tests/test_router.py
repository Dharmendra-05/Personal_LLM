"""Unit tests for QueryRouter keyword and word-boundary behavior."""

import unittest
from core.router import QueryRouter, RouteMode


class TestQueryRouter(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter()

    def test_empty_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.router.route("")
        with self.assertRaises(ValueError):
            self.router.route("   ")

    def test_word_boundary_no_substring_false_positives(self):
        """Ensure short keywords (hi, ok, my, mine) do not trigger on substrings."""
        # 'machine' should not trigger 'hi'
        d1 = self.router.route("machine learning algorithms")
        self.assertNotIn("kw:hi", d1.matched_signals)

        # 'which' should not trigger 'hi'
        d2 = self.router.route("Which model performs best?")
        self.assertNotIn("kw:hi", d2.matched_signals)

        # 'tokens' should not trigger 'ok'
        d3 = self.router.route("How many tokens were generated?")
        self.assertNotIn("kw:ok", d3.matched_signals)

        # 'determine' should not trigger 'mine'
        d4 = self.router.route("Determine the principal values")
        self.assertNotIn("kw:mine", d4.matched_signals)

        # 'book' should not trigger 'ok'
        d5 = self.router.route("Book a room for next week")
        self.assertNotIn("kw:ok", d5.matched_signals)

    def test_intended_keyword_routes(self):
        """Ensure actual keywords match properly when separated by word boundaries."""
        # Greeting
        d_hi = self.router.route("Hi, how are you doing today?")
        self.assertIn("keyword:hi", d_hi.matched_signals)
        self.assertEqual(d_hi.mode, RouteMode.GENERAL_CHAT)

        # Advanced query
        d_adv = self.router.route("Explain the theory of general relativity")
        self.assertEqual(d_adv.mode, RouteMode.ADVANCED_KNOWLEDGE)

        # Personal memory query
        d_mem = self.router.route("What did my notes say about the meeting?")
        self.assertEqual(d_mem.mode, RouteMode.PERSONAL_MEMORY)

    def test_explain_output(self):
        explanation = self.router.explain("What is the history of the universe?")
        self.assertIn("Decision :", explanation)
        self.assertIn("history of the universe", explanation)


if __name__ == "__main__":
    unittest.main()
