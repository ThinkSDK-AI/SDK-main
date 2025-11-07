"""
Unit tests for Agent Thinking Mode (Production Grade).

Tests cover:
- Configuration validation
- Input sanitization
- Error handling
- Rate limiting
- Context truncation
- Query generation
- Search execution
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from agent import Agent, AgentConfig
from fourier import Fourier


class TestAgentConfigValidation(unittest.TestCase):
    """Test AgentConfig parameter validation."""

    def test_valid_config(self):
        """Test that valid configuration is accepted."""
        config = AgentConfig(
            thinking_mode=True,
            thinking_depth=3,
            thinking_web_search_results=5
        )
        self.assertEqual(config.thinking_depth, 3)
        self.assertEqual(config.thinking_web_search_results, 5)

    def test_thinking_depth_clamping_min(self):
        """Test that thinking_depth is clamped to minimum."""
        config = AgentConfig(
            thinking_mode=True,
            thinking_depth=0
        )
        self.assertEqual(config.thinking_depth, 1)

    def test_thinking_depth_clamping_max(self):
        """Test that thinking_depth is clamped to maximum."""
        config = AgentConfig(
            thinking_mode=True,
            thinking_depth=10
        )
        self.assertEqual(config.thinking_depth, 5)

    def test_search_results_clamping_min(self):
        """Test that search results are clamped to minimum."""
        config = AgentConfig(
            thinking_mode=True,
            thinking_web_search_results=0
        )
        self.assertEqual(config.thinking_web_search_results, 1)

    def test_search_results_clamping_max(self):
        """Test that search results are clamped to maximum."""
        config = AgentConfig(
            thinking_mode=True,
            thinking_web_search_results=20
        )
        self.assertEqual(config.thinking_web_search_results, 10)

    def test_temperature_clamping(self):
        """Test that temperature is clamped to valid range."""
        config = AgentConfig(temperature=5.0)
        self.assertEqual(config.temperature, 2.0)

        config = AgentConfig(temperature=-1.0)
        self.assertEqual(config.temperature, 0.0)

    def test_max_iterations_minimum(self):
        """Test that max_iterations has minimum value."""
        config = AgentConfig(max_iterations=0)
        self.assertEqual(config.max_iterations, 1)


class TestQuerySanitization(unittest.TestCase):
    """Test query sanitization functionality."""

    def test_sanitize_valid_query(self):
        """Test sanitization of valid query."""
        query = "What are the latest developments in AI?"
        result = Agent._sanitize_query(query)
        self.assertEqual(result, query)

    def test_sanitize_whitespace(self):
        """Test removal of excessive whitespace."""
        query = "What  are   the    latest     developments?"
        result = Agent._sanitize_query(query)
        self.assertEqual(result, "What are the latest developments?")

    def test_sanitize_too_short(self):
        """Test rejection of too-short queries."""
        query = "a"
        result = Agent._sanitize_query(query)
        self.assertIsNone(result)

    def test_sanitize_too_long(self):
        """Test truncation of too-long queries."""
        query = "a" * 1000
        result = Agent._sanitize_query(query)
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 500)

    def test_sanitize_suspicious_characters(self):
        """Test removal of suspicious characters."""
        query = "What is <script>alert('xss')</script> in AI?"
        result = Agent._sanitize_query(query)
        self.assertIsNotNone(result)
        self.assertNotIn("<script>", result)
        self.assertNotIn("</script>", result)

    def test_sanitize_unicode(self):
        """Test handling of Unicode characters."""
        query = "What are the latest developments in AI研究?"
        result = Agent._sanitize_query(query)
        self.assertIsNotNone(result)

    def test_sanitize_none_input(self):
        """Test handling of None input."""
        result = Agent._sanitize_query(None)
        self.assertIsNone(result)

    def test_sanitize_empty_string(self):
        """Test handling of empty string."""
        result = Agent._sanitize_query("")
        self.assertIsNone(result)

    def test_sanitize_non_string(self):
        """Test handling of non-string input."""
        result = Agent._sanitize_query(12345)
        self.assertIsNone(result)


class TestContextTruncation(unittest.TestCase):
    """Test context truncation functionality."""

    def test_truncate_short_context(self):
        """Test that short context is not truncated."""
        context = "Short context"
        result = Agent._truncate_context(context, max_length=100)
        self.assertEqual(result, context)

    def test_truncate_long_context(self):
        """Test that long context is truncated."""
        context = "a" * 10000
        result = Agent._truncate_context(context, max_length=1000)
        self.assertLessEqual(len(result), 1100)  # Includes truncation message
        self.assertIn("[Context truncated", result)

    def test_truncate_exact_length(self):
        """Test context at exact max length."""
        context = "a" * 1000
        result = Agent._truncate_context(context, max_length=1000)
        self.assertEqual(result, context)


class TestThinkingModeResearch(unittest.TestCase):
    """Test thinking mode research functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock(spec=Fourier)
        self.mock_client.chat = Mock(return_value={
            "response": {"output": "test query 1\ntest query 2"}
        })

        self.agent = Agent(
            client=self.mock_client,
            name="TestAgent",
            model="test-model",
            config=AgentConfig(
                thinking_mode=True,
                thinking_depth=2,
                thinking_web_search_results=3,
                verbose=False
            )
        )

    @patch('agent.web_search')
    def test_research_successful(self, mock_web_search):
        """Test successful research execution."""
        mock_web_search.return_value = "Search results content"

        result = self.agent._perform_thinking_research("Test query")

        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn("Research Query", result)

    @patch('agent.web_search')
    def test_research_empty_results(self, mock_web_search):
        """Test handling of empty search results."""
        mock_web_search.return_value = None

        result = self.agent._perform_thinking_research("Test query")

        # Should return empty string when no results
        self.assertEqual(result, "")

    @patch('agent.web_search')
    def test_research_search_failure(self, mock_web_search):
        """Test handling of search failures."""
        mock_web_search.side_effect = Exception("Search failed")

        # Should not raise when stop_on_error is False
        result = self.agent._perform_thinking_research("Test query")
        self.assertEqual(result, "")

    @patch('agent.web_search')
    def test_research_invalid_input(self, mock_web_search):
        """Test handling of invalid input."""
        result = self.agent._perform_thinking_research(None)
        self.assertEqual(result, "")

        result = self.agent._perform_thinking_research("")
        self.assertEqual(result, "")

    @patch('agent.web_search')
    def test_research_context_truncation(self, mock_web_search):
        """Test that very long context is truncated."""
        # Return very long search results
        mock_web_search.return_value = "a" * 100000

        result = self.agent._perform_thinking_research("Test query")

        # Should be truncated to max context length
        self.assertLessEqual(len(result), 50100)  # MAX + truncation message

    @patch('agent.web_search')
    def test_research_rate_limiting(self, mock_web_search):
        """Test that rate limiting delay is applied."""
        mock_web_search.return_value = "Results"

        start_time = time.time()
        self.agent._perform_thinking_research("Test query")
        elapsed = time.time() - start_time

        # Should have delays between searches (2 queries with 1s delay)
        self.assertGreaterEqual(elapsed, 1.0)


class TestSearchQueryGeneration(unittest.TestCase):
    """Test search query generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock(spec=Fourier)
        self.agent = Agent(
            client=self.mock_client,
            name="TestAgent",
            model="test-model",
            config=AgentConfig(thinking_mode=True, verbose=False)
        )

    def test_generate_queries_success(self):
        """Test successful query generation."""
        self.mock_client.chat.return_value = {
            "response": {"output": "query 1\nquery 2\nquery 3"}
        }

        queries = self.agent._generate_search_queries("Test input", 3)

        self.assertEqual(len(queries), 3)
        self.assertIn("query 1", queries)

    def test_generate_queries_empty_response(self):
        """Test handling of empty LLM response."""
        self.mock_client.chat.return_value = {
            "response": {"output": ""}
        }

        queries = self.agent._generate_search_queries("Test input", 2)

        # Should fallback to original input
        self.assertEqual(len(queries), 1)
        self.assertIn("Test input", queries[0])

    def test_generate_queries_llm_failure(self):
        """Test handling of LLM failure."""
        self.mock_client.chat.side_effect = Exception("LLM failed")

        queries = self.agent._generate_search_queries("Test input", 2)

        # Should fallback to original input
        self.assertEqual(len(queries), 1)

    def test_generate_queries_sanitization(self):
        """Test that generated queries are sanitized."""
        self.mock_client.chat.return_value = {
            "response": {"output": "valid query\n<script>bad</script>\ngood query"}
        }

        queries = self.agent._generate_search_queries("Test input", 3)

        # Bad query should be sanitized or removed
        for query in queries:
            self.assertNotIn("<script>", query)


class TestSingleSearch(unittest.TestCase):
    """Test single search execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock(spec=Fourier)
        self.agent = Agent(
            client=self.mock_client,
            name="TestAgent",
            model="test-model",
            config=AgentConfig(
                thinking_mode=True,
                thinking_web_search_results=5,
                verbose=False
            )
        )

    @patch('agent.web_search')
    @patch('time.sleep')
    def test_search_rate_limiting(self, mock_sleep, mock_web_search):
        """Test that rate limiting is applied for subsequent searches."""
        mock_web_search.return_value = "Results"

        # First search should not sleep
        self.agent._perform_single_search("query1", 1, 3)
        mock_sleep.assert_not_called()

        # Second search should sleep
        self.agent._perform_single_search("query2", 2, 3)
        mock_sleep.assert_called_once()

    @patch('agent.web_search')
    def test_search_success(self, mock_web_search):
        """Test successful search."""
        mock_web_search.return_value = "Search results"

        result = self.agent._perform_single_search("test query", 1, 1)

        self.assertEqual(result, "Search results")

    @patch('agent.web_search')
    def test_search_failure(self, mock_web_search):
        """Test search failure handling."""
        mock_web_search.side_effect = Exception("Search failed")

        result = self.agent._perform_single_search("test query", 1, 1)

        self.assertIsNone(result)

    @patch('agent.web_search')
    def test_search_empty_results(self, mock_web_search):
        """Test handling of empty results."""
        mock_web_search.return_value = None

        result = self.agent._perform_single_search("test query", 1, 1)

        self.assertIsNone(result)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
