"""
Unit tests for metrics utils module.

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

import pytest
from unittest.mock import patch, MagicMock

from elastic_gumby_universal_orch_agent_prototype.metrics.utils import (
    Colors,
    _colorize,
    format_score,
    format_ratio,
    compare_workflow
)


class TestColorizeFunction:
    """Test the _colorize function."""

    def test_colorize_empty_text(self):
        """Test colorizing empty text."""
        result = _colorize("", Colors.ERROR)
        assert result == Colors.ERROR + Colors.RESET

    def test_colorize_with_special_characters(self):
        """Test colorizing text with special characters."""
        text = "test\nwith\ttabs"
        result = _colorize(text, Colors.WARNING)
        assert text in result
        assert result.startswith(Colors.WARNING)
        assert result.endswith(Colors.RESET)

    def test_colorize_with_unicode(self):
        """Test colorizing text with unicode characters."""
        text = "test 🧠 unicode"
        result = _colorize(text, Colors.INFO)
        assert text in result
        assert result.startswith(Colors.INFO)
        assert result.endswith(Colors.RESET)


class TestFormatScore:
    """Test the format_score function."""

    def test_format_score_high_value(self):
        """Test formatting high score values (>= 0.8)."""
        result = format_score(0.9)
        assert "0.900" in result
        assert Colors.SUCCESS in result
        assert Colors.RESET in result

    def test_format_score_medium_value(self):
        """Test formatting medium score values (0.6 <= score < 0.8)."""
        result = format_score(0.7)
        assert "0.700" in result
        assert Colors.WARNING in result
        assert Colors.RESET in result

    def test_format_score_low_value(self):
        """Test formatting low score values (< 0.6)."""
        result = format_score(0.3)
        assert "0.300" in result
        assert Colors.ERROR in result
        assert Colors.RESET in result

    def test_format_score_boundary_values(self):
        """Test formatting boundary values."""
        # Test exactly at good threshold
        result_good = format_score(0.8)
        assert Colors.SUCCESS in result_good
        
        # Test exactly at ok threshold
        result_ok = format_score(0.6)
        assert Colors.WARNING in result_ok
        
        # Test just below ok threshold
        result_bad = format_score(0.59)
        assert Colors.ERROR in result_bad

    def test_format_score_precision(self):
        """Test score formatting precision."""
        result = format_score(0.123456789)
        assert "0.123" in result  # Should be rounded to 3 decimal places


class TestFormatRatio:
    """Test the format_ratio function."""

    def test_format_ratio_zero_denominator(self):
        """Test formatting ratio with zero denominator."""
        result = format_ratio(5, 0)
        assert "N/A" in result
        assert Colors.DIM in result
        assert Colors.RESET in result

    def test_format_ratio_high_completeness(self):
        """Test formatting ratio with high completeness (>= 0.9)."""
        result = format_ratio(9, 10)
        assert "9/10" in result
        assert Colors.SUCCESS in result
        assert Colors.RESET in result

    def test_format_ratio_medium_completeness(self):
        """Test formatting ratio with medium completeness (0.7 <= ratio < 0.9)."""
        result = format_ratio(7, 10)
        assert "7/10" in result
        assert Colors.WARNING in result
        assert Colors.RESET in result

    def test_format_ratio_low_completeness(self):
        """Test formatting ratio with low completeness (< 0.7)."""
        result = format_ratio(3, 10)
        assert "3/10" in result
        assert Colors.ERROR in result
        assert Colors.RESET in result

    def test_format_ratio_boundary_values(self):
        """Test formatting boundary ratio values."""
        # Test exactly at high threshold
        result_high = format_ratio(9, 10)  # 0.9
        assert Colors.SUCCESS in result_high
        
        # Test exactly at medium threshold
        result_medium = format_ratio(7, 10)  # 0.7
        assert Colors.WARNING in result_medium
        
        # Test just below medium threshold
        result_low = format_ratio(6, 10)  # 0.6
        assert Colors.ERROR in result_low


class TestCompareWorkflow:
    """Test the compare_workflow function."""

    @pytest.fixture
    def sample_workflow(self):
        """Sample workflow fixture."""
        return {
            "type": "sequence",
            "children": [
                {
                    "type": "tool_call",
                    "tool": "test_tool",
                    "parameters": {"param1": "value1"}
                }
            ]
        }
        
    @pytest.fixture
    def reference_workflow(self):
        """Reference workflow fixture."""
        return {
            "type": "sequence",
            "children": [
                {
                    "type": "tool_call",
                    "tool": "test_tool",
                    "parameters": {"param1": "value1"}
                },
                {
                    "type": "tool_call",
                    "tool": "missing_tool",
                    "parameters": {"param2": "value2"}
                }
            ]
        }

    @patch('elastic_gumby_universal_orch_agent_prototype.metrics.utils.StructuralMetric')
    @patch('elastic_gumby_universal_orch_agent_prototype.metrics.utils.SemanticMetric')
    def test_compare_workflow_with_missing_structural_patterns(self, mock_semantic, mock_structural, sample_workflow, reference_workflow):
        """Test workflow comparison with missing structural patterns."""
        # Mock structural analysis with misses
        mock_structural_instance = MagicMock()
        mock_structural.return_value = mock_structural_instance
        mock_structural_instance.workflow_structural_analysis.return_value = {
            'subtree_match_ratio': 0.5,
            'weighted_subtree_match_accuracy': 0.4,
            'match_count': 2,
            'total_subtrees': 4,
            'miss_count': 2,
            'misses': [
                {'type': 'tool_call', 'depth': 1, 'children_count': 0},
                {'type': 'sequence', 'depth': 2, 'children_count': 3},
                {'type': 'parallel', 'depth': 1, 'children_count': 2}
            ],
            'detailed_breakdown': {
                'action_nodes_matched': 1,
                'action_nodes_missing': 2,
                'container_nodes_matched': 1,
                'container_nodes_missing': 1,
                'deep_structures_matched': 1,
                'shallow_structures_matched': 1
            }
        }
        
        # Mock semantic analysis
        mock_semantic_instance = MagicMock()
        mock_semantic.return_value = mock_semantic_instance
        mock_semantic_instance.workflow_semantic_analysis.return_value = {
            'average_tool_call_similarity': 0.6,
            'tool_call_similarity': {},
            'missing_tools': [],
            'average_variable_definition_similarity': 0.5,
            'average_variable_usage_similarity': 0.4,
            'variable_usage_similarity': {},
            'missing_variables': []
        }
        
        result = compare_workflow(sample_workflow, reference_workflow)
        
        # Verify missing patterns section is included
        assert "MISSING STRUCTURAL PATTERNS" in result
        assert "Missing Tool Call Nodes:" in result
        assert "Missing Sequence Nodes:" in result
        assert "Missing Parallel Nodes:" in result

    @patch('elastic_gumby_universal_orch_agent_prototype.metrics.utils.StructuralMetric')
    @patch('elastic_gumby_universal_orch_agent_prototype.metrics.utils.SemanticMetric')
    def test_compare_workflow_with_many_missing_items(self, mock_semantic, mock_structural, sample_workflow, reference_workflow):
        """Test workflow comparison with many missing items (>5)."""
        # Create many missing items
        many_misses = [
            {'type': 'tool_call', 'depth': i, 'children_count': 0}
            for i in range(10)
        ]
        
        mock_structural_instance = MagicMock()
        mock_structural.return_value = mock_structural_instance
        mock_structural_instance.workflow_structural_analysis.return_value = {
            'subtree_match_ratio': 0.1,
            'weighted_subtree_match_accuracy': 0.1,
            'match_count': 1,
            'total_subtrees': 10,
            'miss_count': 10,
            'misses': many_misses,
            'detailed_breakdown': {
                'action_nodes_matched': 0,
                'action_nodes_missing': 10,
                'container_nodes_matched': 0,
                'container_nodes_missing': 0,
                'deep_structures_matched': 0,
                'shallow_structures_matched': 0
            }
        }
        
        # Mock semantic analysis with many missing items
        mock_semantic_instance = MagicMock()
        mock_semantic.return_value = mock_semantic_instance
        mock_semantic_instance.workflow_semantic_analysis.return_value = {
            'average_tool_call_similarity': 0.2,
            'tool_call_similarity': {},
            'missing_tools': [f'tool_{i}' for i in range(8)],
            'average_variable_definition_similarity': 0.1,
            'average_variable_usage_similarity': 0.1,
            'variable_usage_similarity': {},
            'missing_variables': [f'var_{i}' for i in range(7)]
        }
        
        result = compare_workflow(sample_workflow, reference_workflow)
        
        # Verify truncation messages are included
        assert "... and 5 more" in result  # For missing structural patterns
        assert "... and 3 more" in result  # For missing tools
        assert "... and 2 more" in result  # For missing variables

    @patch('elastic_gumby_universal_orch_agent_prototype.metrics.utils.StructuralMetric')
    @patch('elastic_gumby_universal_orch_agent_prototype.metrics.utils.SemanticMetric')
    def test_compare_workflow_zero_scores(self, mock_semantic, mock_structural, sample_workflow, reference_workflow):
        """Test workflow comparison with zero scores."""
        # Mock zero structural analysis
        mock_structural_instance = MagicMock()
        mock_structural.return_value = mock_structural_instance
        mock_structural_instance.workflow_structural_analysis.return_value = {
            'subtree_match_ratio': 0.0,
            'weighted_subtree_match_accuracy': 0.0,
            'match_count': 0,
            'total_subtrees': 5,
            'miss_count': 5,
            'misses': []
        }
        
        # Mock zero semantic analysis
        mock_semantic_instance = MagicMock()
        mock_semantic.return_value = mock_semantic_instance
        mock_semantic_instance.workflow_semantic_analysis.return_value = {
            'average_tool_call_similarity': 0.0,
            'average_variable_definition_similarity': 0.0,
            'average_variable_usage_similarity': 0.0
        }
        
        result = compare_workflow(sample_workflow, reference_workflow)
        
        # Verify zero scores are handled
        assert "0.000" in result
        assert "0/5" in result

    @patch('elastic_gumby_universal_orch_agent_prototype.metrics.utils.StructuralMetric')
    @patch('elastic_gumby_universal_orch_agent_prototype.metrics.utils.SemanticMetric')
    def test_compare_workflow_missing_optional_fields(self, mock_semantic, mock_structural, sample_workflow, reference_workflow):
        """Test workflow comparison with missing optional fields."""
        # Mock structural analysis without optional fields
        mock_structural_instance = MagicMock()
        mock_structural.return_value = mock_structural_instance
        mock_structural_instance.workflow_structural_analysis.return_value = {
            'match_count': 3,
            'total_subtrees': 5,
            'miss_count': 2,
            'misses': []
        }
        
        # Mock semantic analysis without optional fields
        mock_semantic_instance = MagicMock()
        mock_semantic.return_value = mock_semantic_instance
        mock_semantic_instance.workflow_semantic_analysis.return_value = {}
        
        result = compare_workflow(sample_workflow, reference_workflow)
        
        # Verify it handles missing fields gracefully
        assert isinstance(result, str)
        assert "WORKFLOW COMPARISON ANALYSIS" in result
        assert "0.000" in result  # Default values should be 0