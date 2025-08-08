"""
Tests for StructuralMetric

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

import pytest

from elastic_gumby_universal_orch_agent_prototype.metrics.structural_metric import StructuralMetric


class TestBuildWorkflowTree:
    """Test _build_workflow_tree method."""

    @pytest.fixture
    def structural_metric(self):
        """Create StructuralMetric instance."""
        return StructuralMetric()

    def test_build_workflow_tree_simple_tool_call(self, structural_metric):
        """Test building tree from simple tool call workflow."""
        workflow = {
            "type": "tool_call",
            "toolName": "test_tool",
            "parameters": {"param1": "value1"}
        }

        tree = structural_metric._build_workflow_tree(workflow)

        assert tree["type"] == "tool_call"
        assert tree["children"] == []
        assert tree["depth"] == 0
        assert tree["tree_weight"] == 2.0  # Action node weight

    def test_build_workflow_tree_with_root_property(self, structural_metric):
        """Test building tree from workflow with root property."""
        workflow = {
            "root": {
                "type": "user_input",
                "prompt": "Enter your name"
            }
        }

        tree = structural_metric._build_workflow_tree(workflow)

        assert tree["type"] == "user_input"
        assert tree["children"] == []
        assert tree["depth"] == 0
        assert tree["tree_weight"] == 2.0  # Action node weight

    def test_build_workflow_tree_sequence(self, structural_metric):
        """Test building tree from sequence workflow."""
        workflow = {
            "type": "sequence",
            "steps": [
                {"type": "tool_call", "toolName": "tool1"},
                {"type": "user_input", "prompt": "test"}
            ]
        }

        tree = structural_metric._build_workflow_tree(workflow)

        assert tree["type"] == "sequence"
        assert len(tree["children"]) == 2
        assert tree["children"][0]["type"] == "tool_call"
        assert tree["children"][1]["type"] == "user_input"
        assert tree["depth"] == 0
        assert tree["tree_weight"] == 1.0  # Container node weight
        assert tree["children"][0]["depth"] == 1
        assert tree["children"][0]["tree_weight"] == 1.0  # 2.0 * (1/2) depth weight

    def test_build_workflow_tree_parallel(self, structural_metric):
        """Test building tree from parallel workflow."""
        workflow = {
            "type": "parallel",
            "branches": [
                {"type": "tool_call", "toolName": "tool1"},
                {"type": "tool_call", "toolName": "tool2"}
            ]
        }

        tree = structural_metric._build_workflow_tree(workflow)

        assert tree["type"] == "parallel"
        assert len(tree["children"]) == 2
        assert tree["children"][0]["type"] == "tool_call"
        assert tree["children"][1]["type"] == "tool_call"
        assert tree["depth"] == 0
        assert tree["tree_weight"] == 1.0  # Container node weight

    def test_build_workflow_tree_branch(self, structural_metric):
        """Test building tree from branch workflow."""
        workflow = {
            "type": "branch",
            "condition": {"type": "comparison", "left": "x", "right": "y"},
            "ifTrue": {"type": "tool_call", "toolName": "tool1"},
            "ifFalse": {"type": "tool_call", "toolName": "tool2"}
        }

        tree = structural_metric._build_workflow_tree(workflow)

        assert tree["type"] == "branch"
        assert len(tree["children"]) == 2
        assert tree["children"][0]["type"] == "tool_call"
        assert tree["children"][1]["type"] == "tool_call"
        assert tree["depth"] == 0
        assert tree["tree_weight"] == 2.0  # Action node weight

    def test_build_workflow_tree_loop(self, structural_metric):
        """Test building tree from loop workflow."""
        workflow = {
            "type": "loop",
            "condition": {"type": "comparison", "left": "counter", "right": "10"},
            "body": {"type": "tool_call", "toolName": "increment"}
        }

        tree = structural_metric._build_workflow_tree(workflow)

        assert tree["type"] == "loop"
        assert len(tree["children"]) == 1
        assert tree["children"][0]["type"] == "tool_call"
        assert tree["depth"] == 0
        assert tree["tree_weight"] == 2.0  # Action node weight

    def test_build_workflow_tree_wait_for_event(self, structural_metric):
        """Test building tree from wait_for_event workflow."""
        workflow = {
            "type": "wait_for_event",
            "eventSource": "user_action",
            "onTimeout": {"type": "tool_call", "toolName": "timeout_handler"}
        }

        tree = structural_metric._build_workflow_tree(workflow)

        assert tree["type"] == "wait_for_event"
        assert len(tree["children"]) == 1
        assert tree["children"][0]["type"] == "tool_call"
        assert tree["depth"] == 0
        assert tree["tree_weight"] == 2.0  # Action node weight

    def test_build_workflow_tree_wait_for_event_no_timeout(self, structural_metric):
        """Test building tree from wait_for_event workflow without timeout."""
        workflow = {
            "type": "wait_for_event",
            "eventSource": "user_action"
        }

        tree = structural_metric._build_workflow_tree(workflow)

        assert tree["type"] == "wait_for_event"
        assert len(tree["children"]) == 0
        assert tree["depth"] == 0
        assert tree["tree_weight"] == 2.0  # Action node weight

    def test_build_workflow_tree_unknown_type(self, structural_metric):
        """Test building tree from workflow with unknown type."""
        workflow = {
            "type": "unknown_type"
        }

        tree = structural_metric._build_workflow_tree(workflow)

        assert tree["type"] == "unknown_type"
        assert tree["children"] == []
        assert tree["depth"] == 0
        assert tree["tree_weight"] == 1.0  # Default weight for unknown types

    def test_build_workflow_tree_missing_type(self, structural_metric):
        """Test building tree from workflow with missing type."""
        workflow = {}

        tree = structural_metric._build_workflow_tree(workflow)

        assert tree["type"] == "unknown"
        assert tree["children"] == []
        assert tree["depth"] == 0
        assert tree["tree_weight"] == 1.0  # Default weight

    def test_build_workflow_tree_depth_weighting(self, structural_metric):
        """Test depth weighting in tree building."""
        workflow = {
            "type": "sequence",
            "steps": [
                {
                    "type": "sequence",
                    "steps": [
                        {"type": "tool_call", "toolName": "deep_tool"}
                    ]
                }
            ]
        }

        tree = structural_metric._build_workflow_tree(workflow)

        # Root sequence: depth 0, weight 1.0 * 1.0 = 1.0
        assert tree["depth"] == 0
        assert tree["tree_weight"] == 1.0

        # Nested sequence: depth 1, weight 1.0 * 0.5 = 0.5
        nested_seq = tree["children"][0]
        assert nested_seq["depth"] == 1
        assert nested_seq["tree_weight"] == 0.5

        # Deep tool call: depth 2, weight 2.0 * (1/3) ≈ 0.667
        deep_tool = nested_seq["children"][0]
        assert deep_tool["depth"] == 2
        assert abs(deep_tool["tree_weight"] - (2.0 / 3.0)) < 0.001


class TestGetSubtrees:
    """Test _get_subtrees method."""

    @pytest.fixture
    def structural_metric(self):
        """Create StructuralMetric instance."""
        return StructuralMetric()

    def test_get_subtrees_single_node(self, structural_metric):
        """Test getting subtrees from single node."""
        tree = {
            "type": "tool_call",
            "children": [],
            "depth": 0,
            "tree_weight": 2.0
        }

        subtrees = structural_metric._get_subtrees(tree)

        assert len(subtrees) == 1
        assert subtrees[0] == tree

    def test_get_subtrees_with_children(self, structural_metric):
        """Test getting subtrees from tree with children."""
        tree = {
            "type": "sequence",
            "children": [
                {"type": "tool_call", "children": [], "depth": 1, "tree_weight": 1.0},
                {"type": "user_input", "children": [], "depth": 1, "tree_weight": 1.0}
            ],
            "depth": 0,
            "tree_weight": 1.0
        }

        subtrees = structural_metric._get_subtrees(tree)

        assert len(subtrees) == 3  # Root + 2 children
        assert subtrees[0]["type"] == "sequence"
        assert subtrees[1]["type"] == "tool_call"
        assert subtrees[2]["type"] == "user_input"

    def test_get_subtrees_nested_structure(self, structural_metric):
        """Test getting subtrees from nested structure."""
        tree = {
            "type": "sequence",
            "children": [
                {
                    "type": "parallel",
                    "children": [
                        {"type": "tool_call", "children": [], "depth": 2, "tree_weight": 0.67}
                    ],
                    "depth": 1,
                    "tree_weight": 0.5
                }
            ],
            "depth": 0,
            "tree_weight": 1.0
        }

        subtrees = structural_metric._get_subtrees(tree)

        assert len(subtrees) == 3  # sequence + parallel + tool_call
        assert subtrees[0]["type"] == "sequence"
        assert subtrees[1]["type"] == "parallel"
        assert subtrees[2]["type"] == "tool_call"


class TestSubtreesMatch:
    """Test _subtrees_match method."""

    @pytest.fixture
    def structural_metric(self):
        """Create StructuralMetric instance."""
        return StructuralMetric()

    def test_subtrees_match_same_type_no_children(self, structural_metric):
        """Test matching subtrees with same type and no children."""
        tree1 = {"type": "tool_call", "children": []}
        tree2 = {"type": "tool_call", "children": []}

        assert structural_metric._subtrees_match(tree1, tree2) is True

    def test_subtrees_match_different_type(self, structural_metric):
        """Test matching subtrees with different types."""
        tree1 = {"type": "tool_call", "children": []}
        tree2 = {"type": "user_input", "children": []}

        assert structural_metric._subtrees_match(tree1, tree2) is False

    def test_subtrees_match_different_children_count(self, structural_metric):
        """Test matching subtrees with different children count."""
        tree1 = {
            "type": "sequence",
            "children": [{"type": "tool_call"}]
        }
        tree2 = {
            "type": "sequence",
            "children": [{"type": "tool_call"}, {"type": "user_input"}]
        }

        assert structural_metric._subtrees_match(tree1, tree2) is False

    def test_subtrees_match_same_children_types_same_order(self, structural_metric):
        """Test matching subtrees with same children types in same order."""
        tree1 = {
            "type": "sequence",
            "children": [
                {"type": "tool_call", "children": []},
                {"type": "user_input", "children": []}
            ]
        }
        tree2 = {
            "type": "sequence",
            "children": [
                {"type": "tool_call", "children": []},
                {"type": "user_input", "children": []}
            ]
        }

        assert structural_metric._subtrees_match(tree1, tree2) is True

    def test_subtrees_match_same_children_types_different_order(self, structural_metric):
        """Test matching subtrees with same children types in different order."""
        tree1 = {
            "type": "sequence",
            "children": [
                {"type": "tool_call", "children": []},
                {"type": "user_input", "children": []}
            ]
        }
        tree2 = {
            "type": "sequence",
            "children": [
                {"type": "user_input", "children": []},
                {"type": "tool_call", "children": []}
            ]
        }

        assert structural_metric._subtrees_match(tree1, tree2) is False

    def test_subtrees_match_missing_children_key(self, structural_metric):
        """Test matching subtrees with missing children key."""
        tree1 = {"type": "tool_call"}
        tree2 = {"type": "tool_call", "children": []}

        assert structural_metric._subtrees_match(tree1, tree2) is True


class TestWorkflowStructuralAnalysis:
    """Test workflow_structural_analysis method."""

    @pytest.fixture
    def structural_metric(self):
        """Create StructuralMetric instance."""
        return StructuralMetric()

    def test_workflow_structural_analysis_identical_workflows(self, structural_metric):
        """Test analysis of identical workflows."""
        workflow = {
            "type": "sequence",
            "steps": [
                {"type": "tool_call", "toolName": "tool1"},
                {"type": "user_input", "prompt": "test"}
            ]
        }

        result = structural_metric.workflow_structural_analysis(workflow, workflow)

        assert result["subtree_match_ratio"] == 1.0
        assert result["weighted_subtree_match_accuracy"] == 1.0
        assert result["match_count"] == 3  # sequence + 2 children
        assert result["miss_count"] == 0
        assert result["total_subtrees"] == 3

    def test_workflow_structural_analysis_completely_different_workflows(self, structural_metric):
        """Test analysis of completely different workflows."""
        generated = {"type": "tool_call", "toolName": "tool1"}
        reference = {"type": "user_input", "prompt": "test"}

        result = structural_metric.workflow_structural_analysis(generated, reference)

        assert result["subtree_match_ratio"] == 0.0
        assert result["weighted_subtree_match_accuracy"] == 0.0
        assert result["match_count"] == 0
        assert result["miss_count"] == 1
        assert result["total_subtrees"] == 1

    def test_workflow_structural_analysis_partial_match(self, structural_metric):
        """Test analysis of partially matching workflows."""
        generated = {
            "type": "sequence",
            "steps": [
                {"type": "tool_call", "toolName": "tool1"}
            ]
        }
        reference = {
            "type": "sequence",
            "steps": [
                {"type": "tool_call", "toolName": "tool1"},
                {"type": "user_input", "prompt": "test"}
            ]
        }

        result = structural_metric.workflow_structural_analysis(generated, reference)

        # Only tool_call matches, sequence and user_input don't match due to different structure
        assert abs(result["subtree_match_ratio"] - (1/3)) < 0.001  # Only 1 out of 3 subtrees match
        assert result["match_count"] == 1
        assert result["miss_count"] == 2
        assert result["total_subtrees"] == 3

    def test_workflow_structural_analysis_empty_reference(self, structural_metric):
        """Test analysis with empty reference workflow."""
        generated = {"type": "tool_call", "toolName": "tool1"}
        reference = {}

        result = structural_metric.workflow_structural_analysis(generated, reference)

        # Empty reference creates a subtree with "unknown" type
        assert result["subtree_match_ratio"] == 0.0  # No matches with unknown type
        assert result["weighted_subtree_match_accuracy"] == 0.0  # No reference weight
        assert result["match_count"] == 0
        assert result["miss_count"] == 1  # The unknown type reference subtree
        assert result["total_subtrees"] == 1

    def test_workflow_structural_analysis_detailed_breakdown(self, structural_metric):
        """Test detailed breakdown in analysis results."""
        generated = {
            "type": "sequence",
            "steps": [
                {"type": "tool_call", "toolName": "tool1"}
            ]
        }
        reference = {
            "type": "sequence",
            "steps": [
                {"type": "tool_call", "toolName": "tool1"},
                {"type": "user_input", "prompt": "test"},
                {"type": "parallel", "branches": [
                    {"type": "tool_call", "toolName": "tool2"}
                ]}
            ]
        }

        result = structural_metric.workflow_structural_analysis(generated, reference)

        # Based on actual behavior: 2 tool_calls match, but structures don't match exactly
        breakdown = result["detailed_breakdown"]
        assert breakdown["action_nodes_matched"] == 2  # Both tool_calls match
        assert breakdown["container_nodes_matched"] == 0  # Sequence and parallel don't match due to different structure
        assert breakdown["action_nodes_missing"] == 1  # user_input missing
        assert breakdown["container_nodes_missing"] == 2  # sequence and parallel missing
        assert breakdown["deep_structures_matched"] == 1  # One tool_call at depth ≥2 matched
        assert breakdown["shallow_structures_matched"] == 1  # One tool_call at depth <2 matched

    def test_workflow_structural_analysis_matches_and_misses_details(self, structural_metric):
        """Test detailed matches and misses information."""
        generated = {"type": "tool_call", "toolName": "tool1"}
        reference = {
            "type": "sequence",
            "steps": [
                {"type": "tool_call", "toolName": "tool1"},
                {"type": "user_input", "prompt": "test"}
            ]
        }

        result = structural_metric.workflow_structural_analysis(generated, reference)

        # Check matches details
        assert len(result["matches"]) == 1
        match = result["matches"][0]
        assert match["type"] == "tool_call"
        assert match["depth"] == 1  # In reference
        assert match["matched_depth"] == 0  # In generated
        assert match["depth_difference"] == 1
        assert match["children_count"] == 0

        # Check misses details
        assert len(result["misses"]) == 2  # sequence and user_input
        sequence_miss = next(m for m in result["misses"] if m["type"] == "sequence")
        assert sequence_miss["depth"] == 0
        assert sequence_miss["children_count"] == 2
        assert "tool_call" in sequence_miss["children_types"]
        assert "user_input" in sequence_miss["children_types"]

        user_input_miss = next(m for m in result["misses"] if m["type"] == "user_input")
        assert user_input_miss["depth"] == 1
        assert user_input_miss["children_count"] == 0

    def test_workflow_structural_analysis_with_root_property(self, structural_metric):
        """Test analysis with workflows containing root property."""
        generated = {
            "root": {"type": "tool_call", "toolName": "tool1"}
        }
        reference = {
            "root": {"type": "tool_call", "toolName": "tool1"}
        }

        result = structural_metric.workflow_structural_analysis(generated, reference)

        assert result["subtree_match_ratio"] == 1.0
        assert result["weighted_subtree_match_accuracy"] == 1.0
        assert result["match_count"] == 1
        assert result["miss_count"] == 0
