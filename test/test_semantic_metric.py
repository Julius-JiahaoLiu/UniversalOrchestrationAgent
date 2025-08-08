"""
Tests for SemanticMetric

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

import pytest

from elastic_gumby_universal_orch_agent_prototype.metrics.semantic_metric import SemanticMetric

class TestWorkflowSemanticAnalysis:
    """Test workflow_semantic_analysis method."""

    @pytest.fixture
    def semantic_metric(self):
        """Create SemanticMetric instance."""
        return SemanticMetric()

    def test_workflow_semantic_analysis_empty_workflows(self, semantic_metric):
        """Test semantic analysis with empty workflows."""
        generated_workflow = {"type": "sequence", "steps": []}
        reference_workflow = {"type": "sequence", "steps": []}

        result = semantic_metric.workflow_semantic_analysis(generated_workflow, reference_workflow)

        assert result["average_tool_call_similarity"] == 1.0
        assert result["missing_tools"] == []
        assert result["average_variable_definition_similarity"] == 1.0
        assert result["average_variable_usage_similarity"] == 1.0

    def test_workflow_semantic_analysis_with_variables(self, semantic_metric):
        """Test semantic analysis with variable definitions and usage."""
        generated_workflow = {
            "type": "sequence",
            "steps": [
                {
                    "type": "user_input",
                    "prompt": "Enter name",
                    "outputVariable": "userName"
                },
                {
                    "type": "tool_call",
                    "toolName": "greet_user",
                    "parameters": {"name": "${userName}"}
                }
            ]
        }
        reference_workflow = {
            "type": "sequence", 
            "steps": [
                {
                    "type": "user_input",
                    "prompt": "Enter name",
                    "outputVariable": "userName"
                },
                {
                    "type": "tool_call",
                    "toolName": "greet_user", 
                    "parameters": {"name": "${userName}"}
                }
            ]
        }

        result = semantic_metric.workflow_semantic_analysis(generated_workflow, reference_workflow)

        assert result["average_tool_call_similarity"] == 1.0
        assert result["average_variable_definition_similarity"] == 1.0
        assert result["average_variable_usage_similarity"] == 1.0


class TestToolCallAnalysis:
    """Test tool_call_analysis method."""

    @pytest.fixture
    def semantic_metric(self):
        """Create SemanticMetric instance."""
        return SemanticMetric()

    def test_tool_call_analysis_identical_tools(self, semantic_metric):
        """Test tool call analysis with identical tools."""
        generated_workflow = {
            "type": "tool_call",
            "toolName": "test_tool",
            "parameters": {"param1": "value1"}
        }
        reference_workflow = {
            "type": "tool_call",
            "toolName": "test_tool", 
            "parameters": {"param1": "value1"}
        }

        result = semantic_metric.tool_call_analysis(generated_workflow, reference_workflow)

        assert result["tool_call_similarity"]["test_tool"] == 1.0
        assert result["average_tool_call_similarity"] == 1.0
        assert result["missing_tools"] == []

    def test_tool_call_analysis_missing_tools(self, semantic_metric):
        """Test tool call analysis with missing tools."""
        generated_workflow = {"type": "sequence", "steps": []}
        reference_workflow = {
            "type": "tool_call",
            "toolName": "missing_tool",
            "parameters": {"param1": "value1"}
        }

        result = semantic_metric.tool_call_analysis(generated_workflow, reference_workflow)

        assert result["tool_call_similarity"]["missing_tool"] == 0.0
        assert result["average_tool_call_similarity"] == 0.0
        assert result["missing_tools"] == ["missing_tool"]

    def test_tool_call_analysis_no_reference_tools(self, semantic_metric):
        """Test tool call analysis with no reference tools."""
        generated_workflow = {
            "type": "tool_call",
            "toolName": "test_tool",
            "parameters": {"param1": "value1"}
        }
        reference_workflow = {"type": "sequence", "steps": []}

        result = semantic_metric.tool_call_analysis(generated_workflow, reference_workflow)

        assert result["tool_call_similarity"] == {}
        assert result["average_tool_call_similarity"] == 0.0
        assert result["missing_tools"] == []

    def test_tool_call_analysis_no_generated_tools(self, semantic_metric):
        """Test tool call analysis with no generated tools."""
        generated_workflow = {"type": "sequence", "steps": []}
        reference_workflow = {"type": "sequence", "steps": []}

        result = semantic_metric.tool_call_analysis(generated_workflow, reference_workflow)

        assert result["tool_call_similarity"] == {}
        assert result["average_tool_call_similarity"] == 1.0
        assert result["missing_tools"] == []


class TestExtractToolCalls:
    """Test _extract_tool_calls method."""

    @pytest.fixture
    def semantic_metric(self):
        """Create SemanticMetric instance."""
        return SemanticMetric()

    def test_extract_tool_calls_sequence(self, semantic_metric):
        """Test extracting tool calls from sequence workflow."""
        workflow = {
            "type": "sequence",
            "steps": [
                {
                    "type": "tool_call",
                    "toolName": "tool1",
                    "parameters": {"param1": "value1"}
                },
                {
                    "type": "tool_call", 
                    "toolName": "tool2",
                    "parameters": {"param2": "value2"}
                }
            ]
        }

        tool_calls = semantic_metric._extract_tool_calls(workflow)

        assert "tool1" in tool_calls
        assert "tool2" in tool_calls
        assert tool_calls["tool1"][0]["path"] == "root.sequence.0.tool_call"
        assert tool_calls["tool2"][0]["path"] == "root.sequence.1.tool_call"

    def test_extract_tool_calls_parallel(self, semantic_metric):
        """Test extracting tool calls from parallel workflow."""
        workflow = {
            "type": "parallel",
            "branches": [
                {
                    "type": "tool_call",
                    "toolName": "parallel_tool1",
                    "parameters": {"param1": "value1"}
                },
                {
                    "type": "tool_call",
                    "toolName": "parallel_tool2", 
                    "parameters": {"param2": "value2"}
                }
            ]
        }

        tool_calls = semantic_metric._extract_tool_calls(workflow)

        assert "parallel_tool1" in tool_calls
        assert "parallel_tool2" in tool_calls
        assert tool_calls["parallel_tool1"][0]["path"] == "root.parallel.0.tool_call"
        assert tool_calls["parallel_tool2"][0]["path"] == "root.parallel.1.tool_call"

    def test_extract_tool_calls_branch(self, semantic_metric):
        """Test extracting tool calls from branch workflow."""
        workflow = {
            "type": "branch",
            "condition": {"type": "comparison", "left": "x", "operator": "==", "right": "1"},
            "ifTrue": {
                "type": "tool_call",
                "toolName": "true_tool",
                "parameters": {"param1": "value1"}
            },
            "ifFalse": {
                "type": "tool_call",
                "toolName": "false_tool",
                "parameters": {"param2": "value2"}
            }
        }

        tool_calls = semantic_metric._extract_tool_calls(workflow)

        assert "true_tool" in tool_calls
        assert "false_tool" in tool_calls
        assert tool_calls["true_tool"][0]["path"] == "root.branch.true.tool_call"
        assert tool_calls["false_tool"][0]["path"] == "root.branch.false.tool_call"

    def test_extract_tool_calls_loop(self, semantic_metric):
        """Test extracting tool calls from loop workflow."""
        workflow = {
            "type": "loop",
            "condition": {"type": "comparison", "left": "i", "operator": "<", "right": "10"},
            "body": {
                "type": "tool_call",
                "toolName": "loop_tool",
                "parameters": {"param1": "value1"}
            }
        }

        tool_calls = semantic_metric._extract_tool_calls(workflow)

        assert "loop_tool" in tool_calls
        assert tool_calls["loop_tool"][0]["path"] == "root.loop.body.tool_call"

    def test_extract_tool_calls_wait_for_event_with_timeout(self, semantic_metric):
        """Test extracting tool calls from wait_for_event with timeout."""
        workflow = {
            "type": "wait_for_event",
            "eventSource": "user_action",
            "onTimeout": {
                "type": "tool_call",
                "toolName": "timeout_tool",
                "parameters": {"param1": "timeout"}
            }
        }

        tool_calls = semantic_metric._extract_tool_calls(workflow)

        assert "timeout_tool" in tool_calls
        assert tool_calls["timeout_tool"][0]["path"] == "root.wait_for_event.onTimeout.tool_call"

    def test_extract_tool_calls_multiple_same_tool(self, semantic_metric):
        """Test extracting multiple calls to the same tool."""
        workflow = {
            "type": "sequence",
            "steps": [
                {
                    "type": "tool_call",
                    "toolName": "repeated_tool",
                    "parameters": {"param1": "value1"}
                },
                {
                    "type": "tool_call",
                    "toolName": "repeated_tool",
                    "parameters": {"param1": "value2"}
                }
            ]
        }

        tool_calls = semantic_metric._extract_tool_calls(workflow)

        assert "repeated_tool" in tool_calls
        assert len(tool_calls["repeated_tool"]) == 2
        assert tool_calls["repeated_tool"][0]["parameters"] == {"param1": "value1"}
        assert tool_calls["repeated_tool"][1]["parameters"] == {"param1": "value2"}

    def test_extract_tool_calls_missing_tool_name(self, semantic_metric):
        """Test extracting tool calls with missing toolName."""
        workflow = {
            "type": "tool_call",
            "parameters": {"param1": "value1"}
        }

        tool_calls = semantic_metric._extract_tool_calls(workflow)

        assert tool_calls == {}

    def test_extract_tool_calls_nested_complex(self, semantic_metric):
        """Test extracting tool calls from complex nested workflow."""
        workflow = {
            "type": "sequence",
            "steps": [
                {
                    "type": "parallel",
                    "branches": [
                        {
                            "type": "branch",
                            "condition": {"type": "comparison", "left": "x", "operator": "==", "right": "1"},
                            "ifTrue": {
                                "type": "tool_call",
                                "toolName": "nested_tool",
                                "parameters": {"level": "deep"}
                            },
                            "ifFalse": {
                                "type": "user_input",
                                "prompt": "Enter value"
                            }
                        }
                    ]
                }
            ]
        }

        tool_calls = semantic_metric._extract_tool_calls(workflow)

        assert "nested_tool" in tool_calls
        assert tool_calls["nested_tool"][0]["path"] == "root.sequence.0.parallel.0.branch.true.tool_call"


class TestCalculateToolPathSimilarity:
    """Test _calculate_tool_path_similarity method."""

    @pytest.fixture
    def semantic_metric(self):
        """Create SemanticMetric instance."""
        return SemanticMetric()

    def test_calculate_tool_path_similarity_partial_match(self, semantic_metric):
        """Test path similarity with partial match."""
        path1 = "root.sequence.0.tool_call"
        path2 = "root.sequence.1.tool_call"

        similarity = semantic_metric._calculate_tool_path_similarity(path1, path2)

        assert 0.5 < similarity < 1.0

    def test_calculate_tool_path_similarity_one_empty(self, semantic_metric):
        """Test path similarity with one empty path."""
        path1 = "root.tool_call"
        path2 = ""

        similarity = semantic_metric._calculate_tool_path_similarity(path1, path2)

        assert similarity == 0.0


class TestCalculateToolParameterSimilarity:
    """Test _calculate_tool_parameter_similarity method."""

    @pytest.fixture
    def semantic_metric(self):
        """Create SemanticMetric instance."""
        return SemanticMetric()

    def test_calculate_tool_parameter_similarity_one_empty(self, semantic_metric):
        """Test parameter similarity with one empty parameter set."""
        params1 = {"param1": "value1"}
        params2 = {}

        similarity = semantic_metric._calculate_tool_parameter_similarity(params1, params2)

        assert similarity == 0.0

    def test_calculate_tool_parameter_similarity_variable_references(self, semantic_metric):
        """Test parameter similarity with variable references."""
        params1 = {"name": "${userName}", "greeting": "Hello"}
        params2 = {"name": "${userName}", "greeting": "Hello"}

        similarity = semantic_metric._calculate_tool_parameter_similarity(params1, params2)

        assert similarity == 1.0

    def test_calculate_tool_parameter_similarity_different_variables(self, semantic_metric):
        """Test parameter similarity with different variable references."""
        params1 = {"name": "${userName}"}
        params2 = {"name": "${userID}"}

        similarity = semantic_metric._calculate_tool_parameter_similarity(params1, params2)

        assert similarity == 0.0  # No variable intersection

    def test_calculate_tool_parameter_similarity_mixed_variables(self, semantic_metric):
        """Test parameter similarity with mixed variable and literal values."""
        params1 = {"name": "${userName}", "age": "25"}
        params2 = {"name": "John", "age": "25"}

        similarity = semantic_metric._calculate_tool_parameter_similarity(params1, params2)

        assert 0.5 < similarity < 1.0

    def test_calculate_tool_parameter_similarity_same_type_different_values(self, semantic_metric):
        """Test parameter similarity with same type but different values."""
        params1 = {"count": 5, "name": "test"}
        params2 = {"count": 10, "name": "demo"}

        similarity = semantic_metric._calculate_tool_parameter_similarity(params1, params2)

        assert similarity == 0.5  # Same types get 0.5 each, average is 0.5

    def test_calculate_tool_parameter_similarity_complex_variables(self, semantic_metric):
        """Test parameter similarity with complex variable patterns."""
        params1 = {"template": "Hello ${name}, you have ${count} messages"}
        params2 = {"template": "Hello ${name}, you have ${count} messages"}

        similarity = semantic_metric._calculate_tool_parameter_similarity(params1, params2)

        assert similarity == 1.0

class TestDataFlowAnalysis:
    """Test data_flow_analysis method."""

    @pytest.fixture
    def semantic_metric(self):
        """Create SemanticMetric instance."""
        return SemanticMetric()

    def test_data_flow_analysis_identical_workflows(self, semantic_metric):
        """Test data flow analysis with identical workflows."""
        workflow = {
            "type": "sequence",
            "steps": [
                {
                    "type": "user_input",
                    "prompt": "Enter name",
                    "outputVariable": "userName"
                },
                {
                    "type": "tool_call",
                    "toolName": "greet_user",
                    "parameters": {"name": "${userName}"}
                }
            ]
        }

        result = semantic_metric.data_flow_analysis(workflow, workflow)

        assert result["missing_variables"] == []
        assert result["average_variable_definition_similarity"] == 1.0
        assert result["variable_usage_similarity"]["userName"] == 1.0
        assert result["average_variable_usage_similarity"] == 1.0

    def test_data_flow_analysis_missing_variables(self, semantic_metric):
        """Test data flow analysis with missing variables."""
        generated_workflow = {
            "type": "tool_call",
            "toolName": "simple_tool",
            "parameters": {"param": "value"}
        }
        reference_workflow = {
            "type": "sequence",
            "steps": [
                {
                    "type": "user_input",
                    "prompt": "Enter name",
                    "outputVariable": "userName"
                },
                {
                    "type": "tool_call",
                    "toolName": "greet_user",
                    "parameters": {"name": "${userName}"}
                }
            ]
        }

        result = semantic_metric.data_flow_analysis(generated_workflow, reference_workflow)

        assert "user_input -> userName" in result["missing_variables"]
        assert result["average_variable_definition_similarity"] == 0.0

    def test_data_flow_analysis_partial_variable_match(self, semantic_metric):
        """Test data flow analysis with partial variable matches."""
        generated_workflow = {
            "type": "sequence",
            "steps": [
                {
                    "type": "user_input",
                    "prompt": "Enter name",
                    "outputVariable": "userName"
                },
                {
                    "type": "tool_call",
                    "toolName": "different_tool",
                    "parameters": {"name": "${userName}"}
                }
            ]
        }
        reference_workflow = {
            "type": "sequence",
            "steps": [
                {
                    "type": "wait_for_event",
                    "eventSource": "user_action",
                    "outputVariable": "userName"
                },
                {
                    "type": "tool_call",
                    "toolName": "greet_user",
                    "parameters": {"name": "${userName}"}
                }
            ]
        }

        result = semantic_metric.data_flow_analysis(generated_workflow, reference_workflow)

        assert result["average_variable_definition_similarity"] == 0.5  # Different contexts
        assert result["variable_usage_similarity"]["userName"] < 1.0  # Different usage contexts


class TestExtractVariables:
    """Test _extract_variables method."""

    @pytest.fixture
    def semantic_metric(self):
        """Create SemanticMetric instance."""
        return SemanticMetric()

    def test_extract_variables_tool_call_with_output(self, semantic_metric):
        """Test extracting variables from tool_call with output."""
        workflow = {
            "type": "tool_call",
            "toolName": "get_weather",
            "parameters": {"city": "Seattle"},
            "outputVariable": "weatherData"
        }

        definitions, usages = semantic_metric._extract_variables(workflow)

        assert definitions["weatherData"] == "tool_call.get_weather"
        assert usages == {}

    def test_extract_variables_wait_for_event_with_output(self, semantic_metric):
        """Test extracting variables from wait_for_event with output."""
        workflow = {
            "type": "wait_for_event",
            "eventSource": "user_action",
            "outputVariable": "eventData"
        }

        definitions, usages = semantic_metric._extract_variables(workflow)

        assert definitions["eventData"] == "wait_for_event.user_action"
        assert usages == {}

    def test_extract_variables_wait_for_event_with_variable_usage(self, semantic_metric):
        """Test extracting variable usage from wait_for_event entityId."""
        workflow = {
            "type": "wait_for_event",
            "eventSource": "database",
            "entityId": "${recordId}",
            "outputVariable": "dbResult"
        }

        definitions, usages = semantic_metric._extract_variables(workflow)

        assert definitions["dbResult"] == "wait_for_event.database"
        assert "recordId" in usages
        assert "wait_for_event.entityId" in usages["recordId"]

    def test_extract_variables_logical_condition(self, semantic_metric):
        """Test extracting variables from logical conditions."""
        workflow = {
            "type": "branch",
            "condition": {
                "type": "logical",
                "operator": "and",
                "conditions": [
                    {
                        "type": "comparison",
                        "left": "${userAge}",
                        "operator": ">",
                        "right": "18"
                    },
                    {
                        "type": "comparison", 
                        "left": "${userStatus}",
                        "operator": "==",
                        "right": "active"
                    }
                ]
            },
            "ifTrue": {"type": "user_input", "prompt": "Welcome!"},
            "ifFalse": {"type": "user_input", "prompt": "Access denied"}
        }

        definitions, usages = semantic_metric._extract_variables(workflow)

        assert definitions == {}
        assert "userAge" in usages
        assert "userStatus" in usages

    def test_extract_variables_sequence_workflow(self, semantic_metric):
        """Test extracting variables from sequence workflow."""
        workflow = {
            "type": "sequence",
            "steps": [
                {
                    "type": "user_input",
                    "prompt": "Enter name",
                    "outputVariable": "userName"
                },
                {
                    "type": "tool_call",
                    "toolName": "greet_user",
                    "parameters": {"name": "${userName}"},
                    "outputVariable": "greeting"
                },
                {
                    "type": "user_input",
                    "prompt": "${greeting}"
                }
            ]
        }

        definitions, usages = semantic_metric._extract_variables(workflow)

        assert "userName" in definitions
        assert "greeting" in definitions
        assert definitions["userName"] == "user_input"
        assert definitions["greeting"] == "tool_call.greet_user"
        assert "userName" in usages
        assert "greeting" in usages

    def test_extract_variables_multiple_variable_usages(self, semantic_metric):
        """Test extracting multiple usages of the same variable."""
        workflow = {
            "type": "sequence",
            "steps": [
                {
                    "type": "tool_call",
                    "toolName": "tool1",
                    "parameters": {"param1": "${sharedVar}"}
                },
                {
                    "type": "tool_call",
                    "toolName": "tool2", 
                    "parameters": {"param2": "${sharedVar}"}
                }
            ]
        }

        definitions, usages = semantic_metric._extract_variables(workflow)

        assert definitions == {}
        assert "sharedVar" in usages
        assert len(usages["sharedVar"]) == 2
        assert "tool_call.tool1.param1" in usages["sharedVar"]
        assert "tool_call.tool2.param2" in usages["sharedVar"]
