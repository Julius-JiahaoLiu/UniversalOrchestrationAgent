"""
Tests for BedrockClientManager

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

import json
import time
from unittest.mock import Mock, patch

import pytest

from elastic_gumby_universal_orch_agent_prototype.planner.bedrock_client_manager import (
    BedrockClientManager,
)


class TestBedrockClientManagerInitialization:
    """Test BedrockClientManager initialization."""

    @patch("boto3.client")
    def test_partial_initialization_failure(self, mock_boto_client):
        """Test initialization when some regions fail."""

        def side_effect(service_name, region_name, config):
            if region_name == "us-east-1":
                raise Exception("Region unavailable")
            return Mock()

        mock_boto_client.side_effect = side_effect

        manager = BedrockClientManager()

        # Should still initialize successfully with remaining regions
        assert len(manager.bedrock_clients) == 2
        assert "us-east-2" in manager.bedrock_clients
        assert "us-west-2" in manager.bedrock_clients
        assert "us-east-1" not in manager.bedrock_clients

    @patch("boto3.client")
    def test_complete_initialization_failure(self, mock_boto_client):
        """Test initialization when all regions fail."""
        mock_boto_client.side_effect = Exception("All regions unavailable")

        with pytest.raises(Exception, match="Failed to initialize any Bedrock clients"):
            BedrockClientManager()


class TestSelectBestClient:
    """Test select_best_client method."""

    @patch("boto3.client")
    def test_select_best_client_with_usage(self, mock_boto_client):
        """Test client selection based on token usage."""
        mock_boto_client.return_value = Mock()

        manager = BedrockClientManager()

        # Simulate different usage levels
        manager.client_usage["us-east-1"]["tokens_this_minute"] = 15000  # 5000 available
        manager.client_usage["us-east-2"]["tokens_this_minute"] = 8000  # 12000 available
        manager.client_usage["us-west-2"]["tokens_this_minute"] = 2000  # 18000 available

        region, client = manager.select_best_client()

        # Should select us-west-2 (most available tokens: 20000 - 2000 = 18000)
        assert region == "us-west-2"
        assert client == manager.bedrock_clients["us-west-2"]

    @patch("boto3.client")
    def test_select_best_client_minute_reset(self, mock_boto_client):
        """Test that counters reset after a minute."""
        mock_boto_client.return_value = Mock()

        manager = BedrockClientManager()

        # Set usage and simulate time passing
        region = "us-east-1"
        manager.client_usage[region]["tokens_this_minute"] = 15000
        manager.client_usage[region]["requests_this_minute"] = 5
        manager.client_usage[region]["current_minute_start"] = time.time() - 61  # 61 seconds ago

        selected_region, client = manager.select_best_client()

        # Counters should be reset
        assert manager.client_usage[region]["tokens_this_minute"] == 0
        assert manager.client_usage[region]["requests_this_minute"] == 0
        assert manager.client_usage[region]["current_minute_start"] > time.time() - 5

    @patch("boto3.client")
    @patch("time.sleep")
    def test_select_best_client_all_exhausted(self, mock_sleep, mock_boto_client):
        """Test client selection when all regions are at token limit."""
        mock_boto_client.return_value = Mock()

        manager = BedrockClientManager()

        current_time = time.time()

        # Set all regions to token limit with different start times
        manager.client_usage["us-east-1"]["tokens_this_minute"] = 20000
        manager.client_usage["us-east-1"]["current_minute_start"] = (
            current_time - 30
        )  # Will reset in 30s

        manager.client_usage["us-east-2"]["tokens_this_minute"] = 20000
        manager.client_usage["us-east-2"]["current_minute_start"] = (
            current_time - 45
        )  # Will reset in 15s

        manager.client_usage["us-west-2"]["tokens_this_minute"] = 20000
        manager.client_usage["us-west-2"]["current_minute_start"] = (
            current_time - 20
        )  # Will reset in 40s

        region, client = manager.select_best_client()

        # Should wait for us-east-2 (resets soonest) and then select it
        mock_sleep.assert_called_once()
        sleep_time = mock_sleep.call_args[0][0]
        assert 14 <= sleep_time <= 16  # Should wait ~15 seconds (with some tolerance for timing)

        assert region == "us-east-2"
        assert client == manager.bedrock_clients["us-east-2"]

        # Verify the selected region's counters were reset
        assert manager.client_usage["us-east-2"]["tokens_this_minute"] == 0
        assert manager.client_usage["us-east-2"]["requests_this_minute"] == 0

    @patch("boto3.client")
    @patch("time.sleep")
    def test_select_best_client_exhausted_no_wait_needed(self, mock_sleep, mock_boto_client):
        """Test client selection when a region has already naturally reset."""
        mock_boto_client.return_value = Mock()

        manager = BedrockClientManager()

        current_time = time.time()

        # Set all regions to token limit, but one has already passed the minute mark
        manager.client_usage["us-east-1"]["tokens_this_minute"] = 20000
        manager.client_usage["us-east-1"]["current_minute_start"] = current_time - 30

        manager.client_usage["us-east-2"]["tokens_this_minute"] = 20000
        manager.client_usage["us-east-2"]["current_minute_start"] = (
            current_time - 65
        )  # Already past 60s

        manager.client_usage["us-west-2"]["tokens_this_minute"] = 20000
        manager.client_usage["us-west-2"]["current_minute_start"] = current_time - 20

        region, client = manager.select_best_client()

        # Should not sleep since us-east-2 naturally resets
        mock_sleep.assert_not_called()

        assert region == "us-east-2"
        assert client == manager.bedrock_clients["us-east-2"]


class TestUpdateClientUsage:
    """Test update_client_usage method."""

    @patch("boto3.client")
    def test_update_client_usage_with_metadata(self, mock_boto_client):
        """Test updating usage with response metadata."""
        mock_boto_client.return_value = Mock()

        manager = BedrockClientManager()
        region = "us-east-1"

        response_metadata = {"usage": {"input_tokens": 100, "output_tokens": 50}}

        initial_requests = manager.client_usage[region]["total_requests"]
        initial_tokens = manager.client_usage[region]["total_tokens"]

        manager.update_client_usage(region, response_metadata)

        # Verify updates
        assert manager.client_usage[region]["total_requests"] == initial_requests + 1
        assert manager.client_usage[region]["total_tokens"] == initial_tokens + 150
        assert manager.client_usage[region]["requests_this_minute"] == 1
        assert manager.client_usage[region]["tokens_this_minute"] == 150

    @patch("boto3.client")
    def test_update_client_usage_without_metadata(self, mock_boto_client):
        """Test updating usage without response metadata."""
        mock_boto_client.return_value = Mock()

        manager = BedrockClientManager()
        region = "us-east-1"

        initial_requests = manager.client_usage[region]["total_requests"]
        initial_tokens = manager.client_usage[region]["total_tokens"]

        manager.update_client_usage(region, None)

        # Should still update request count
        assert manager.client_usage[region]["total_requests"] == initial_requests + 1
        assert manager.client_usage[region]["total_tokens"] == initial_tokens  # No change
        assert manager.client_usage[region]["requests_this_minute"] == 1
        assert manager.client_usage[region]["tokens_this_minute"] == 0

    @patch("boto3.client")
    def test_update_client_usage_invalid_region(self, mock_boto_client):
        """Test updating usage for non-existent region."""
        mock_boto_client.return_value = Mock()

        manager = BedrockClientManager()

        # Should not raise exception for invalid region
        manager.update_client_usage("invalid-region", {"usage": {"input_tokens": 10}})

        # Usage should remain unchanged for valid regions
        assert manager.client_usage["us-east-1"]["total_requests"] == 0


class TestInvokeModel:
    """Test invoke_model method."""

    @patch("boto3.client")
    def test_invoke_model_success(self, mock_boto_client):
        """Test successful model invocation."""
        mock_client = Mock()
        mock_response = {"body": Mock(), "ResponseMetadata": {"HTTPStatusCode": 200}}
        mock_response_body = {
            "content": [{"text": "Test response"}],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        mock_response["body"].read.return_value = json.dumps(mock_response_body)
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client

        manager = BedrockClientManager()
        messages = [{"role": "user", "content": "Test message"}]

        result = manager.invoke_model(messages)

        assert result == mock_response_body
        assert "error" not in result

        # Verify client was called correctly
        mock_client.invoke_model.assert_called_once()
        call_args = mock_client.invoke_model.call_args
        assert call_args[1]["modelId"] == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        assert call_args[1]["contentType"] == "application/json"

    @patch("boto3.client")
    def test_invoke_model_with_system_prompt(self, mock_boto_client):
        """Test model invocation with system prompt."""
        mock_client = Mock()
        mock_response = {"body": Mock(), "ResponseMetadata": {"HTTPStatusCode": 200}}
        mock_response_body = {"content": [{"text": "Test response"}]}
        mock_response["body"].read.return_value = json.dumps(mock_response_body)
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client

        manager = BedrockClientManager()
        messages = [{"role": "user", "content": "Test message"}]
        system_prompt = "You are a helpful assistant"

        result = manager.invoke_model(messages, system_prompt=system_prompt)

        # Verify system prompt was included in request
        call_args = mock_client.invoke_model.call_args
        request_body = json.loads(call_args[1]["body"])
        assert request_body["system"] == system_prompt

    @patch("boto3.client")
    def test_invoke_model_with_tools(self, mock_boto_client):
        """Test model invocation with tools."""
        mock_client = Mock()
        mock_response = {"body": Mock(), "ResponseMetadata": {"HTTPStatusCode": 200}}
        mock_response_body = {"content": [{"text": "Test response"}]}
        mock_response["body"].read.return_value = json.dumps(mock_response_body)
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client

        manager = BedrockClientManager()
        messages = [{"role": "user", "content": "Test message"}]
        tools = [{"name": "test_tool", "description": "A test tool"}]

        result = manager.invoke_model(messages, tools=tools)

        # Verify tools were included in request
        call_args = mock_client.invoke_model.call_args
        request_body = json.loads(call_args[1]["body"])
        assert request_body["tools"] == tools

    @patch("boto3.client")
    def test_invoke_model_client_selection_failure(self, mock_boto_client):
        """Test model invocation when client selection fails."""
        mock_boto_client.return_value = Mock()

        manager = BedrockClientManager()

        # Mock select_best_client to raise exception
        with patch.object(
            manager, "select_best_client", side_effect=Exception("No clients available")
        ):
            messages = [{"role": "user", "content": "Test message"}]
            result = manager.invoke_model(messages)

            assert "error" in result
            assert "Failed to select Bedrock client" in result["error"]
            assert result["region"] == "unknown"

    @patch("boto3.client")
    def test_invoke_model_bedrock_error(self, mock_boto_client):
        """Test model invocation when Bedrock call fails."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = Exception("Bedrock error")
        mock_boto_client.return_value = mock_client

        manager = BedrockClientManager()
        messages = [{"role": "user", "content": "Test message"}]

        result = manager.invoke_model(messages)

        assert "error" in result
        assert "Error invoking model" in result["error"]
        assert "region" in result

        # Verify error was recorded in client_usage
        region = result["region"]
        assert manager.client_usage[region]["errors"] == 1

    @patch("boto3.client")
    def test_invoke_model_custom_parameters(self, mock_boto_client):
        """Test model invocation with custom parameters."""
        mock_client = Mock()
        mock_response = {"body": Mock(), "ResponseMetadata": {"HTTPStatusCode": 200}}
        mock_response_body = {"content": [{"text": "Test response"}]}
        mock_response["body"].read.return_value = json.dumps(mock_response_body)
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client

        manager = BedrockClientManager()
        messages = [{"role": "user", "content": "Test message"}]

        result = manager.invoke_model(messages, max_tokens=8000, model_id="custom-model-id")

        # Verify custom parameters were used
        call_args = mock_client.invoke_model.call_args
        request_body = json.loads(call_args[1]["body"])
        assert request_body["max_tokens"] == 8000
        assert call_args[1]["modelId"] == "custom-model-id"
