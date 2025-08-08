"""
Bedrock Client Manager

Handles AWS Bedrock client initialization, load balancing, and usage tracking
for the workflow planner agent.
"""

import json
import time

import boto3
from colorama import Fore, Style


class BedrockClientManager:
    """
    Manages multiple Bedrock clients across different regions for load balancing
    and tracks usage statistics.
    """

    def __init__(self):
        """Initialize the Bedrock client manager."""
        self.bedrock_clients = self._initialize_bedrock_clients()
        self.client_usage = self._initialize_client_usage()

    def _initialize_bedrock_clients(self):
        """
        Initialize multiple Bedrock clients across different regions for load balancing.

        Returns:
            dict: Dictionary of region -> bedrock client mappings
        """
        # List of AWS regions to initialize clients for claude-3-7-sonnet
        regions = ["us-east-1", "us-east-2", "us-west-2"]

        clients = {}
        successful_regions = []

        for region in regions:
            try:
                client = boto3.client(
                    "bedrock-runtime",
                    region_name=region,
                    config=boto3.session.Config(
                        retries={"max_attempts": 3, "mode": "adaptive"},
                        read_timeout=300,
                        connect_timeout=60,
                    ),
                )
                clients[region] = client
                successful_regions.append(region)
                print(
                    f"{Fore.GREEN}✓ Initialized Bedrock client for {region}{Style.RESET_ALL}"
                )
            except Exception as e:
                print(
                    f"{Fore.YELLOW}Warning: Failed to initialize Bedrock client for {region}: {e}{Style.RESET_ALL}"
                )

        if not clients:
            raise Exception("Failed to initialize any Bedrock clients")

        return clients

    def _initialize_client_usage(self):
        """
        Initialize usage tracking for each client focusing on rate limits.

        Claude 3.7 Sonnet limits per region:
        - 6 requests per minute
        - 20,000 tokens per minute

        Returns:
            dict: Dictionary tracking current minute usage per region
        """
        usage = {}
        for region in self.bedrock_clients.keys():
            usage[region] = {
                "current_minute_start": time.time(),
                "requests_this_minute": 0,
                "tokens_this_minute": 0,
                "total_requests": 0,
                "total_tokens": 0,
                "errors": 0,
            }
        return usage

    def select_best_client(self):
        """
        Select the Bedrock client with the most available tokens in the current minute.

        Claude 3.7 Sonnet limits per region:
        - 6 requests per minute
        - 20,000 tokens per minute

        If all regions are exhausted, waits until the earliest region resets.

        Returns:
            tuple: (selected_region, bedrock_client)
        """

        current_time = time.time()
        best_region = None
        max_available_tokens = -1
        earliest_reset_time = float("inf")
        earliest_region = None

        for region, client in self.bedrock_clients.items():
            usage = self.client_usage[region]

            # Reset counters if we've moved to a new minute
            if current_time - usage["current_minute_start"] >= 60:
                usage["current_minute_start"] = current_time
                usage["requests_this_minute"] = 0
                usage["tokens_this_minute"] = 0

            # Calculate available tokens
            available_tokens = 20000 - usage["tokens_this_minute"]

            # Choose client with most available tokens (including those with 0 available for comparison)
            if available_tokens > max_available_tokens:
                max_available_tokens = available_tokens
                best_region = region

            # Track which region will reset earliest (for fallback)
            reset_time = usage["current_minute_start"] + 60
            if reset_time < earliest_reset_time:
                earliest_reset_time = reset_time
                earliest_region = region

        # If all regions are exhausted (max_available_tokens <= 0), wait for the earliest one to reset
        if max_available_tokens <= 0:
            print(
                f"{Fore.YELLOW}All regions exhausted, waiting for rate limit reset...{Style.RESET_ALL}"
            )

            # Wait until the earliest region resets
            wait_time = max(0, earliest_reset_time - current_time)
            if wait_time > 0:
                print(
                    f"{Fore.YELLOW}Waiting {wait_time:.1f} seconds for {earliest_region} to reset...{Style.RESET_ALL}"
                )
                time.sleep(wait_time)

                # Reset the counters for the region that just reset
                usage = self.client_usage[earliest_region]
                usage["current_minute_start"] = time.time()
                usage["requests_this_minute"] = 0
                usage["tokens_this_minute"] = 0

            best_region = earliest_region

        return best_region, self.bedrock_clients[best_region]

    def update_client_usage(self, region, response_metadata):
        """
        Update usage statistics for a client after a request.

        Args:
            region (str): The region that was used
            response_metadata (dict): Response metadata from Bedrock containing actual token usage
        """
        if region not in self.client_usage:
            return

        usage = self.client_usage[region]

        # Extract actual token usage from response metadata
        actual_tokens = 0
        if response_metadata and "usage" in response_metadata:
            input_tokens = response_metadata["usage"].get("input_tokens", 0)
            output_tokens = response_metadata["usage"].get("output_tokens", 0)
            actual_tokens = input_tokens + output_tokens

        # Update current minute counters
        usage["requests_this_minute"] += 1
        usage["tokens_this_minute"] += actual_tokens

        # Update total counters
        usage["total_requests"] += 1
        usage["total_tokens"] += actual_tokens

    def invoke_model(
        self,
        messages,
        system_prompt=None,
        max_tokens=4000,
        tools=None,
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    ):
        """
        Invoke Claude model via Bedrock using the client with most available tokens.

        Args:
            messages (list): List of conversation messages
            system_prompt (str): System prompt for the model
            max_tokens (int): Maximum tokens to generate
            tools (list): Available tools for the model
            model_id (str): Bedrock model ID to use

        Returns:
            dict: Model response or error information
        """
        # Select best client (no token estimation needed)
        try:
            selected_region, bedrock_client = self.select_best_client()
            print(f"{Fore.CYAN}Using Bedrock client in region: {selected_region} ")
        except Exception as e:
            return {"error": f"Failed to select Bedrock client: {str(e)}", "region": "unknown"}

        # Prepare request body
        request_body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "anthropic_version": "bedrock-2023-05-31",
        }

        if system_prompt:
            request_body["system"] = system_prompt

        if tools:
            request_body["tools"] = tools

        # Make the request
        try:
            start_time = time.time()

            response = bedrock_client.invoke_model(
                modelId=model_id, body=json.dumps(request_body), contentType="application/json"
            )

            end_time = time.time()
            response_time = end_time - start_time

            # Parse response
            response_body = json.loads(response["body"].read())

            # Update usage statistics with actual token usage from response
            self.update_client_usage(selected_region, response_body)

            print(
                f"{Fore.GREEN}✓ Model invocation successful (took {response_time:.2f}s){Style.RESET_ALL}"
            )

            return response_body

        except Exception as e:

            self.client_usage[selected_region]["errors"] += 1

            error_msg = f"Error invoking model in {selected_region}: {str(e)}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return {"error": error_msg, "region": selected_region}
