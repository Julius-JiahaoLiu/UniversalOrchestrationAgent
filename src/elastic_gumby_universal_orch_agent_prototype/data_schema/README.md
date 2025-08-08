# Workflow Orchestration Schema Documentation

This folder contains JSON schemas that define the structure for workflow orchestration with LLMs. These schemas provide a standardized format for both input requirements and output workflow definitions, enabling LLMs to generate structured, executable workflows from natural language descriptions.

## Schema Files

1. **tools_schema.json**: Defines the structure of available tools input for workflow orchestration
2. **workflow_schema.json**: Defines the structure of the workflow that the LLM should generate
3. **state_machine_schema.json**: Defines the Amazon States Language schema specialized for JSONata expressions. (Just as ASL syntax reference for transform/state_machine_transformer.py)

## Schema Utilities Module

The `utils.py` module provides centralized access to all schemas through a clean API:

### Available Functions

- `get_workflow_schema()`: Returns the workflow orchestration schema
- `get_tools_schema()`: Returns the tools schema for available tools

## Tools Schema (tools_schema.json)

The tools schema defines the structure of available tools that can be used in workflow orchestration. It specifies how tools should be described to the LLM for workflow generation.

### Available Tools Structure

The schema defines an array of tool definitions, where each tool includes:

- **Name**: Identifier for the tool
- **Description**: What the tool does
- **Resource**: Amazon Resource Name (ARN) for the tool resource
- **Parameters**: What inputs the tool accepts
  - Name, type, description
  - Required/optional status
  - Default values and constraints
- **Return**: What the tool returns
  - Type and description
  - Optional JSON schema for complex return types

### Example Tools Input

```json
{
  "available_tools": [
    {
      "name": "sendNotification",
      "description": "Sends a notification to the user",
      "resource": "arn:aws:sns:us-east-1:123456789012:notification-topic",
      "parameters": [
        {
          "name": "message",
          "type": "string",
          "description": "Notification message content",
          "required": true
        }
      ],
      "return": {
        "type": "boolean",
        "description": "Whether the notification was sent successfully"
      }
    }
  ]
}
```

## Workflow Schema (workflow_schema.json)

The workflow schema defines the structure of the workflow that the LLM should generate. It uses a hierarchical tree structure with containers and nodes.

### Core Components

#### 1. Containers

Containers organize and control the execution flow:

- **SequenceContainer**: Executes steps in sequential order
- **ParallelContainer**: Executes branches concurrently

Containers can contain other containers or nodes, allowing for complex nested structures.

#### 2. Nodes

Nodes represent individual operations:

- **ToolCall**: Calls an external tool with parameters and stores the result
- **UserInput**: Requests and validates input from the user
- **Branch**: Conditional execution based on a structured condition
- **Loop**: Repeats execution while a condition is true
- **WaitForEvent**: Pauses workflow execution until a specific event occurs or times out

#### 3. Conditions

Structured representation of logical conditions:

- **Comparison Conditions**: Compare values using operators (==, !=, >, <, >=, <=, in, not_in)
- **Logical Conditions**: Combine conditions using logical operators (and, or, not)

### Workflow Structure

Every workflow has:

1. A name and description
2. A root container (either sequence or parallel)
3. A hierarchical structure of containers and nodes

### Example Workflow

```json
{
  "name": "Weather Alert Workflow",
  "description": "Checks weather and sends notifications for rain",
  "root": {
    "type": "sequence",
    "description": "Main workflow sequence",
    "steps": [
      {
        "type": "user_input",
        "prompt": "Please enter your city",
        "inputType": "text",
        "outputVariable": "userLocation"
      },
      {
        "type": "tool_call",
        "toolName": "getWeatherForecast",
        "parameters": {
          "location": "{% $userLocation %}",
          "days": 1
        },
        "outputVariable": "forecast"
      },
      {
        "type": "branch",
        "condition": {
          "type": "comparison",
          "left": "{% $forecast.precipitation %}",
          "operator": ">",
          "right": "0"
        },
        "ifTrue": {
          "type": "tool_call",
          "toolName": "sendNotification",
          "parameters": {
            "message": "{% 'Rain expected today in ' & $userLocation & '!' %}"
          },
          "outputVariable": "notificationResult"
        },
        "ifFalse": {
          "type": "tool_call",
          "toolName": "sendNotification",
          "parameters": {
            "message": "{% 'No rain expected today in ' & $userLocation & '.' %}"
          },
          "outputVariable": "notificationResult"
        }
      },
      {
        "type": "wait_for_event",
        "description": "Wait for user to read notification",
        "eventSource": "notification_service",
        "eventType": "notification_read",
        "entityId": "{% $notificationResult.notification_id %}",
        "timeout": 3600,
        "outputVariable": "readEvent",
        "onTimeout": {
          "type": "tool_call",
          "toolName": "sendReminder",
          "parameters": {
            "message": "Don't forget to check today's weather forecast!"
          },
          "outputVariable": "reminderResult"
        }
      }
    ]
  }
}
```