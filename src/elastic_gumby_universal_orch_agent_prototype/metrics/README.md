# Workflow Metrics

This directory contains modules for evaluating and comparing workflows defined in the workflow schema format. The metrics provide both structural and semantic analysis to determine how closely a generated workflow matches a reference workflow.

## Overview

The metrics module implements a comprehensive workflow comparison system with three main components:

- **`utils.py`** - Main interface providing the `compare_workflow()` method
- **`structural_metric.py`** - Analyzes workflow structure and node hierarchies  
- **`semantic_metric.py`** - Analyzes tool calls, parameters, and variable usage patterns

## Main Interface

### `compare_workflow(generated_workflow, reference_workflow)`

The primary entry point for workflow comparison, located in `utils.py`. This method:

- Combines structural and semantic analysis results
- Provides color-coded terminal output with visual formatting
- Returns a comprehensive formatted summary string
- Calculates overall assessment scores

## Structural Analysis (`StructuralMetric`)

The `StructuralMetric` class in `structural_metric.py` analyzes workflow structure through hierarchical tree comparison.

### Key Methods

#### `workflow_structural_analysis(generated_workflow, reference_workflow)`
Performs comprehensive structural comparison and returns detailed analysis including:

- **Subtree match ratio** - Percentage of reference subtrees found in generated workflow
- **Weighted subtree match accuracy** - Weight-adjusted matching score considering node importance
- **Detailed breakdown** - Analysis by node types (action vs container nodes, depth analysis)
- **Missing patterns** - Specific structural elements not found in generated workflow

### Implementation Details

- **Tree Building**: Converts workflows to hierarchical tree structures with depth tracking
- **Node Weighting**: Action nodes (tool_call, user_input, branch, loop, wait_for_event) weighted 2.0x, container nodes (sequence, parallel) weighted 1.0x
- **Depth Weighting**: Applies 1/(depth+1) weighting to emphasize top-level structures
- **Subtree Matching**: Compares node types and child structure patterns recursively

### Supported Node Types
- `sequence` - Sequential execution with steps
- `parallel` - Parallel execution with branches  
- `branch` - Conditional execution with ifTrue/ifFalse
- `loop` - Iterative execution with body
- `tool_call` - External tool invocation
- `user_input` - User interaction points
- `wait_for_event` - Event waiting with optional timeout handling

## Semantic Analysis (`SemanticMetric`)

The `SemanticMetric` class in `semantic_metric.py` analyzes workflow semantics through tool usage and data flow patterns.

### Key Methods

#### `workflow_semantic_analysis(generated_workflow, reference_workflow)`
Performs comprehensive semantic comparison combining:

- **Tool call analysis** - Similarity of tool usage patterns
- **Data flow analysis** - Variable definition and usage pattern matching

#### `tool_call_analysis(generated_workflow, reference_workflow)`
Analyzes tool call similarities including:

- **Tool-specific similarities** - Individual tool matching scores
- **Average tool call similarity** - Overall tool usage accuracy
- **Missing tools** - Tools present in reference but absent in generated workflow

#### `data_flow_analysis(generated_workflow, reference_workflow)`
Analyzes variable usage patterns including:

- **Variable definition similarity** - How variables are defined and scoped
- **Variable usage similarity** - Context and pattern matching for variable usage
- **Missing variables** - Variables defined in reference but not in generated workflow

### Implementation Details

- **Tool Call Extraction**: Traverses workflow tree to identify all tool invocations with path tracking
- **Parameter Matching**: Compares tool parameters with support for variable reference patterns (`${variable}`)
- **Variable Pattern Recognition**: Uses regex to identify `${variable}` patterns in strings
- **Context Analysis**: Tracks where variables are defined and used (tool parameters, conditions, prompts)
- **Similarity Scoring**: Uses Jaccard similarity for context matching and weighted scoring for parameters

### Variable Context Types
- `tool_call.{toolName}.{parameter}` - Tool parameter usage
- `condition.left/right` - Conditional expressions
- `user_input.prompt` - User prompts
- `wait_for_event.entityId` - Event entity references
