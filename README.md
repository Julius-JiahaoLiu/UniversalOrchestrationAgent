# ElasticGumbyUniversalOrchAgentPrototype

A comprehensive Universal Transformation Orchestration Agent (UTOA) system for workflow orchestration, tool management, and execution with advanced planning, validation, reflection, and backup capabilities.

## Overview

The ElasticGumbyUniversalOrchAgentPrototype is an intelligent orchestration system that guides users through a three-phase workflow process:

1. **Phase 1: Tools Onboarding** - Collect and structure available tools/services
2. **Phase 2: Planning and Reflecting** - Generate and refine workflow plans using AI
3. **Phase 3: Transform Execution** - Convert workflows into executable AWS Step Functions

The system leverages AWS Bedrock for intelligent planning and provides comprehensive visualization, validation, and metrics capabilities.

For detailed design and architecture information, see the [Universal Transformation Orchestration Agent Project Document](https://quip-amazon.com/9tebAwm2bT1R/ATX-Intern-Project-Universal-Transformation-Orchestration-Agent-Project-Doc).

## Prerequisites

### AWS Credentials

You need temporary AWS credentials with access to AWS Bedrock. The easiest way to obtain these is through Isengard:

#### For bash/zsh users:
```bash
export ISENGARD_PRODUCTION_ACCOUNT=<false>
export AWS_ACCESS_KEY_ID=<your_access_key_id>
export AWS_SECRET_ACCESS_KEY=<your_secret_access_key>
export AWS_SESSION_TOKEN=<your_session_token>
```

**Note**: Replace the placeholder values with your actual credentials from Isengard. The credentials typically expire after a few hours, so need to refresh them periodically.

### Required Permissions

Your AWS credentials need the following permissions:
- `bedrock:InvokeModel` - For AI-powered planning and reflection

## Installation

1. Set up Brazil workspace and clone the package:
```bash
brazil ws use -p ElasticGumbyUniversalOrchAgentPrototype
cd src/ElasticGumbyUniversalOrchAgentPrototype
```

2. Install in development mode:
```bash
pip install -e .
```

## Usage

```bash
brazil-runtime-exec python -m elastic_gumby_universal_orch_agent_prototype
```
