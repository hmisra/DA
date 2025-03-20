# DB Agent System Architecture

## Overview
The DB Agent System is an AI-powered framework for querying PostgreSQL databases using natural language. It integrates multiple frameworks (NVIDIA AgentIQ, LangChain, CrewAI, and Ollama) to provide robust and scalable database querying capabilities.

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          DB Agent System                                   │
│                                                                           │
│  ┌───────────────┐       ┌───────────────┐      ┌───────────────┐         │
│  │ User Interface│       │ AgentIQ       │      │ CrewAI        │         │
│  │               │ ─────▶│ Workflow      │ ────▶│ Agents        │         │
│  │ Natural Lang  │       │ Orchestration │      │               │         │
│  └───────────────┘       └───────────────┘      └───────┬───────┘         │
│                                 │                        │                 │
│                                 │                        │                 │
│                          ┌──────▼────────┐       ┌──────▼────────┐        │
│                          │ LangChain     │◀──────│ Ollama LLM    │        │
│                          │ Tools & DB    │       │ Integration   │        │
│                          └───────┬───────┘       └───────────────┘        │
│                                  │                                        │
│                                  │                                        │
│                          ┌───────▼───────┐                                │
│                          │ PostgreSQL    │                                │
│                          │ Database      │                                │
│                          └───────────────┘                                │
└───────────────────────────────────────────────────────────────────────────┘
```

## Component Description

### 1. User Interface
- Accepts natural language queries from users
- Passes these queries to the AgentIQ Workflow component

### 2. AgentIQ Workflow
- The main orchestration layer that coordinates the entire query process
- Implements a sequential workflow with clearly defined steps
- Uses YAML configuration that can be used with the AgentIQ CLI
- Handles error recovery and result formatting

### 3. CrewAI Agents
- Specialized agents with distinct roles in the query process:
  - **SQL Developer Agent**: Converts natural language to SQL queries
  - **Data Analyst Agent**: Analyzes user intent and query results
  - **Report Generator Agent**: Creates user-friendly reports

### 4. Ollama LLM Integration
- Connects to locally running LLM models via Ollama
- Provides model selection based on task requirements
- Ensures consistent formatting and parameter passing

### 5. LangChain Tools & Database
- Database connection management
- Tools for database operations:
  - Listing tables
  - Getting schema information
  - Validating SQL queries
  - Executing SQL queries

### 6. PostgreSQL Database
- The target database system
- Stores the actual data to be queried

## Data Flow

1. **Query Input**: User submits a natural language query
2. **Intent Analysis**: The Data Analyst Agent determines the user's intent
3. **Schema Retrieval**: System fetches relevant database schema information
4. **SQL Generation**: SQL Developer Agent converts the natural language to SQL
5. **Query Validation**: System validates the generated SQL for correctness
6. **Query Execution**: System executes the SQL query against the database
7. **Result Analysis**: Data Analyst Agent analyzes the query results
8. **Report Generation**: Report Generator Agent creates a user-friendly report
9. **Response Delivery**: System returns the final report to the user

## Tool Adaptation Layer

A critical component is the tool adaptation layer that allows LangChain tools to work with CrewAI agents:

```
┌───────────────────┐     ┌───────────────────┐
│ LangChain Tools   │────▶│ Tool Adapter      │
│ (Database ops)    │     │ (Parameter        │
└───────────────────┘     │  conversion)      │
                          └─────────┬─────────┘
                                    │
                                    ▼
                          ┌───────────────────┐
                          │ CrewAI-compatible │
                          │ Tools             │
                          └───────────────────┘
```

This adapter handles:
- Tool registration with CrewAI
- Parameter conversion between frameworks
- Error handling and reporting

## Deployment Considerations

The system is designed to be deployed in various configurations:
- **Local Development**: Using local Ollama models
- **Production**: Can be containerized with Docker for scalability
- **Integration**: Can be integrated with existing systems via API

## Error Handling

The system includes robust error handling at multiple levels:
- Tool execution error recovery
- SQL query validation and auto-correction
- Timeout management for LLM operations
- Comprehensive logging for debugging

## Configuration

The system is configurable through:
- Environment variables for connection settings
- YAML files for workflow definition
- API parameters for runtime behavior 