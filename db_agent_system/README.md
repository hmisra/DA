# DB Agent System

<p align="center">
  <img src="docs/assets/dbagent-logo.png" alt="DB Agent System Logo" width="400"/>
</p>

<p align="center">
  <strong>A complete AI agent system for natural language querying of PostgreSQL databases</strong>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#system-architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#advanced-configuration">Advanced Configuration</a> •
  <a href="#development">Development</a> •
  <a href="#license">License</a>
</p>

---

## Overview

DB Agent System transforms natural language queries into precise SQL, executes them against PostgreSQL databases, and generates insightful analytics and reports. It integrates multiple AI frameworks (NVIDIA AgentIQ, LangChain, CrewAI, and Ollama) to create a robust, scalable solution with local LLM support.

```
User: "What were our top-selling products last quarter?"

DB Agent: 
- Identified intent: Product sales performance analysis for Q1 2023
- Generated SQL: SELECT p.name, SUM(oi.quantity) as units_sold, SUM(oi.subtotal) as revenue...
- Results: 15 rows retrieved from database
- Analysis: Electronics category dominated sales with Laptop Pro generating 38% of revenue
- Full report available at: results/20240320_product_sales_analysis.json
```

## Key Features

### Core Capabilities
- ✅ **Natural Language to SQL**: Convert plain language questions to optimized SQL queries
- ✅ **Schema Understanding**: Automatically analyze database structure to generate accurate queries
- ✅ **Query Validation**: Detect and correct SQL errors before execution
- ✅ **Result Analytics**: Extract meaningful insights from query results
- ✅ **Report Generation**: Create formatted, user-friendly reports

### Technical Highlights
- ✅ **Local LLM Support**: Run entirely locally using Ollama (no data leaves your environment)
- ✅ **Multi-Agent Architecture**: Specialized agents for SQL development, data analysis, and reporting
- ✅ **Comprehensive Error Handling**: Robust recovery from failures with detailed diagnostics
- ✅ **Observability**: Full tracing and performance metrics for all operations
- ✅ **Containerized**: Complete Docker setup for easy deployment

### Integration Options
- ✅ **AgentIQ Orchestration**: Function-based workflows with enhanced observability
- ✅ **Python API**: Programmatic access for integration with existing applications
- ✅ **Interactive UI**: Web interface for non-technical users (via AgentIQ serve)
- ✅ **CLI Interface**: Command line operations for scripts and automation

## System Architecture

<p align="center">
  <img src="docs/assets/architecture-diagram.png" alt="DB Agent System Architecture" width="800"/>
</p>

### Multi-Agent System

The DB Agent System employs a specialized team of AI agents, each with distinct responsibilities:

1. **Data Analyst Agent** 🔍
   - Understands user intent and requirements
   - Determines which tables and fields are relevant
   - Converts business questions to technical requirements

2. **SQL Developer Agent** 💻
   - Transforms requirements into optimized SQL
   - Validates query syntax and semantics 
   - Ensures database-specific compatibility

3. **Report Generator Agent** 📊
   - Interprets query results for business context
   - Identifies patterns, anomalies and insights
   - Creates formatted reports with visualizations

### Workflow Orchestration

The system is powered by NVIDIA AgentIQ's workflow architecture:

```
query input → intent analysis → schema retrieval → SQL generation → 
validation → execution → result analysis → report generation
```

Each step is implemented as an AgentIQ function with:
- Standardized inputs/outputs
- Detailed performance metrics
- Error handling and recovery
- Configurable LLM selection

## Quick Start

### Prerequisites

- Python 3.10+ 
- Docker and Docker Compose (for PostgreSQL)
- Ollama (for local LLMs) - [Install Guide](https://ollama.ai/download)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/db-agent-system.git
   cd db-agent-system
   ```

2. **Environment setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start Database**:
   ```bash
   docker-compose up -d postgres
   ```

5. **Install required Ollama models**:
   ```bash
   ollama pull llama3
   # Optional: for improved SQL generation
   ollama pull sqlcoder:15b
   ```

6. **Register Ollama plugin with AgentIQ**:
   ```bash
   python install_ollama_plugin.py
   ```

### Docker Deployment (Optional)

For a complete containerized setup:

```bash
docker-compose up -d
```

This starts:
- PostgreSQL database with sample data
- Ollama with required models
- DB Agent System application with web UI

## Usage

### CLI Interface

Query your database directly from command line:

```bash
python run_agentiq.py --query "How many orders were placed in the last month?"
```

Options:
- `--query TEXT`: Natural language query to process
- `--output PATH`: Save results to a specific file (default: results/timestamp_query.json)
- `--format [json|text|csv]`: Output format (default: json)
- `--verbose`: Show detailed processing logs

### Web Interface

Start the interactive web UI:

```bash
python run_agentiq.py --serve
```

Then open http://localhost:8000 in your browser.

### Python API

Integrate with your existing Python applications:

```python
from src.agentiq.workflow import create_db_query_workflow

# Initialize the workflow
workflow = create_db_query_workflow()

# Process a query
result = workflow.process_query("What are our top 5 selling products?")

# Access results
if result["success"]:
    print(f"SQL Query: {result['sql_query']}")
    print(f"Results: {result['results']}")
    print(f"Analysis: {result['report']}")
else:
    print(f"Error: {result['error']}")
```

## Advanced Configuration

### Custom Database Schema

For using with your own PostgreSQL database:

1. Update database connection settings in `.env`:
   ```
   DB_HOST=localhost
   DB_PORT=5432
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_NAME=your_database
   ```

2. If needed, add custom initialization scripts to `docker/init-scripts/`

### LLM Model Selection

Control which Ollama models are used for different tasks:

1. Edit the model selection in `.env`:
   ```
   OLLAMA_SQL_MODEL=sqlcoder:15b
   OLLAMA_ANALYSIS_MODEL=llama3
   OLLAMA_REPORT_MODEL=mistral:7b
   ```

2. Or customize model selection logic:
   ```python
   # src/ollama/llm.py
   def select_best_model(self, task_description: str) -> str:
       """Select the most appropriate model for a given task."""
       # Customize model selection logic
   ```

### AgentIQ Workflow Customization

Modify the workflow to add/change processing steps:

1. Edit `src/agentiq/workflow.py` to add new processing steps
2. Register additional functions in `src/agentiq/functions.py`
3. Update the workflow configuration in `configs/db_query_workflow.yaml`

## Development

### Project Structure

```
db_agent_system/
├── configs/              # AgentIQ workflow configurations
├── docker/               # Docker configuration and init scripts
├── docs/                 # Documentation and assets
├── logs/                 # Log files (generated at runtime)
├── results/              # Query results (generated at runtime)
├── src/                  # Source code
│   ├── agentiq/          # AgentIQ workflow and functions
│   ├── crewai/           # CrewAI agents and tasks
│   ├── langchain/        # LangChain database tools
│   ├── ollama/           # Ollama LLM integration
│   └── utils/            # Utilities (logging, config, etc.)
└── tests/                # Test suite
```

### Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/test_database_consolidated.py
pytest tests/test_ollama_consolidated.py
pytest tests/test_workflow_consolidated.py
```

### Extending the System

#### Adding New Agent Capabilities

1. Add a new agent in `src/crewai/agents.py`
2. Register the agent in `src/agentiq/workflow.py`
3. Create corresponding functions in `src/agentiq/functions.py`
4. Update the workflow configuration

#### Adding Database Support

The system can be extended to support other databases:

1. Create a new connection class in `src/langchain/database.py`
2. Implement database-specific tools
3. Update validation logic for the target database

## Performance Optimization

For better performance:

1. Use more capable Ollama models:
   ```bash
   ollama pull llama3:70b
   ollama pull deepseek-coder:33b
   ```

2. Configure system resources in `docker-compose.yml`:
   ```yaml
   ollama:
     deploy:
       resources:
         limits:
           cpus: '4'
           memory: 16G
   ```

3. Adjust model parameters in `.env`:
   ```
   OLLAMA_NUM_CTX=4096
   OLLAMA_NUM_THREAD=8
   ```

## License

Released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA AgentIQ for workflow orchestration
- LangChain for database tools
- CrewAI for multi-agent capabilities
- Ollama for local LLM support
- PostgreSQL for database functionality 