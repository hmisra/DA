# Agentic AI Data Analysis System

An intelligent data analysis system that combines the power of LLMs (deepseek-r1:14b), Crew AI for agent orchestration, and advanced data analytics capabilities.

## Features

- Natural language query understanding using deepseek-r1:14b
- Automated SQL generation for PostgreSQL
- Advanced data analysis and insights generation
- Time series analysis and forecasting
- Data validation and quality checks
- Comprehensive test suite

## Prerequisites

- Python 3.9+
- PostgreSQL 13+
- Ollama (for running deepseek-r1:14b)
- Docker (optional, for containerized deployment)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agentic-ai-analysis.git
cd agentic-ai-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Ollama and install deepseek-r1:14b:
```bash
# Install Ollama (if not already installed)
curl https://ollama.ai/install.sh | sh

# Pull the deepseek-r1:14b model
ollama pull deepseek-r1:14b
```

5. Configure environment variables:
```bash
# Create .env file
cp .env.example .env

# Edit .env with your configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_user
DB_PASSWORD=your_password
```

## Project Structure

```
/agentic_ai/
  ├── crew_config/
  │   └── agents.yaml       # Crew AI agent configurations
  ├── ingestion/
  │   └── ingest_data.py    # Data ingestion utilities
  ├── analysis/
  │   └── analysis_agent.py # Analysis agent implementation
  ├── tools/
  │   ├── database.py       # Database connector
  │   └── transformer.py    # Data transformation utilities
  ├── tests/
  │   ├── test_agents.py    # Unit tests
  │   └── test_integration.py # Integration tests
  ├── main.py              # Main application entry point
  └── requirements.txt     # Project dependencies
```

## Usage

1. Start the system:
```bash
python main.py
```

2. Example queries:
```python
from analysis.analysis_agent import AnalysisAgent

# Initialize agent
agent = AnalysisAgent()

# Execute analysis
results = agent.execute_analysis(
    "What was the total revenue by product category last quarter?",
    "sales_data"
)

# Get insights
insights = agent.explain_results(results)
print(insights)
```

3. Data ingestion:
```python
from ingestion.ingest_data import DataIngestionManager

# Initialize manager
manager = DataIngestionManager()

# Ingest CSV data
manager.ingest_csv(
    "path/to/data.csv",
    "sales_data",
    if_exists="replace"
)
```

## Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_agents.py
pytest tests/test_integration.py -v

# Run tests with coverage
pytest --cov=.
```

## Configuration

### Database Setup

1. Create PostgreSQL database:
```sql
CREATE DATABASE your_database;
```

2. Create necessary tables:
```python
from tools.database import PostgresConnector

db = PostgresConnector()
db.create_table_schema(
    "sales_data",
    {
        "date": "timestamp",
        "product": "varchar(100)",
        "category": "varchar(50)",
        "revenue": "numeric",
        "units": "integer"
    }
)
```

### Crew AI Configuration

The `crew_config/agents.yaml` file defines:
- Agent roles and responsibilities
- Task definitions and workflows
- Tool configurations
- Flow orchestration

Example configuration:
```yaml
agents:
  - name: "QueryUnderstandingAgent"
    description: "Parses user queries using deepseek-r1:14b"
    tasks:
      - name: "parse_query"
        input_format: "text"
        output_format: "json"

tools:
  - name: "PostgresConnector"
    description: "Database operations"
    module: "tools.database"
    class: "PostgresConnector"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Crew AI team for the agent orchestration framework
- deepseek-r1:14b team for the language model
- Contributors and maintainers

## Support

For support, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with:
   - System information
   - Steps to reproduce
   - Expected vs actual behavior 