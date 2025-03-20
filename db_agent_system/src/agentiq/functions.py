"""
AgentIQ functions for database query processing.
These functions implement the steps of the database query workflow using AgentIQ.
"""
import time
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import decimal

# Import AgentIQ components
from aiq.cli.register_workflow import register_function
from aiq.builder.builder import Builder

# Import local components
from src.langchain.database import DatabaseConnection, get_db_tools
from src.ollama.llm import OllamaManager, get_ollama_llm
from src.utils.logging import AgentLogger

# Initialize logger
logger = AgentLogger(agent_name="AgentIQFunctions", agent_type="AgentIQ Functions")


@register_function()
async def parse_user_query(config: Dict[str, Any], builder: Builder):
    """Parse the user's natural language query to extract intent.
    
    Args:
        config: Function configuration
        builder: AgentIQ builder
        
    Returns:
        Dict containing the parsed query and intent
    """
    logger.info("Parsing user query with AgentIQ function")
    
    # Get the query from input
    query = builder.context.get("input", "")
    if not query:
        query = builder.context.get("query", "")
    
    logger.info(f"Analyzing query: {query}")
    
    # Get the LLM for intent analysis
    ollama_manager = OllamaManager()
    model_name = ollama_manager.select_best_model("Analyze user intent for database queries")
    llm = get_ollama_llm(model_name)
    
    # Create the prompt for intent analysis
    intent_prompt = f"""
    Analyze the following user query to understand their intent:

    USER QUERY: {query}

    Determine what database information they need and any implicit 
    requirements (e.g., time periods, sorting, grouping, etc.).
    The goal is to ensure we return exactly what the user is looking for,
    even if they didn't state it explicitly.
    
    Be specific and detailed in your analysis.
    """
    
    # Get the intent analysis from the LLM
    intent_result = await builder.llm.agenerate(intent_prompt)
    intent = intent_result.generations[0][0].text
    
    logger.info("User query parsed successfully")
    return {
        "original_query": query,
        "parsed_intent": intent,
    }


@register_function()
async def get_database_schema(config: Dict[str, Any], builder: Builder):
    """Get the database schema information.
    
    Args:
        config: Function configuration
        builder: AgentIQ builder
        
    Returns:
        Dict containing database schema information
    """
    logger.info("Getting database schema with AgentIQ function")
    
    # Get previous state from context
    previous_state = builder.context.get("result", {})
    
    # Initialize database connection
    db_conn = DatabaseConnection()
    
    # Get the list of tables
    tables = db_conn.get_table_names()
    logger.info(f"Found {len(tables)} tables in the database")
    
    # Get the schema for each table
    schema = {}
    for table in tables:
        schema[table] = db_conn.get_table_schema(table)
    
    logger.info("Database schema retrieved successfully")
    return {
        **previous_state,
        "db_schema": schema,
    }


@register_function()
async def generate_sql_query(config: Dict[str, Any], builder: Builder):
    """Generate an SQL query from the natural language query.
    
    Args:
        config: Function configuration
        builder: AgentIQ builder
        
    Returns:
        Dict containing the generated SQL query
    """
    logger.info("Generating SQL query with AgentIQ function")
    
    # Get previous state from context
    previous_state = builder.context.get("result", {})
    query = previous_state.get("original_query", "")
    intent = previous_state.get("parsed_intent", "")
    schema = previous_state.get("db_schema", {})
    
    # Format the schema for the prompt
    schema_str = _format_schema(schema)
    
    # Get the LLM for SQL generation
    ollama_manager = OllamaManager()
    model_name = ollama_manager.select_best_model("Convert natural language to SQL queries with precision")
    llm = get_ollama_llm(model_name)
    
    # Create the prompt for SQL generation
    sql_prompt = f"""
    Convert the following natural language query into a valid SQL query:

    QUERY: {query}
    
    USER INTENT: {intent}

    DATABASE SCHEMA:
    {schema_str}

    Return only the SQL query without any explanation. Ensure the SQL is valid,
    optimized, and accurately extracts the information requested in the query.
    """
    
    # Get the SQL query from the LLM
    sql_result = await builder.llm.agenerate(sql_prompt)
    sql_query = sql_result.generations[0][0].text
    
    # Clean the SQL query
    sql_query = _clean_sql_query(sql_query)
    
    logger.info(f"SQL query generated: {sql_query}")
    return {
        **previous_state,
        "sql_query": sql_query,
    }


@register_function()
async def validate_sql_query(config: Dict[str, Any], builder: Builder):
    """Validate the SQL query.
    
    Args:
        config: Function configuration
        builder: AgentIQ builder
        
    Returns:
        Dict containing validation results
    """
    logger.info("Validating SQL query with AgentIQ function")
    
    # Get previous state from context
    previous_state = builder.context.get("result", {})
    sql_query = previous_state.get("sql_query", "")
    
    # Initialize database tools
    db_tools = get_db_tools()
    
    # Use the validation tool
    validation_result = {"is_valid": False, "error": "Validation tool not found"}
    for tool in db_tools:
        if tool.name == "validate_sql_query":
            validation_result = tool._run(query=sql_query)
            break
    
    # If the query is invalid, attempt to fix it
    if not validation_result["is_valid"]:
        logger.warning(f"SQL validation failed: {validation_result['error']}")
        
        # Get the LLM for SQL fixing
        ollama_manager = OllamaManager()
        model_name = ollama_manager.select_best_model("Fix SQL query errors")
        llm = get_ollama_llm(model_name)
        
        # Create the prompt for SQL fixing
        query = previous_state.get("original_query", "")
        schema = previous_state.get("db_schema", {})
        schema_str = _format_schema(schema)
        
        fix_prompt = f"""
        Fix this SQL query that has the following error: {validation_result['error']}

        ORIGINAL QUERY: {query}
        SQL QUERY WITH ERROR: {sql_query}

        DATABASE SCHEMA:
        {schema_str}

        Return only the fixed SQL query, nothing else.
        """
        
        # Get the fixed SQL query from the LLM
        fix_result = await builder.llm.agenerate(fix_prompt)
        fixed_sql = fix_result.generations[0][0].text
        
        # Clean the fixed SQL query
        fixed_sql = _clean_sql_query(fixed_sql)
        
        logger.info(f"Fixed SQL query: {fixed_sql}")
        
        # Update the query_info
        previous_state["sql_query"] = fixed_sql
        previous_state["sql_query_fixed"] = True
        previous_state["original_validation_error"] = validation_result["error"]
        
        # Validate again
        for tool in db_tools:
            if tool.name == "validate_sql_query":
                validation_result = tool._run(query=fixed_sql)
                break
    
    logger.info(f"SQL validation result: {validation_result}")
    return {
        **previous_state,
        "validation_result": validation_result,
    }


@register_function()
async def execute_sql_query(config: Dict[str, Any], builder: Builder):
    """Execute the SQL query.
    
    Args:
        config: Function configuration
        builder: AgentIQ builder
        
    Returns:
        Dict containing query results
    """
    logger.info("Executing SQL query with AgentIQ function")
    
    # Get previous state from context
    previous_state = builder.context.get("result", {})
    
    # Check if the query is valid
    if not previous_state.get("validation_result", {}).get("is_valid", False):
        logger.error("Cannot execute invalid SQL query")
        return {
            **previous_state,
            "execution_result": {"error": "Invalid SQL query"},
            "execution_success": False,
        }
    
    # Extract the SQL query
    sql_query = previous_state.get("sql_query", "")
    
    # Initialize database connection
    db_conn = DatabaseConnection()
    
    # Execute the query
    try:
        start_time = time.time()
        rows, error = db_conn.execute_query(sql_query)
        execution_time = time.time() - start_time
        
        if error:
            logger.error(f"SQL execution error: {error}")
            return {
                **previous_state,
                "execution_result": {"error": error},
                "execution_success": False,
            }
        
        logger.info(f"SQL query executed successfully in {execution_time:.2f} seconds")
        return {
            **previous_state,
            "results": rows,
            "result_count": len(rows),
            "execution_time": execution_time,
            "execution_success": True,
        }
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        return {
            **previous_state,
            "execution_result": {"error": str(e)},
            "execution_success": False,
        }


@register_function()
async def analyze_query_results(config: Dict[str, Any], builder: Builder):
    """Analyze the query results.
    
    Args:
        config: Function configuration
        builder: AgentIQ builder
        
    Returns:
        Dict containing analysis results
    """
    logger.info("Analyzing query results with AgentIQ function")
    
    # Get previous state from context
    previous_state = builder.context.get("result", {})
    
    # Check if query execution was successful
    if not previous_state.get("execution_success", False):
        logger.error("Cannot analyze failed query execution")
        return {
            **previous_state,
            "analysis": "Query execution failed, no analysis possible.",
        }
    
    # Extract the necessary information
    query = previous_state.get("original_query", "")
    sql_query = previous_state.get("sql_query", "")
    results = previous_state.get("results", [])
    
    # Ensure results is a list of dictionaries
    if not isinstance(results, list):
        logger.warning(f"Expected results to be a list, got {type(results).__name__}")
        results = []
    
    # Format the results for the prompt
    results_str = _format_results(results)
    
    # Get the LLM for analysis
    ollama_manager = OllamaManager()
    model_name = ollama_manager.select_best_model("Analyze data and interpret results")
    llm = get_ollama_llm(model_name)
    
    # Create the prompt for analysis
    analysis_prompt = f"""
    Analyze the following query results to extract key insights:

    USER QUERY: {query}
    SQL QUERY: {sql_query}
    
    RESULTS:
    {results_str}

    Provide meaningful analysis of these results that directly answers
    the user's question. Include notable patterns, outliers, or important
    observations. Verify that the results correctly address the user's query.
    """
    
    # Get the analysis from the LLM
    analysis_result = await builder.llm.agenerate(analysis_prompt)
    analysis = analysis_result.generations[0][0].text
    
    logger.info("Query results analyzed successfully")
    return {
        **previous_state,
        "analysis": analysis,
    }


@register_function()
async def generate_report(config: Dict[str, Any], builder: Builder):
    """Generate a user-friendly report.
    
    Args:
        config: Function configuration
        builder: AgentIQ builder
        
    Returns:
        Dict containing the generated report
    """
    logger.info("Generating report with AgentIQ function")
    
    # Get previous state from context
    previous_state = builder.context.get("result", {})
    
    # Check if query execution was successful
    if not previous_state.get("execution_success", False):
        logger.error("Cannot generate report for failed query execution")
        return {
            **previous_state,
            "report": "Query execution failed, no report possible.",
        }
    
    # Extract the necessary information
    query = previous_state.get("original_query", "")
    sql_query = previous_state.get("sql_query", "")
    results = previous_state.get("results", [])
    analysis = previous_state.get("analysis", "No analysis available")
    
    # Format the results for the prompt
    results_summary = _format_results_summary(results)
    
    # Get the LLM for report generation
    ollama_manager = OllamaManager()
    model_name = ollama_manager.select_best_model("Generate clear and informative reports")
    llm = get_ollama_llm(model_name)
    
    # Create the prompt for report generation
    report_prompt = f"""
    Generate a comprehensive report based on the following:

    USER QUERY: {query}
    SQL QUERY: {sql_query}
    
    RESULTS SUMMARY: 
    {results_summary}
    
    ANALYSIS:
    {analysis}

    Create a well-structured report that:
    1. Clearly answers the user's question
    2. Provides context and background information
    3. Highlights key findings and insights
    4. Presents relevant data points to support conclusions
    5. Uses markdown formatting for readability
    
    Format the report with appropriate headings, bullet points, and emphasis.
    """
    
    # Get the report from the LLM
    report_result = await builder.llm.agenerate(report_prompt)
    report = report_result.generations[0][0].text
    
    # Save the final results
    _save_results({
        **previous_state,
        "report": report,
    })
    
    logger.info("Report generated successfully")
    return {
        **previous_state,
        "report": report,
    }


# Helper functions

def _format_schema(schema: Dict[str, Any]) -> str:
    """Format the schema information.

    Args:
        schema: Database schema information

    Returns:
        str: Formatted schema string
    """
    result = []
    
    for table_name, columns in schema.items():
        table_str = f"Table: {table_name}\nColumns:"
        
        for column in columns:
            col_str = f"  - {column['name']} ({column['type']})"
            if column.get('primary_key'):
                col_str += " PRIMARY KEY"
            table_str += f"\n{col_str}"
        
        result.append(table_str)
    
    return "\n\n".join(result)


def _clean_sql_query(sql_query: str) -> str:
    """Clean SQL query from markdown syntax and other formatting.
    
    Args:
        sql_query: The SQL query string, potentially with markdown formatting
        
    Returns:
        str: Cleaned SQL query
    """
    logger.info("Cleaning SQL query from markdown syntax")
    
    # Convert to string if needed
    if not isinstance(sql_query, str):
        sql_query = str(sql_query)
        
    # Remove markdown code block markers
    sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
    
    # Remove any "SQL query:" prefix that might be present
    prefixes = ["SQL query:", "SQL:", "Query:"]
    for prefix in prefixes:
        if sql_query.startswith(prefix):
            sql_query = sql_query[len(prefix):].strip()
            
    # Remove asterisks (sometimes used for emphasis)
    sql_query = sql_query.replace('*', '')
    
    # Remove extra newlines and ensure semicolon at the end
    sql_query = sql_query.strip()
    if sql_query and not sql_query.endswith(';'):
        sql_query += ';'
        
    logger.info(f"Cleaned SQL query: {sql_query}")
    return sql_query


def _format_results(results: List[Dict[str, Any]]) -> str:
    """Format the query results.

    Args:
        results: Query results from the database

    Returns:
        str: Formatted results string
    """
    if not results:
        return "No results returned."
    
    # Get column names from the first row
    columns = list(results[0].keys())
    
    # Build the header row
    header = " | ".join(columns)
    separator = "-" * len(header)
    
    # Build the data rows
    rows = []
    for result in results[:20]:  # Limit to 20 rows to avoid overly long prompts
        row_values = []
        for col in columns:
            val = result.get(col, "")
            # Convert to string and truncate if too long
            val_str = str(val)
            if len(val_str) > 50:
                val_str = val_str[:47] + "..."
            row_values.append(val_str)
        rows.append(" | ".join(row_values))
    
    # Add a note if we truncated the results
    if len(results) > 20:
        rows.append(f"... (showing 20 of {len(results)} total rows)")
    
    return f"{header}\n{separator}\n" + "\n".join(rows)


def _format_results_summary(results: List[Dict[str, Any]]) -> str:
    """Format a summary of the query results.

    Args:
        results: Query results from the database

    Returns:
        str: Formatted results summary
    """
    if not results:
        return "No results returned."
    
    # Get column names from the first row
    columns = list(results[0].keys())
    
    # Count total rows
    total_rows = len(results)
    
    # Sample a few rows
    sample_size = min(5, total_rows)
    sample = results[:sample_size]
    
    # Format the sample
    sample_str = _format_results(sample)
    
    return f"Total rows: {total_rows}\nColumns: {', '.join(columns)}\n\nSample rows:\n{sample_str}"


def _convert_decimal_to_float(obj: Any) -> None:
    """Recursively convert Decimal values to float in a nested structure.
    
    Args:
        obj: The object to process
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, decimal.Decimal):
                obj[key] = float(value)
            else:
                _convert_decimal_to_float(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, decimal.Decimal):
                obj[i] = float(item)
            else:
                _convert_decimal_to_float(item)


def _save_results(result: Dict[str, Any]) -> None:
    """Save the results to a file.

    Args:
        result: The query processing results
    """
    # Define a custom JSON encoder that can handle Decimal objects
    class DecimalEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, decimal.Decimal):
                return float(obj)
            return super().default(obj)
                
    output_file = Path("results/db_query_result.json")
    output_file.parent.mkdir(exist_ok=True)
    
    # Create a sanitized copy of the result for JSON serialization
    output_result = {
        "query": result.get("original_query", ""),
        "sql_query": result.get("sql_query", ""),
        "success": True,
        "processing_time_seconds": result.get("execution_time", 0),
        "result_count": result.get("result_count", 0),
    }
    
    output_result["report"] = result.get("report", "")
    # Only include a sample of results to keep file size reasonable
    output_result["results_sample"] = result.get("results", [])[:10]
    
    with open(output_file, "w") as f:
        json.dump(output_result, f, indent=2, cls=DecimalEncoder) 