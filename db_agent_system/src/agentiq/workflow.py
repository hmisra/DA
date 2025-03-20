"""
NVIDIA AgentIQ workflow orchestrator.
Orchestrates the database query process using AgentIQ concepts.
"""
from typing import Dict, List, Any, Optional, Tuple
import time
import yaml
import os
import json
from pathlib import Path
import threading
import decimal

# We'll implement a workflow structure that follows AgentIQ patterns
# but works with our existing code
from pydantic import BaseModel, Field

from crewai import Crew, Process

from src.crewai.agents import (
    create_sql_developer_agent,
    create_data_analyst_agent,
    create_report_generator_agent,
)
from src.langchain.database import (
    get_db_tools,
    DatabaseConnection,
)
from src.ollama.llm import OllamaManager
from src.utils.logging import AgentLogger


# Custom exception for task timeouts
class TimeoutError(Exception):
    """Raised when a task execution times out."""
    pass


class DBQueryWorkflow:
    """Database Query Workflow using AgentIQ concepts."""

    def __init__(self):
        """Initialize the DBQueryWorkflow."""
        self.logger = AgentLogger(
            agent_name="DBQueryWorkflow", agent_type="AgentIQ Workflow"
        )
        self.logger.info("Initializing DB Query Workflow")
        
        # Initialize the database connection
        self.db_conn = DatabaseConnection()
        
        # Get the database tools
        self.db_tools = get_db_tools()
        
        # Initialize the CrewAI agents
        self.sql_developer = create_sql_developer_agent(self.db_tools)
        self.data_analyst = create_data_analyst_agent(self.db_tools)
        self.report_generator = create_report_generator_agent()
        
        # Create the workflow configuration
        self._create_workflow_config()
        
        self.logger.info("DB Query Workflow initialized")

    def _create_workflow_config(self) -> None:
        """Create the workflow configuration following AgentIQ patterns.
        This will create a config that can be used with the AgentIQ CLI.
        """
        self.logger.info("Creating workflow configuration")
        
        # Define the workflow steps as functions
        workflow_config = {
            "functions": {
                "parse_user_query": {
                    "_type": "function",
                    "description": "Parse the user's natural language query"
                },
                "get_database_schema": {
                    "_type": "function",
                    "description": "Get the database schema information"
                },
                "generate_sql_query": {
                    "_type": "function",
                    "description": "Generate an SQL query from the natural language query"
                },
                "validate_sql_query": {
                    "_type": "function",
                    "description": "Validate the SQL query"
                },
                "execute_sql_query": {
                    "_type": "function",
                    "description": "Execute the validated SQL query"
                },
                "analyze_query_results": {
                    "_type": "function",
                    "description": "Analyze the query results"
                },
                "generate_report": {
                    "_type": "function",
                    "description": "Generate a user-friendly report"
                }
            },
            "llms": {
                "ollama_llm": {
                    "_type": "ollama",
                    "model_name": "llama3",
                    "base_url": "http://localhost:11434",
                    "temperature": 0.2
                }
            },
            "workflow": {
                "_type": "sequential",
                "steps": [
                    {"function_name": "parse_user_query", "llm": "ollama_llm"},
                    {"function_name": "get_database_schema"},
                    {"function_name": "generate_sql_query", "llm": "ollama_llm"},
                    {"function_name": "validate_sql_query", "llm": "ollama_llm"},
                    {"function_name": "execute_sql_query"},
                    {"function_name": "analyze_query_results", "llm": "ollama_llm"},
                    {"function_name": "generate_report", "llm": "ollama_llm"}
                ],
                "verbose": True
            }
        }
        
        # Save the configuration to a file for potential use with the AgentIQ CLI
        config_dir = Path("configs")
        config_dir.mkdir(exist_ok=True)
        
        with open(config_dir / "db_query_workflow.yaml", "w") as f:
            yaml.dump(workflow_config, f)
        
        self.logger.info("Workflow configuration created")
        self.workflow_config = workflow_config

    def _execute_task(self, task):
        """Helper method to execute a CrewAI task properly.
        
        Creates a mini-Crew with just the task's agent and executes the task.
        
        Args:
            task: The CrewAI task to execute
            
        Returns:
            The result of the task execution
        """
        # Log detailed information about the task
        task_desc_short = task.description[:50] + "..." if len(task.description) > 50 else task.description
        self.logger.info(f"Executing task: {task_desc_short}")
        self.logger.info(f"Task agent: {task.agent.role}")
        
        # Log full task description for debugging
        self.logger.info(f"FULL TASK DESCRIPTION: {task.description}")
        
        # Get and log LLM info if available
        if hasattr(task.agent, 'llm') and hasattr(task.agent.llm, 'model'):
            self.logger.info(f"Task agent model: {task.agent.llm.model}")
        
        self.logger.info(f"Task agent LLM class: {task.agent.llm.__class__.__name__}")
        
        try:
            # Get model name from environment if testing
            model_name = "llama3" 
            if os.environ.get("ENV") == "test":
                model_name = "deepseek-r1:14b"  # Use deepseek-r1:14b for testing
            
            # Specify the model directly in the CrewAI format
            model_spec = f"ollama/{model_name}"
            
            # Create a crew directly using the model parameter
            self.logger.info(f"Creating mini-crew for task execution with model {model_spec}")
            crew = Crew(
                agents=[task.agent],
                tasks=[task],
                process=Process.sequential,
                model=model_spec,  # Set the model directly in the crew
                base_url="http://localhost:11434",  # Set the Ollama API endpoint
                verbose=True  # Enable verbose output for debugging
            )
            
            # Execute the task
            self.logger.info("Starting task execution with CrewAI...")
            start_time = time.time()
            
            # Use threading for timeout
            result_container = [None]
            error_container = [None]
            
            def threaded_kickoff():
                try:
                    self.logger.info("Thread started for CrewAI kickoff")
                    result_container[0] = crew.kickoff()
                    self.logger.info("CrewAI kickoff completed in thread")
                except Exception as e:
                    self.logger.error(f"Exception in kickoff thread: {str(e)}", exc_info=True)
                    error_container[0] = e
            
            # Start the thread
            self.logger.info("Starting kickoff in thread with timeout...")
            kickoff_thread = threading.Thread(target=threaded_kickoff)
            kickoff_thread.daemon = True
            kickoff_thread.start()
            
            # Wait for completion or timeout
            timeout = 180  # Increase timeout to 3 minutes
            self.logger.info(f"Waiting up to {timeout} seconds for task completion...")
            
            # Log progress during wait
            elapsed = 0
            check_interval = 15  # Log every 15 seconds
            while kickoff_thread.is_alive() and elapsed < timeout:
                kickoff_thread.join(check_interval)
                elapsed += check_interval
                if kickoff_thread.is_alive():
                    self.logger.info(f"Still waiting for kickoff to complete... ({elapsed} seconds elapsed)")
            
            if kickoff_thread.is_alive():
                self.logger.error(f"Kickoff timed out after {timeout} seconds")
                raise TimeoutError(f"CrewAI kickoff timed out after {timeout} seconds")
            
            if error_container[0] is not None:
                self.logger.error(f"Error in kickoff thread: {error_container[0]}")
                raise error_container[0]
            
            self.logger.info("Task completed successfully within timeout")
            result = result_container[0]
            execution_time = time.time() - start_time
            
            # Handle CrewOutput object (returned by newer versions of CrewAI)
            if hasattr(result, '__class__') and result.__class__.__name__ == 'CrewOutput':
                self.logger.info("Converting CrewOutput to string")
                # Extract the actual result string from the CrewOutput object
                if hasattr(result, 'raw_output'):
                    result = result.raw_output
                elif hasattr(result, 'output'):
                    result = result.output
                else:
                    # Last resort: convert to string
                    result = str(result)
            
            # Log the result
            result_length = len(str(result)) if result else 0
            self.logger.info(f"Task execution completed in {execution_time:.2f} seconds")
            self.logger.info(f"Result length: {result_length} characters")
            if result_length < 500:  # Only log the full result if it's not too large
                self.logger.info(f"Result: {result}")
            else:
                self.logger.info(f"Result (truncated): {str(result)[:500]}...")
                
            return result
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}", exc_info=True)
            raise

    def _parse_user_query(self, query: str) -> Dict[str, Any]:
        """Parse the user's natural language query.

        Args:
            query: Natural language query from the user

        Returns:
            Dict[str, Any]: Parsed query information
        """
        self.logger.info(f"Parsing user query: {query}")
        
        # Create the intent analysis task
        intent_task = self.data_analyst.create_intent_task(query)
        
        # Execute the task with a mini-crew
        intent_result = self._execute_task(intent_task)
        
        self.logger.info("User query parsed successfully")
        return {
            "original_query": query,
            "parsed_intent": intent_result,
        }

    def _get_database_schema(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Get the database schema information.

        Args:
            parsed_query: Parsed query information from previous step

        Returns:
            Dict[str, Any]: Database schema information
        """
        self.logger.info("Getting database schema")
        
        # Get the list of tables
        tables = self.db_conn.get_table_names()
        self.logger.info(f"Found {len(tables)} tables in the database")
        
        # Get the schema for each table
        schema = {}
        for table in tables:
            schema[table] = self.db_conn.get_table_schema(table)
        
        self.logger.info("Database schema retrieved successfully")
        return {
            **parsed_query,
            "db_schema": schema,
        }

    def _generate_sql_query(self, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an SQL query from the natural language query.

        Args:
            query_info: Query information and schema from previous steps

        Returns:
            Dict[str, Any]: Updated information with SQL query
        """
        self.logger.info("Generating SQL query")
        
        # Extract the necessary information
        query = query_info["original_query"]
        intent = query_info["parsed_intent"]
        schema = query_info["db_schema"]
        
        # Create the SQL generation task
        sql_task = self.sql_developer.create_task(
            query=f"{query}\nUser intent: {intent}",
            db_schema=schema
        )
        
        # Execute the task with a mini-crew
        sql_query = self._execute_task(sql_task)
        
        # Clean the SQL query from markdown syntax and other formatting
        sql_query = self._clean_sql_query(sql_query)
        
        self.logger.info(f"SQL query generated: {sql_query}")
        return {
            **query_info,
            "sql_query": sql_query,
        }
        
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean SQL query from markdown syntax and other formatting.
        
        Args:
            sql_query: The SQL query string, potentially with markdown formatting
            
        Returns:
            str: Cleaned SQL query
        """
        self.logger.info("Cleaning SQL query from markdown syntax")
        
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
            
        self.logger.info(f"Cleaned SQL query: {sql_query}")
        return sql_query

    def _validate_sql_query(self, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the SQL query.

        Args:
            query_info: Query information with SQL query

        Returns:
            Dict[str, Any]: Validation results
        """
        self.logger.info("Validating SQL query")
        
        # Extract the SQL query
        sql_query = query_info["sql_query"]
        
        # Use the validation tool
        for tool in self.db_tools:
            if tool.name == "validate_sql_query":
                validation_result = tool._run(query=sql_query)
                break
        else:
            self.logger.error("Validation tool not found")
            validation_result = {"is_valid": False, "error": "Validation tool not found"}
        
        # If the query is invalid, attempt to fix it
        if not validation_result["is_valid"]:
            self.logger.warning(f"SQL validation failed: {validation_result['error']}")
            
            # Use the SQL developer to fix the query
            fix_task = self.sql_developer.create_task(
                query=f"{query_info['original_query']}\nFix this SQL query that has the following error: {validation_result['error']}\nOriginal query: {sql_query}",
                db_schema=query_info["db_schema"]
            )
            
            # Execute the task with a mini-crew
            fixed_sql = self._execute_task(fix_task)
            
            self.logger.info(f"Fixed SQL query: {fixed_sql}")
            
            # Update the query_info
            query_info["sql_query"] = fixed_sql
            query_info["sql_query_fixed"] = True
            query_info["original_validation_error"] = validation_result["error"]
            
            # Validate again
            for tool in self.db_tools:
                if tool.name == "validate_sql_query":
                    validation_result = tool._run(query=fixed_sql)
                    break
        
        self.logger.info(f"SQL validation result: {validation_result}")
        return {
            **query_info,
            "validation_result": validation_result,
        }

    def _execute_sql_query(self, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the SQL query.

        Args:
            query_info: Query information with validated SQL query

        Returns:
            Dict[str, Any]: Query results
        """
        self.logger.info("Executing SQL query")
        
        # Check if the query is valid
        if not query_info.get("validation_result", {}).get("is_valid", False):
            self.logger.error("Cannot execute invalid SQL query")
            return {
                **query_info,
                "execution_result": {"error": "Invalid SQL query"},
                "execution_success": False,
            }
        
        # Extract the SQL query
        sql_query = query_info["sql_query"]
        
        # Execute the query
        try:
            start_time = time.time()
            rows, error = self.db_conn.execute_query(sql_query)
            execution_time = time.time() - start_time
            
            if error:
                self.logger.error(f"SQL execution error: {error}")
                return {
                    **query_info,
                    "execution_result": {"error": error},
                    "execution_success": False,
                }
            
            self.logger.info(f"SQL query executed successfully in {execution_time:.2f} seconds")
            return {
                **query_info,
                "results": rows,
                "result_count": len(rows),
                "execution_time": execution_time,
                "execution_success": True,
            }
        except Exception as e:
            self.logger.error(f"Error executing SQL query: {str(e)}")
            return {
                **query_info,
                "execution_result": {"error": str(e)},
                "execution_success": False,
            }

    def _analyze_query_results(self, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the query results.

        Args:
            query_info: Query information with results

        Returns:
            Dict[str, Any]: Analysis results
        """
        self.logger.info("Analyzing query results")
        
        # Check if query execution was successful
        if not query_info.get("execution_success", False):
            self.logger.error("Cannot analyze failed query execution")
            return {
                **query_info,
                "analysis": "Query execution failed, no analysis possible.",
            }
        
        # Extract the necessary information
        query = query_info["original_query"]
        sql_query = query_info["sql_query"]
        results = query_info.get("results", [])
        
        # Ensure results is a list of dictionaries
        if not isinstance(results, list):
            self.logger.warning(f"Expected results to be a list, got {type(results).__name__}")
            results = []
        
        # Log the result size
        self.logger.info(f"Creating results analysis task for {len(results)} rows")
        
        # Create the analysis task
        analysis_task = self.data_analyst.create_analysis_task(
            query=query,
            sql_query=sql_query,
            results=results
        )
        
        # Execute the task with a mini-crew
        analysis = self._execute_task(analysis_task)
        
        self.logger.info("Query results analyzed successfully")
        return {
            **query_info,
            "analysis": analysis,
        }

    def _generate_report(self, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a user-friendly report.

        Args:
            query_info: Query information with analysis

        Returns:
            Dict[str, Any]: Report information
        """
        self.logger.info("Generating report")
        
        # Check if query execution was successful
        if not query_info.get("execution_success", False):
            self.logger.error("Cannot generate report for failed query execution")
            return {
                **query_info,
                "report": "Query execution failed, no report possible.",
            }
        
        # Extract the necessary information
        query = query_info["original_query"]
        sql_query = query_info["sql_query"]
        results = query_info.get("results", [])
        analysis = query_info.get("analysis", "No analysis available")
        
        # Ensure results is a list
        if not isinstance(results, list):
            self.logger.warning(f"Expected results to be a list, got {type(results).__name__}")
            results = []
        
        # Create and run the report task
        report_task = self.report_generator.create_report_task(
            query=query,
            sql_query=sql_query,
            results=results,
            analysis=analysis
        )
        
        # Execute the task with a mini-crew
        report = self._execute_task(report_task)
        
        self.logger.info("Report generated successfully")
        return {
            **query_info,
            "report": report,
        }

    def _handle_error(self, error: Exception, step: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors in the workflow.

        Args:
            error: The exception that occurred
            step: The step where the error occurred
            context: The context at the time of the error

        Returns:
            Dict[str, Any]: Error information and context
        """
        self.logger.error(f"Error in step '{step}': {str(error)}", exc_info=True)
        
        # If we have a query, include it in the error information
        original_query = context.get("original_query", "No query available")
        
        return {
            "success": False,
            "error": f"Error in step '{step}': {str(error)}",
            "step": step,
            "original_query": original_query,
        }

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query.

        Args:
            query: Natural language query from the user

        Returns:
            Dict[str, Any]: Query results and report
        """
        self.logger.info(f"Processing query: {query}")
        start_time = time.time()
        
        try:
            # Execute the workflow
            result = query
            result = self._parse_user_query(result)
            result = self._get_database_schema(result)
            result = self._generate_sql_query(result)
            result = self._validate_sql_query(result)
            result = self._execute_sql_query(result)
            result = self._analyze_query_results(result)
            result = self._generate_report(result)
            
            # Convert any Decimal values to float for JSON serialization
            self._convert_decimal_to_float(result)
            
            # Add success flag and processing time
            result["success"] = True
            result["processing_time_seconds"] = time.time() - start_time
            
            self.logger.info(f"Query processed successfully in {result['processing_time_seconds']:.2f} seconds")
            
            # Save the results to a file for AgentIQ integrations
            self._save_results(result)
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to process query: {str(e)}")
            
            # Return an error response
            error_result = {
                "success": False,
                "error": f"Failed to process query: {str(e)}",
                "processing_time_seconds": time.time() - start_time
            }
            
            return error_result

    def _convert_decimal_to_float(self, obj: Any) -> None:
        """Recursively convert Decimal values to float in a nested structure.
        
        Args:
            obj: The object to process
        """
        import decimal
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, decimal.Decimal):
                    obj[key] = float(value)
                else:
                    self._convert_decimal_to_float(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, decimal.Decimal):
                    obj[i] = float(item)
                else:
                    self._convert_decimal_to_float(item)

    def _save_results(self, result: Dict[str, Any]) -> None:
        """Save the results to a file for AgentIQ integrations.

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
            "success": result.get("success", False),
            "processing_time_seconds": result.get("processing_time_seconds", 0),
            "result_count": result.get("result_count", 0),
        }
        
        if result.get("success", False):
            output_result["report"] = result.get("report", "")
            # Only include a sample of results to keep file size reasonable
            output_result["results_sample"] = result.get("results", [])[:10]
        
        with open(output_file, "w") as f:
            json.dump(output_result, f, indent=2, cls=DecimalEncoder)
            
        self.logger.info(f"Results saved to {output_file}")


def create_db_query_workflow() -> DBQueryWorkflow:
    """Create a new DBQueryWorkflow instance.

    Returns:
        DBQueryWorkflow: Initialized workflow
    """
    return DBQueryWorkflow() 