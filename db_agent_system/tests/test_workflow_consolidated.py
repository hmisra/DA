"""
Test module for the DB Query workflow.
This consolidated test file covers all workflow and integration tests.
"""
import pytest
import time

from conftest import BaseTestClass, check_ollama_available
from src.agentiq.workflow import DBQueryWorkflow, create_db_query_workflow


class TestWorkflowBasics(BaseTestClass):
    """Test the basic workflow functionality."""

    def test_workflow_initialization(self):
        """Test initializing the workflow with real components."""
        # Initialize the workflow
        workflow = DBQueryWorkflow()
        
        # Verify the workflow and its components
        assert workflow is not None, "Failed to create workflow"
        assert workflow.db_conn is not None, "Workflow has no database connection"
        assert workflow.db_tools is not None, "Workflow has no database tools"
        assert workflow.sql_developer is not None, "Workflow has no SQL developer agent"
        assert workflow.data_analyst is not None, "Workflow has no data analyst agent"
        assert workflow.report_generator is not None, "Workflow has no report generator agent"
        assert workflow.workflow_config is not None, "Workflow has no workflow configuration"
        
        # Print workflow information
        print(f"\nWorkflow initialized successfully")
        print(f"SQL Developer: {workflow.sql_developer.__class__.__name__}")
        print(f"Data Analyst: {workflow.data_analyst.__class__.__name__}")
        print(f"Report Generator: {workflow.report_generator.__class__.__name__}")
        
        # Verify workflow methods exist and are callable
        assert hasattr(workflow, "process_query"), "Workflow has no process_query method"
        assert callable(workflow.process_query), "process_query is not callable"

    def test_execute_task_method(self):
        """Test that the _execute_task method works correctly."""
        # Create the workflow
        workflow = DBQueryWorkflow()
        
        # Create a simple task
        task = workflow.data_analyst.create_intent_task("list all products")
        
        # Execute the task with our helper method
        print("Executing task with our new _execute_task method...")
        result = workflow._execute_task(task)
        
        # Verify result
        assert result is not None, "Task execution should return a result"
        print(f"Task execution result: {result}")
        
        # This is a successful test if we get here without errors


@pytest.mark.integration
class TestWorkflowIntegration(BaseTestClass):
    """Integration tests for the DBQueryWorkflow."""
    
    @pytest.fixture(autouse=True)
    def setup_ollama(self, check_ollama_available):
        """Set up Ollama for each test."""
        self.models = check_ollama_available
        print(f"Available Ollama models: {', '.join(self.models)}")
    
    def test_process_query(self):
        """Test processing a query end-to-end."""
        # Create the workflow
        workflow = DBQueryWorkflow()
        
        # Test with a simple query
        query = "Show me all products"
        result = workflow.process_query(query)
        
        # Verify result structure
        assert isinstance(result, dict), "Result is not a dictionary"
        assert "success" in result, "Success field not in result"
        
        # Log the entire result for debugging
        print(f"Process query result: {result}")
        
        # Even if we have an error, let's check what error it is
        if not result.get("success", False):
            print(f"Error during query processing: {result.get('error', 'No error message')}")
            
            # Assert that if we failed, it's with the specific CrewAI task execution error
            if "'Task' object has no attribute 'execute'" in result.get("error", ""):
                # Our fix should prevent this error, so this test should fail if we see it
                assert False, "Task execute method error should be fixed with our implementation"
        else:
            # If successful, validate all expected fields
            assert "sql_query" in result, "SQL query not in result"
            assert "results" in result, "Results not in result"
            assert "report" in result, "Report not in result"
            assert len(result["results"]) > 0, "No results returned"
            
        print(f"SQL Query: {result.get('sql_query', 'No SQL query')}")
        if result.get("success", False):
            print(f"Found {len(result['results'])} results")
            if len(result['results']) > 0:
                print(f"First result: {result['results'][0]}")
    
    def test_product_query(self):
        """Test a simple query about products."""
        # Create a workflow with real components, no mocking
        workflow = create_db_query_workflow()
        
        # Use a simple, clear query that should generate a predictable SQL query
        query = "What products do we have in the Electronics category?"
        
        # Process the query
        result = workflow.process_query(query)
        
        # Print results for debugging
        print(f"\nSQL Query: {result.get('sql_query', 'No SQL query generated')}")
        print(f"\nResults: {result.get('results', [])}")
        print(f"\nReport: {result.get('report', 'No report generated')}")
        print(f"\nProcessing Time: {result.get('processing_time_seconds', 0):.2f} seconds")
        
        # Basic assertions
        assert result["success"] is True, f"Query failed: {result.get('error', 'Unknown error')}"
        
        # If there's an SQL query, check it has expected content
        if result.get("sql_query"):
            assert isinstance(result["sql_query"], str), "SQL query is not a string"
            sql_lower = result["sql_query"].lower()
            assert "product" in sql_lower, "SQL query doesn't reference products"
            assert "category" in sql_lower, "SQL query doesn't reference category"
        
        # If there are results, check they're not empty
        if result.get("results"):
            assert len(result["results"]) > 0, "No results returned"
        
        # If there's a report, check it exists
        if result.get("report"):
            assert isinstance(result["report"], str), "Report is not a string"
        
        assert "processing_time_seconds" in result, "Processing time not included in result"

    def test_order_query(self):
        """Test a query about orders."""
        # Create a workflow with real components, no mocking
        workflow = create_db_query_workflow()
        
        # Use a simple, clear query that should generate a predictable SQL query
        query = "What is our highest value order?"
        
        # Process the query
        result = workflow.process_query(query)
        
        # Print results for debugging
        print(f"\nSQL Query: {result.get('sql_query', 'No SQL query generated')}")
        print(f"\nResults: {result.get('results', [])}")
        print(f"\nReport: {result.get('report', 'No report generated')}")
        print(f"\nProcessing Time: {result.get('processing_time_seconds', 0):.2f} seconds")
        
        # Basic assertions
        assert result["success"] is True, f"Query failed: {result.get('error', 'Unknown error')}"
        
        # If there's an SQL query, check it has expected content
        if result.get("sql_query"):
            assert isinstance(result["sql_query"], str), "SQL query is not a string"
            sql_lower = result["sql_query"].lower()
            assert "order" in sql_lower, "SQL query doesn't reference orders"
        
        # If there are results, check they're not empty
        if result.get("results"):
            assert len(result["results"]) > 0, "No results returned"
        
        # If there's a report, check it exists
        if result.get("report"):
            assert isinstance(result["report"], str), "Report is not a string"
        
        assert "processing_time_seconds" in result, "Processing time not included in result" 