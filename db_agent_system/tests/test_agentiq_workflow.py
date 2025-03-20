"""
Test script for executing a simple query using the AgentIQ workflow.
Tests the integration between AgentIQ, CrewAI, and Ollama.
"""
import os
import sys
import os.path
import time
import threading

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set test environment
os.environ["ENV"] = "test"

# Import the workflow
from src.agentiq.workflow import create_db_query_workflow
from src.utils.logging import setup_logging

# Set a timeout for the entire workflow execution
WORKFLOW_TIMEOUT = 180  # 3 minutes timeout

def main():
    """Run a test of the AgentIQ workflow."""
    # Set up logging
    setup_logging()
    
    print("Creating AgentIQ workflow...")
    workflow = create_db_query_workflow()
    
    # Simple query to test
    query = "list all customers"
    
    print(f"Processing query: '{query}'")
    
    # Use threading for timeout
    result_container = [None]
    error_container = [None]
    
    def execute_workflow():
        try:
            # Process the query with the workflow
            result_container[0] = workflow.process_query(query)
        except Exception as e:
            error_container[0] = e
    
    # Start the thread
    print("Starting workflow execution with timeout...")
    workflow_thread = threading.Thread(target=execute_workflow)
    workflow_thread.daemon = True
    start_time = time.time()
    workflow_thread.start()
    
    # Wait for completion or timeout
    workflow_thread.join(WORKFLOW_TIMEOUT)
    
    if workflow_thread.is_alive():
        print(f"ERROR: Workflow execution timed out after {WORKFLOW_TIMEOUT} seconds")
        return 1
    
    if error_container[0] is not None:
        print(f"ERROR: Workflow execution failed: {str(error_container[0])}")
        return 1
    
    result = result_container[0]
    execution_time = time.time() - start_time
    
    print(f"Workflow execution completed in {execution_time:.2f} seconds")
    if result.get("success", False):
        print("Success!")
        print(f"SQL Query: {result.get('sql_query', 'N/A')}")
        print(f"Results: {len(result.get('results', []))} rows")
    else:
        print(f"Failed: {result.get('error', 'Unknown error')}")
    
    print("Test successful!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 