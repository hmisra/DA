#!/usr/bin/env python
"""
Test script for executing a simple query using the pure AgentIQ implementation.
Tests the integration between AgentIQ, function-based workflow, and Ollama.
"""
import os
import sys
import os.path
import json
import yaml
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set test environment
os.environ["ENV"] = "test"

# Import after setting up the environment
from aiq.builder.builder import Builder
from src.utils.logging import setup_logging, AgentLogger

# Set up logging
setup_logging()
logger = AgentLogger(agent_name="AgentIQPureTester", agent_type="Test Script")

async def run_test_query():
    """Run a test query using the pure AgentIQ workflow."""
    logger.info("Starting pure AgentIQ workflow test")
    
    # Define a test query
    test_query = "What are the top 5 products by quantity sold?"
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Load the workflow configuration from YAML
    config_path = project_root / "configs" / "db_query_workflow.yaml"
    logger.info(f"Looking for config file at: {config_path}")
    
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return
        
    # Load the YAML file using standard PyYAML
    with open(config_path, "r") as f:
        workflow_config = yaml.safe_load(f)
    
    # Create the builder
    builder = Builder()
    
    # Initialize the context with the query
    context = {"input": test_query, "query": test_query}
    
    # Run the workflow
    logger.info(f"Running workflow with query: {test_query}")
    try:
        result = await builder.run_workflow(workflow_config, context)
        
        # Print the results
        print("\n=== QUERY RESULT ===")
        print(f"Query: {result.get('original_query', '')}")
        print(f"\nSQL: {result.get('sql_query', '')}")
        print(f"\nReport:\n{result.get('report', '')}")
        
        # Save the result to a JSON file
        output_path = project_root / "results" / "test_result.json"
        output_path.parent.mkdir(exist_ok=True)
        
        # Use a custom JSON encoder for any special types
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    return super().default(obj)
                except TypeError:
                    return str(obj)
        
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, cls=CustomEncoder)
        
        logger.info(f"Test results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}", exc_info=True)
        return None
        
    logger.info("Pure AgentIQ workflow test completed successfully")
    return result

def main():
    """Run the test and return exit code."""
    try:
        result = asyncio.run(run_test_query())
        if result is None:
            return 1
        return 0
    except Exception as e:
        logger.error(f"Fatal error in test execution: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 