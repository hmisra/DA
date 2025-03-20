#!/usr/bin/env python3
"""
Simple test script for the DB Query Workflow.
This scripts runs the workflow on a simple query without using pytest.
"""
import os
import logging
import sys
import os.path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agentiq.workflow import create_db_query_workflow
from src.utils.logging import setup_logging

# Set environment for testing
os.environ["ENV"] = "test"

# Setup logging
setup_logging()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():
    """Run a simple test of the DB Query Workflow."""
    try:
        logger.info("Starting simple workflow test")
        
        # Create the workflow
        logger.info("Creating DB Query Workflow")
        workflow = create_db_query_workflow()
        
        # Test a very simple query
        test_query = "list all customers"
        
        logger.info(f"Processing query: {test_query}")
        result = workflow.process_query(test_query)
        
        # Check the result
        if result.get("success", False):
            logger.info("Workflow executed successfully!")
            logger.info(f"SQL Query: {result.get('sql_query', 'N/A')}")
            logger.info(f"Result count: {result.get('result_count', 0)}")
            logger.info("Report (truncated):")
            report = result.get("report", "No report generated")
            logger.info(report[:500] + "..." if len(report) > 500 else report)
        else:
            logger.error(f"Workflow execution failed: {result.get('error', 'Unknown error')}")
            
        return 0 if result.get("success", False) else 1
    
    except Exception as e:
        logger.error(f"Error in test script: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 