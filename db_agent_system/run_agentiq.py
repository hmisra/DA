#!/usr/bin/env python
"""
Run the DB Query workflow using AgentIQ.
"""
import os
import sys
import json
import yaml
import argparse
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Set up environment
os.environ["ENV"] = "development"

# Import here after setting up the environment
from aiq.builder.builder import Builder
from src.utils.config import config
from src.utils.logging import setup_logging, AgentLogger

logger = AgentLogger(agent_name="AgentIQRunner", agent_type="Script")

async def run_with_query(config_path: str, query: str):
    """Run the workflow with a natural language query.
    
    Args:
        config_path: Path to the workflow configuration file
        query: Natural language query to process
    """
    logger.info(f"Running workflow with query: {query}")
    
    # Load the workflow configuration
    with open(config_path, "r") as f:
        workflow_config = yaml.safe_load(f)
    
    # Create the builder
    builder = Builder()
    
    # Initialize the context with the query
    context = {"input": query, "query": query}
    
    # Run the workflow
    result = await builder.run_workflow(workflow_config, context)
    
    # Print the results
    print("\n=== QUERY RESULT ===")
    print(f"Query: {result.get('original_query', '')}")
    print(f"\nSQL: {result.get('sql_query', '')}")
    print(f"\nReport:\n{result.get('report', '')}")
    
    return result

def main():
    """Run the DB Query workflow using AgentIQ."""
    parser = argparse.ArgumentParser(description="Run the DB Query workflow using AgentIQ")
    parser.add_argument("--serve", action="store_true", help="Serve the workflow using the FastAPI frontend")
    parser.add_argument("--query", type=str, help="Natural language query to process")
    parser.add_argument("--config", type=str, default="configs/db_query_workflow.yaml", 
                        help="Path to workflow configuration file")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    if args.serve:
        # Serve the workflow using the FastAPI frontend
        from subprocess import run
        cmd = ["aiq", "serve", args.config]
        run(cmd)
    elif args.query:
        # Run the workflow with the query
        asyncio.run(run_with_query(args.config, args.query))
    else:
        # Start interactive mode
        from subprocess import run
        cmd = ["aiq", "run", args.config]
        run(cmd)

if __name__ == "__main__":
    main() 