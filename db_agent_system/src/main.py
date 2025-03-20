"""
Main module for the DB Agent System.
Serves as the entry point for the application.
"""
import argparse
import json
import time
from typing import Dict, Any

from loguru import logger

from src.agentiq.workflow import create_db_query_workflow
from src.utils.config import config
from src.utils.logging import setup_logging


def process_query(query: str) -> Dict[str, Any]:
    """Process a natural language query.

    Args:
        query: Natural language query from the user

    Returns:
        Dict[str, Any]: Processing results
    """
    logger.info(f"Processing query: {query}")
    
    # Create the workflow
    workflow = create_db_query_workflow()
    
    # Start timing
    start_time = time.time()
    
    # Process the query
    result = workflow.process_query(query)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Add timing information to the result
    result["processing_time_seconds"] = round(elapsed_time, 2)
    
    logger.info(f"Query processed in {elapsed_time:.2f} seconds")
    return result


def process_query_from_file(file_path: str) -> Dict[str, Any]:
    """Process a natural language query from a file.

    Args:
        file_path: Path to the file containing the query

    Returns:
        Dict[str, Any]: Processing results
    """
    logger.info(f"Reading query from file: {file_path}")
    
    try:
        with open(file_path, "r") as f:
            query = f.read().strip()
        
        return process_query(query)
    except Exception as e:
        logger.error(f"Error reading query from file: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Error reading query from file: {str(e)}",
        }


def save_results(results: Dict[str, Any], output_file: str) -> None:
    """Save the processing results to a file.

    Args:
        results: Processing results
        output_file: Path to the output file
    """
    logger.info(f"Saving results to file: {output_file}")
    
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to file: {str(e)}", exc_info=True)


def main() -> None:
    """Main entry point of the application."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="DB Agent System - Natural language to SQL")
    
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "-q", "--query", 
        help="Natural language query to process"
    )
    query_group.add_argument(
        "-f", "--file", 
        help="Path to file containing the query"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path to save the output (default: stdout)"
    )
    
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Process the query
        if args.query:
            results = process_query(args.query)
        else:
            results = process_query_from_file(args.file)
        
        # Output the results
        if args.output:
            save_results(results, args.output)
        else:
            # Print to stdout
            if args.pretty:
                print(json.dumps(results, indent=2))
            else:
                print(json.dumps(results))
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        if args.pretty:
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
        else:
            print(json.dumps({"success": False, "error": str(e)}))


def start_interactive_mode() -> None:
    """Start the interactive mode of the application."""
    logger.info("Starting interactive mode")
    
    print("DB Agent System - Interactive Mode")
    print("Enter 'exit' or 'quit' to exit")
    print("-----------------------------------")
    
    while True:
        # Get the query from the user
        try:
            query = input("\nEnter your query: ").strip()
            
            # Check if the user wants to exit
            if query.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            
            # Skip empty queries
            if not query:
                continue
            
            # Process the query
            start_time = time.time()
            results = process_query(query)
            elapsed_time = time.time() - start_time
            
            # Print the results
            print("\n" + "=" * 50)
            
            if results["success"]:
                print(f"SQL query: {results.get('sql_query', 'N/A')}")
                print("\nReport:")
                print(results.get("report", "No report generated"))
            else:
                print(f"Error: {results.get('error', 'Unknown error')}")
            
            print(f"\nProcessing time: {elapsed_time:.2f} seconds")
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}", exc_info=True)
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Initialize logging
    setup_logging()
    
    # Check if interactive mode is requested
    parser = argparse.ArgumentParser(description="DB Agent System - Natural language to SQL")
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start in interactive mode"
    )
    
    # Parse only the interactive argument
    args, _ = parser.parse_known_args()
    
    if args.interactive:
        start_interactive_mode()
    else:
        main() 