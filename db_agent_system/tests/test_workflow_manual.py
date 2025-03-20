"""
Manual test script for the workflow implementation.
This will test our fix for the CrewAI task execution issue.
"""
import os
import sys
import os.path
import time
import signal
import threading

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set test environment
os.environ["ENV"] = "test"

from crewai import Crew, Agent, Process
from src.langchain.database import get_db_tools
from src.crewai.agents import create_sql_developer_agent, create_data_analyst_agent
from src.utils.logging import setup_logging

# Set a timeout for task execution (in seconds)
TASK_TIMEOUT = 90  # 90 seconds timeout

class TimeoutError(Exception):
    """Raised when a function times out."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeouts."""
    raise TimeoutError("Task execution timed out")

def run_with_timeout(func, *args, timeout=TASK_TIMEOUT, **kwargs):
    """Run a function with a timeout."""
    result = [None]
    error = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e
    
    # Create and start the thread
    thread = threading.Thread(target=target)
    thread.daemon = True
    
    # Start the timeout timer
    start_time = time.time()
    thread.start()
    
    # Wait for the thread to complete or timeout
    thread.join(timeout)
    
    # Check if the thread is still alive (timeout occurred)
    if thread.is_alive():
        return None, TimeoutError(f"Function execution timed out after {timeout} seconds")
    
    # Return the result or error
    if error[0] is not None:
        return None, error[0]
    else:
        execution_time = time.time() - start_time
        return result[0], execution_time

def main():
    """Run a manual test of the workflow."""
    # Set up logging
    setup_logging()
    
    # Get database tools
    db_tools = get_db_tools()

    # Create agents
    sql_agent = create_sql_developer_agent(db_tools)
    data_agent = create_data_analyst_agent(db_tools)

    # Create a crew
    crew = Crew(
        agents=[sql_agent.agent, data_agent.agent],
        tasks=[
            sql_agent.create_task("list all customers", {}),
            data_agent.create_intent_task("list all customers")
        ],
        process=Process.sequential,
        verbose=True
    )

    # Run the crew
    result = crew.kickoff()
    print(f"Result: {result}")

    print("Creating workflow...")
    workflow = DBQueryWorkflow()
    
    print("Creating a simple task...")
    task = workflow.data_analyst.create_intent_task("list all products")
    
    # Print details about the model being used
    print(f"Task agent: {task.agent.role}")
    print(f"Task agent model: {task.agent.llm.model}")
    print(f"Task agent LLM class: {task.agent.llm.__class__.__name__}")

    # Check if the LLM implements the required CrewAI methods
    print(f"LLM supports_stop_words method available: {hasattr(task.agent.llm, 'supports_stop_words')}")
    if hasattr(task.agent.llm, 'supports_stop_words'):
        print(f"LLM supports_stop_words result: {task.agent.llm.supports_stop_words()}")
    
    print(f"LLM get_num_tokens method available: {hasattr(task.agent.llm, 'get_num_tokens')}")
    if hasattr(task.agent.llm, 'get_num_tokens'):
        # Test the method
        print(f"LLM token count for 'test': {task.agent.llm.get_num_tokens('test')}")
    
    # Check if Ollama is responding
    print("Testing Ollama connection...")
    try:
        # Simple test prompt
        test_result = task.agent.llm.invoke("Say 'Hello, test successful!'")
        print(f"Ollama test response: {test_result}")
    except Exception as e:
        print(f"ERROR: Ollama test failed: {str(e)}")
        return 1
    
    print("Executing task with our _execute_task method with timeout...")
    result, execution_info = run_with_timeout(workflow._execute_task, task)
    
    if isinstance(execution_info, Exception):
        print(f"ERROR: Task execution failed: {str(execution_info)}")
        return 1
    else:
        execution_time = execution_info
        print(f"Task execution completed in {execution_time:.2f} seconds")
        print(f"Result: {result}")
    
    print("Test successful!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 