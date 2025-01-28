import pytest
from unittest.mock import Mock
from langchain_ollama import OllamaLLM
from analysis.analysis_agent import AnalysisAgent
from tools.database import PostgresConnector
from tools.transformer import DataTransformer
import pandas as pd
import json
import logging
from datetime import datetime, timedelta

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)8s] %(message)s\n',  # Added newline for better readability
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('test.log', mode='w'),  # 'w' mode to start fresh each test run
        logging.StreamHandler()  # This will continue to print to console
    ]
)
logger = logging.getLogger(__name__)

# Also capture langchain and httpx debug logs
logging.getLogger('langchain').setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.DEBUG)
logging.getLogger('httpcore').setLevel(logging.DEBUG)

def log_separator(logger):
    """Log a separator line for better visibility."""
    logger.info("\n" + "="*120 + "\n")

@pytest.fixture(autouse=True)
def log_test_start_end(request):
    """Log the start and end of each test."""
    log_separator(logger)
    logger.info(f"Starting test: {request.node.name}")
    log_separator(logger)
    yield
    log_separator(logger)
    logger.info(f"Finished test: {request.node.name}")
    log_separator(logger)

@pytest.fixture
def llm():
    """Create OllamaLLM instance for testing."""
    logger.info("Initializing OllamaLLM for testing")
    llm = OllamaLLM(
        model="deepseek-r1:70b",
        temperature=0,
        stop=["</o>", "</sql>", "</think>", "</system>", "</user>", "</assistant>"],
        num_predict=2000,
        num_ctx=4096,
        top_k=1,
        top_p=0.1,
        repeat_penalty=1.2,
        format="json",
        mirostat=2,
        mirostat_tau=3.0,
        mirostat_eta=0.1
    )
    return llm

@pytest.fixture
def mock_db():
    """Mock database for testing."""
    logger.info("Setting up mock database")
    db = Mock(spec=PostgresConnector)
    db.get_table_schema.return_value = [
        {"column_name": "date", "data_type": "timestamp"},
        {"column_name": "product", "data_type": "varchar"},
        {"column_name": "revenue", "data_type": "numeric"}
    ]
    logger.debug("Mock database schema configured")
    return db

@pytest.fixture
def analysis_agent(mock_db, llm):
    """Create AnalysisAgent instance for testing."""
    return AnalysisAgent(db=mock_db, llm=llm)

def test_parse_user_query(analysis_agent):
    """Test parsing of user query."""
    logger.info("Testing query parsing")
    query = "Show me total revenue by product"
    result = analysis_agent.parse_user_query(query)
    logger.debug(f"Parsed query result: {result}")
    assert isinstance(result, dict)
    assert "metrics" in result
    assert "dimensions" in result

def test_generate_sql(analysis_agent):
    """Test SQL query generation."""
    logger.info("Testing SQL generation")
    params = {
        "metrics": ["revenue"],
        "dimensions": ["product"],
        "time_range": {"start": "2024-01-01", "end": "2024-01-31"},
        "filters": {},
        "sort": {}
    }
    logger.debug(f"Input parameters: {params}")
    sql = analysis_agent.generate_sql("test_table", params)
    logger.debug(f"Generated SQL: {sql}")
    assert isinstance(sql, str)
    assert "SELECT" in sql
    assert "FROM" in sql

def test_execute_analysis(analysis_agent):
    """Test executing complete analysis workflow."""
    query = "What was the total revenue?"
    results = analysis_agent.execute_analysis(query, "test_table")
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert "revenue" in results[0]

def test_validate_results(analysis_agent):
    """Test validating analysis results."""
    results = {
        "results": [
            {"date": "2024-01-01", "product": "A", "revenue": 100},
            {"date": "2024-01-02", "product": "B", "revenue": 200}
        ]
    }
    
    validations = analysis_agent.validate_results(results)
    assert isinstance(validations, list)

def test_data_validation(analysis_agent):
    """Test data validation functionality."""
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "product": ["A", "B"],
        "revenue": [100, 200]
    })
    
    validations = analysis_agent.validate_results({"results": df.to_dict("records")})
    assert isinstance(validations, list)

@pytest.mark.parametrize("query,expected_metrics", [
    ("What was the total revenue?", ["revenue"]),
    ("How many units were sold by product?", ["units"]),
    ("What's the average price by category?", ["price"])
])
def test_query_understanding(analysis_agent, query, expected_metrics):
    """Test query understanding with different queries."""
    result = analysis_agent.parse_user_query(query)
    assert isinstance(result, dict)
    assert "metrics" in result
    assert any(metric in result["metrics"] for metric in expected_metrics)

def test_error_handling(analysis_agent):
    """Test error handling in analysis workflow."""
    analysis_agent.db.execute_query.side_effect = Exception("Database error")
    
    with pytest.raises(Exception):
        analysis_agent.execute_analysis("Invalid query", "nonexistent_table")

def test_end_to_end_analysis(analysis_agent):
    """Test end-to-end analysis workflow."""
    query = "What was the total revenue by product?"
    results = analysis_agent.execute_analysis(query, "test_table")
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert "revenue" in results[0]
    assert "product" in results[0]

def test_explain_results(analysis_agent):
    """Test generating explanation for analysis results."""
    results = {
        "results": [
            {"date": "2024-01-01", "product": "A", "revenue": 100},
            {"date": "2024-01-02", "product": "B", "revenue": 200}
        ]
    }
    
    explanation = analysis_agent.explain_results(results)
    assert isinstance(explanation, str)
    assert len(explanation) > 0

@pytest.mark.integration
def test_end_to_end_analysis(analysis_agent):
    """Integration test for end-to-end analysis flow."""
    # Create test data
    test_data = pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=10),
        "category": ["A", "B"] * 5,
        "revenue": range(100, 1100, 100)
    })
    
    try:
        # Create table and insert data
        analysis_agent.db.execute_query("DROP TABLE IF EXISTS test_sales;")
        analysis_agent.db.execute_query("""
            CREATE TABLE test_sales (
                date timestamp,
                category varchar(50),
                revenue numeric
            );
        """)
        analysis_agent.db.ingest_dataframe(test_data, "test_sales")
        
        # Execute analysis
        query = "What is the total revenue by category this year?"
        result = analysis_agent.execute_analysis(query, "test_sales")
        
        # Verify complete flow
        assert "query_understanding" in result
        assert "sql_query" in result
        assert "results" in result
        assert "insights" in result
        
        # Verify data
        assert len(result["results"]) > 0
        total_revenue = sum(r["revenue"] for r in result["results"])
        assert abs(total_revenue - test_data["revenue"].sum()) < 0.01  # Account for floating point
        
    finally:
        # Cleanup
        analysis_agent.db.execute_query("DROP TABLE IF EXISTS test_sales;")

def test_error_handling(analysis_agent):
    """Test error handling in analysis execution."""
    # Simulate database error
    analysis_agent.db.execute_query.side_effect = Exception("Database connection failed")
    
    with pytest.raises(Exception) as exc_info:
        analysis_agent.execute_analysis("What was the revenue?", "sales")
    
    assert "Analysis execution failed" in str(exc_info.value)

@pytest.mark.parametrize("query,expected_format", [
    (
        "Show me total revenue by product for last month",
        {
            "json_tags": "<o>",
            "required_fields": ["metrics", "dimensions", "time_range", "filters", "sort"],
            "metrics_type": list,
            "dimensions_type": list
        }
    ),
    (
        "Generate SQL to calculate daily sales",
        {
            "sql_tags": "<sql>",
            "sql_start": "SELECT",
            "has_from": True,
            "has_groupby": True
        }
    )
])
def test_model_output_formatting(analysis_agent, query, expected_format):
    """Test that model outputs follow the expected format and structure."""
    logger.info(f"Testing model output formatting for query: {query}")
    
    try:
        if "json_tags" in expected_format:
            # Test query parsing format
            result = analysis_agent.parse_user_query(query)
            logger.info(f"Query parsing result: {result}")
            
            # Verify required fields
            for field in expected_format["required_fields"]:
                assert field in result, f"Missing required field: {field}"
            
            # Verify field types
            if "metrics_type" in expected_format:
                assert isinstance(result["metrics"], expected_format["metrics_type"]), "Metrics should be a list"
            if "dimensions_type" in expected_format:
                assert isinstance(result["dimensions"], expected_format["dimensions_type"]), "Dimensions should be a list"
                
        elif "sql_tags" in expected_format:
            # First parse the query to get parameters
            params = analysis_agent.parse_user_query(query)
            
            # Test SQL generation format
            sql = analysis_agent.generate_sql("test_table", params)
            logger.info(f"Generated SQL: {sql}")
            
            # Verify SQL format
            if expected_format.get("sql_start"):
                assert sql.strip().upper().startswith(expected_format["sql_start"]), f"SQL should start with {expected_format['sql_start']}"
            if expected_format.get("has_from"):
                assert "FROM" in sql.upper(), "SQL should contain FROM clause"
            if expected_format.get("has_groupby"):
                assert "GROUP BY" in sql.upper(), "SQL should contain GROUP BY clause"
    
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise 