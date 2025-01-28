import pytest
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
from analysis.analysis_agent import AnalysisAgent
from ingestion.ingest_data import DataIngestionManager
from tools.database import PostgresConnector
from tools.transformer import DataTransformer
from langchain_ollama import OllamaLLM
import os

@pytest.fixture(scope="module")
def test_data():
    """Create test dataset."""
    return pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=100),
        "product": ["A", "B", "C", "D"] * 25,
        "category": ["X", "X", "Y", "Y"] * 25,
        "revenue": range(100, 10100, 100),
        "units": range(1, 101)
    })

@pytest.fixture(scope="module")
def db_connector(test_db_params):
    """Create database connector with test configuration."""
    connector = PostgresConnector()
    connector.db_params = test_db_params.copy()
    connector.engine = connector._create_engine()
    return connector

@pytest.fixture(scope="module")
def ingestion_manager(db_connector):
    """Create data ingestion manager."""
    manager = DataIngestionManager()
    manager.db = db_connector
    return manager

@pytest.fixture(scope="module")
def llm():
    """Create OllamaLLM instance with DeepSeek model."""
    return OllamaLLM(
        model="deepseek-r1:1.5b",
        temperature=0.1,
        stop=["</think>"],
        callback_manager=None
    )

@pytest.fixture(scope="module")
def db():
    """Create real database connection for integration tests."""
    db_url = os.getenv("TEST_DATABASE_URL", "postgresql://localhost:5432/test_db")
    return PostgresConnector(db_url)

@pytest.fixture(scope="module")
def analysis_agent(db, llm):
    """Create AnalysisAgent instance with real components."""
    return AnalysisAgent(db=db, llm=llm)

@pytest.mark.integration
class TestEndToEndFlow:
    """Integration tests for complete data analysis workflow."""
    
    def setup_method(self, method):
        """Set up test environment before each test."""
        self.table_name = f"test_{method.__name__.lower()}"
    
    def teardown_method(self, method):
        """Clean up after each test."""
        try:
            self.analysis_agent.db.execute_query(f"DROP TABLE IF EXISTS {self.table_name};")
        except Exception:
            pass
    
    def test_data_ingestion_and_query(self, test_data, ingestion_manager, analysis_agent):
        """Test complete flow from data ingestion to analysis."""
        self.analysis_agent = analysis_agent
        
        # Create test table
        schema = {
            "date": "timestamp",
            "product": "varchar(50)",
            "category": "varchar(50)",
            "revenue": "numeric",
            "units": "integer"
        }
        
        try:
            # Create table and ingest data
            ingestion_manager.create_table_schema(self.table_name, schema, drop_existing=True)
            ingestion_manager.db.ingest_dataframe(test_data, self.table_name)
            
            # Execute analysis
            query = "What is the total revenue by product category this year?"
            results = analysis_agent.execute_analysis(query, self.table_name)
            
            # Verify results
            assert results["query_understanding"]["metrics"] == ["revenue"]
            assert "category" in results["query_understanding"]["dimensions"]
            assert len(results["results"]) > 0
            
            # Verify aggregations
            total_revenue = sum(r["revenue"] for r in results["results"])
            assert abs(total_revenue - test_data["revenue"].sum()) < 0.01
            
        finally:
            # Cleanup handled by teardown_method
            pass
    
    def test_time_series_analysis(self, test_data, ingestion_manager, analysis_agent):
        """Test time series analysis capabilities."""
        self.analysis_agent = analysis_agent
        
        try:
            # Create table and ingest data
            schema = {
                "date": "timestamp",
                "product": "varchar(50)",
                "revenue": "numeric"
            }
            ingestion_manager.create_table_schema(self.table_name, schema, drop_existing=True)
            
            # Prepare time series data
            ts_data = test_data[["date", "product", "revenue"]].copy()
            ingestion_manager.db.ingest_dataframe(ts_data, self.table_name)
            
            # Execute analysis
            query = "Show me the daily revenue trend over time"
            results = analysis_agent.execute_analysis(query, self.table_name)
            
            # Verify results
            assert "time" in results["query_understanding"]["dimensions"]
            assert len(results["results"]) > 0
            assert "insights" in results
            
        finally:
            # Cleanup handled by teardown_method
            pass
    
    def test_multi_metric_analysis(self, test_data, ingestion_manager, analysis_agent):
        """Test analysis with multiple metrics."""
        self.analysis_agent = analysis_agent
        
        try:
            # Create table and ingest data
            schema = {
                "date": "timestamp",
                "product": "varchar(50)",
                "revenue": "numeric",
                "units": "integer"
            }
            ingestion_manager.create_table_schema(self.table_name, schema, drop_existing=True)
            
            # Prepare multi-metric data
            mm_data = test_data[["date", "product", "revenue", "units"]].copy()
            ingestion_manager.db.ingest_dataframe(mm_data, self.table_name)
            
            # Execute analysis
            query = "What are the total revenue and units sold by product?"
            results = analysis_agent.execute_analysis(query, self.table_name)
            
            # Verify results
            metrics = results["query_understanding"]["metrics"]
            assert "revenue" in metrics
            assert "units" in metrics
            assert len(results["results"]) > 0
            
        finally:
            # Cleanup handled by teardown_method
            pass
    
    def test_data_validation_flow(self, test_data, ingestion_manager, analysis_agent):
        """Test data validation in analysis flow."""
        self.analysis_agent = analysis_agent
        
        try:
            # Create table with some invalid data
            schema = {
                "date": "timestamp",
                "product": "varchar(50)",
                "revenue": "numeric"
            }
            ingestion_manager.create_table_schema(self.table_name, schema, drop_existing=True)
            
            # Add some anomalies to the data
            invalid_data = test_data[["date", "product", "revenue"]].copy()
            invalid_data.loc[0, "revenue"] = -1000  # Negative revenue
            invalid_data.loc[1, "revenue"] = 1000000  # Outlier
            
            ingestion_manager.db.ingest_dataframe(invalid_data, self.table_name)
            
            # Execute analysis
            query = "Analyze the revenue patterns and identify any anomalies"
            results = analysis_agent.execute_analysis(query, self.table_name)
            
            # Verify results
            assert "insights" in results
            insights_text = str(results["insights"]).lower()
            assert any(word in insights_text for word in ["anomaly", "outlier", "pattern"])
            
        finally:
            # Cleanup handled by teardown_method
            pass
    
    def test_insight_generation(self, test_data, ingestion_manager, analysis_agent):
        """Test insight generation capabilities."""
        self.analysis_agent = analysis_agent
        
        try:
            # Create table and ingest data
            schema = {
                "date": "timestamp",
                "product": "varchar(50)",
                "category": "varchar(50)",
                "revenue": "numeric",
                "units": "integer"
            }
            ingestion_manager.create_table_schema(self.table_name, schema, drop_existing=True)
            ingestion_manager.db.ingest_dataframe(test_data, self.table_name)
            
            # Execute analysis with insight generation
            query = """
            Provide a comprehensive analysis of sales performance:
            1. Revenue trends over time
            2. Product category performance
            3. Key insights and recommendations
            """
            results = analysis_agent.execute_analysis(query, self.table_name)
            
            # Verify insights
            assert "insights" in results
            insights = results["insights"]
            assert isinstance(insights, dict)
            assert "insights" in insights
            
            # Check insight content
            insight_text = str(insights).lower()
            assert any(word in insight_text for word in [
                "trend", "performance", "recommendation", "pattern"
            ])
            
        finally:
            # Cleanup handled by teardown_method
            pass

@pytest.mark.integration
def test_error_recovery(ingestion_manager, analysis_agent):
    """Test system's ability to handle and recover from errors."""
    # Test with invalid table
    with pytest.raises(Exception) as exc_info:
        analysis_agent.execute_analysis(
            "What was the revenue last month?",
            "nonexistent_table"
        )
    assert "Analysis execution failed" in str(exc_info.value)
    
    # Test with invalid query
    results = analysis_agent.execute_analysis(
        "Show me the XYZ metric that doesn't exist",
        "test_sales"
    )
    validations = analysis_agent.validate_results(results)
    assert any("no data found" in msg.lower() for msg in validations)

@pytest.mark.integration
def test_data_transformation_flow(test_data, ingestion_manager, analysis_agent):
    """Test data transformation capabilities in the analysis flow."""
    try:
        # Ingest raw data
        ingestion_manager.db.ingest_dataframe(test_data, "test_transform", if_exists="replace")
        
        # Create transformer
        transformer = DataTransformer()
        
        # Test different transformations
        queries = [
            "What is the daily revenue trend?",
            "Show me the weekly average revenue by product",
            "Calculate the month-over-month growth rate"
        ]
        
        for query in queries:
            results = analysis_agent.execute_analysis(query, "test_transform")
            df_results = pd.DataFrame(results["results"])
            
            # Verify transformations
            assert not df_results.empty
            assert all(col in df_results.columns for col in ["date", "revenue"])
            
            if "trend" in query.lower():
                assert any(col.startswith("rolling_avg") for col in df_results.columns)
            if "growth" in query.lower():
                assert "growth_rate" in df_results.columns
                
    finally:
        # Cleanup
        ingestion_manager.db.execute_query("DROP TABLE IF EXISTS test_transform;")

@pytest.mark.integration
def test_end_to_end_analysis(analysis_agent):
    """Test complete analysis workflow with real components."""
    # Create test data
    test_data = pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=10),
        "category": ["A", "B"] * 5,
        "revenue": range(100, 1100, 100)
    })
    
    try:
        # Setup test table
        analysis_agent.db.execute_query("DROP TABLE IF EXISTS test_sales;")
        analysis_agent.db.execute_query("""
            CREATE TABLE test_sales (
                date timestamp,
                category varchar(50),
                revenue numeric
            );
        """)
        
        # Insert test data
        for _, row in test_data.iterrows():
            analysis_agent.db.execute_query(
                """
                INSERT INTO test_sales (date, category, revenue)
                VALUES (%s, %s, %s);
                """,
                (row["date"], row["category"], row["revenue"])
            )
        
        # Test query parsing
        query = "What is the total revenue by category this year?"
        params = analysis_agent.parse_user_query(query)
        assert "metrics" in params
        assert "revenue" in params["metrics"]
        assert "dimensions" in params
        assert "category" in params["dimensions"]
        
        # Test SQL generation
        sql = analysis_agent.generate_sql("test_sales", params)
        assert sql.lower().startswith("select")
        assert "category" in sql.lower()
        assert "sum" in sql.lower()
        assert "revenue" in sql.lower()
        
        # Test query execution
        results = analysis_agent.execute_analysis(query, "test_sales")
        assert len(results) > 0
        assert all(r["revenue"] > 0 for r in results)
        
        # Test results validation
        validations = analysis_agent.validate_results({"results": results})
        assert isinstance(validations, list)
        assert len(validations) > 0
        
        # Test results explanation
        explanation = analysis_agent.explain_results({"results": results})
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        
    finally:
        # Cleanup
        analysis_agent.db.execute_query("DROP TABLE IF EXISTS test_sales;") 