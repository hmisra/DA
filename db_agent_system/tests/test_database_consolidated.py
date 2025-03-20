"""
Test module for the database components.
This consolidated test file covers database connection and tools functionality.
"""
import pytest

from conftest import BaseTestClass
from src.langchain.database import (
    DatabaseConnection,
    ListTablesTool,
    GetTableSchemaTool,
    ValidateSQLQueryTool,
    ExecuteSQLQueryTool,
    get_db_tools
)
from src.crewai.tools import adapt_langchain_tools


class TestDatabaseConnection(BaseTestClass):
    """Test the DatabaseConnection class."""

    def test_singleton_instance(self):
        """Test that DatabaseConnection is a singleton."""
        # Create the first instance
        db1 = DatabaseConnection()
        # Create the second instance
        db2 = DatabaseConnection()
        # Check they are the same object
        assert db1 is db2

    def test_get_table_names(self):
        """Test get_table_names method."""
        db = DatabaseConnection()
        tables = db.get_table_names()
        assert "products" in tables, "Products table not found"
        assert "customers" in tables, "Customers table not found"
        assert "orders" in tables, "Orders table not found"
        print(f"Tables: {tables}")

    def test_get_table_schema(self):
        """Test get_table_schema method."""
        db = DatabaseConnection()
        schema = db.get_table_schema("products")
        assert len(schema) > 0, "No schema returned"
        
        # Check if common fields exist
        column_names = [col["name"] for col in schema]
        assert "id" in column_names, "id column not found"
        assert "name" in column_names, "name column not found"
        assert "price" in column_names, "price column not found"
        print(f"Schema: {schema}")

    def test_execute_query(self):
        """Test execute_query method."""
        db = DatabaseConnection()
        
        # Test a simple query
        results, error = db.execute_query("SELECT * FROM products LIMIT 5")
        assert error is None, f"Query error: {error}"
        assert len(results) > 0, "No results returned"
        print(f"Query results: {results}")
        
        # Test a join query
        join_results, error = db.execute_query(
            """
            SELECT c.name as customer_name, p.name as product_name, o.total_amount
            FROM customers c
            JOIN orders o ON c.id = o.customer_id
            JOIN order_items oi ON o.id = oi.order_id
            JOIN products p ON oi.product_id = p.id
            WHERE p.category = 'Electronics'
            LIMIT 10
            """
        )
        assert error is None, f"Join query error: {error}"
        assert len(join_results) > 0, "No join results returned"
        print(f"Join query results: {join_results}")


class TestDatabaseTools(BaseTestClass):
    """Test the database tools."""
    
    def test_list_tables_tool(self):
        """Test ListTablesTool."""
        tool = ListTablesTool()
        result = tool._run()
        assert isinstance(result, list), "Result is not a list"
        assert "products" in result, "products table not found"
        assert "customers" in result, "customers table not found"
        assert "orders" in result, "orders table not found"
        print(f"Tables: {result}")

    def test_get_table_schema_tool(self):
        """Test GetTableSchemaTool."""
        tool = GetTableSchemaTool()
        result = tool._run("products")
        assert isinstance(result, list), "Result is not a list"
        assert len(result) > 0, "Empty schema returned"
        
        # Check if returned schema has expected structure
        first_column = result[0]
        assert "name" in first_column, "Column name missing"
        assert "type" in first_column, "Column type missing"
        print(f"Schema: {result}")

    def test_validate_sql_query_tool(self):
        """Test ValidateSQLQueryTool."""
        tool = ValidateSQLQueryTool()
        result = tool._run("SELECT * FROM products LIMIT 5")
        assert isinstance(result, dict), "Result is not a dictionary"
        assert result["is_valid"], "Query incorrectly marked as invalid"
        assert "products" in result["tables_referenced"], "Table reference not detected"
        print(f"Validation result: {result}")

    def test_execute_sql_query_tool(self):
        """Test ExecuteSQLQueryTool."""
        tool = ExecuteSQLQueryTool()
        result = tool._run("SELECT * FROM products LIMIT 5")
        assert isinstance(result, dict), "Result is not a dictionary"
        assert result["error"] is None, f"Query error: {result['error']}"
        assert len(result["results"]) > 0, "No results returned"
        assert result["row_count"] > 0, "Row count is zero"
        print(f"Execution result: {result}")
    
    def test_langchain_tool_adapter(self):
        """Test the LangChain tool adapter."""
        # Get database tools
        db_tools = get_db_tools()
        assert len(db_tools) > 0, "No database tools found"
        
        # Convert to CrewAI tools
        crewai_tools = adapt_langchain_tools(db_tools)
        assert len(crewai_tools) == len(db_tools), "Tool count mismatch after adaptation"
        
        # Check tool properties
        for i, tool in enumerate(crewai_tools):
            assert tool.name == db_tools[i].name, f"Tool name mismatch: {tool.name} != {db_tools[i].name}"
            assert tool.description, f"Tool description missing for {tool.name}"
            assert hasattr(tool, "_run"), f"Tool _run method missing for {tool.name}"
        
        print(f"\nSuccessfully adapted {len(db_tools)} LangChain tools to CrewAI tools")
        
        # Test running a tool
        list_tables_tool = next((t for t in crewai_tools if t.name == "list_tables"), None)
        assert list_tables_tool, "List tables tool not found"
        
        tables = list_tables_tool._run(tool_input="")
        assert isinstance(tables, list), "Tool did not return a list"
        assert "products" in tables, "Products table not found in tool results"
        
        print(f"\nTool execution successful. Tables found: {', '.join(tables)}") 