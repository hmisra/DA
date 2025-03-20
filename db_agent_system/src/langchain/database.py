"""
Database module for LangChain integration.
Provides tools to interact with the PostgreSQL database.
"""
from typing import List, Dict, Any, Optional, Tuple, ClassVar

from langchain_community.utilities import SQLDatabase
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, ToolException
from loguru import logger
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from src.utils.config import config
from src.utils.logging import AgentLogger


class DatabaseConnection:
    """Database connection singleton."""

    _instance = None
    _db = None
    _inspector = None
    _logger = None

    def __new__(cls):
        """Create a singleton instance.

        Returns:
            DatabaseConnection: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._logger = AgentLogger(
                agent_name="DatabaseConnection", agent_type="Database"
            )
            cls._connect()
        return cls._instance

    @classmethod
    def _connect(cls) -> None:
        """Connect to the database."""
        try:
            cls._logger.info(f"Connecting to database at {config.DB_HOST}")
            cls._db = SQLDatabase.from_uri(
                config.DB_URI,
                include_tables=None,  # Include all tables
                sample_rows_in_table_info=3,
                indexes_in_table_info=True,
                custom_table_info=None,
                view_support=True,
            )
            # Create inspector for schema information
            engine = cls._db._engine
            cls._inspector = inspect(engine)
            cls._logger.info("Database connection established successfully")
        except Exception as e:
            cls._logger.error(f"Failed to connect to database: {str(e)}", exc_info=True)
            raise

    @property
    def db(self) -> SQLDatabase:
        """Get the SQLDatabase instance.

        Returns:
            SQLDatabase: The database instance
        """
        return self._db

    @property
    def inspector(self):
        """Get the SQLAlchemy inspector.

        Returns:
            Inspector: The database inspector
        """
        return self._inspector
    
    def get_table_names(self) -> List[str]:
        """Get all table names in the database.

        Returns:
            List[str]: List of table names
        """
        return self._inspector.get_table_names()
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a table.

        Args:
            table_name: Name of the table

        Returns:
            List[Dict[str, Any]]: List of column information
        """
        if table_name not in self.get_table_names():
            raise ValueError(f"Table '{table_name}' does not exist")
        
        columns = self._inspector.get_columns(table_name)
        result = []
        
        for column in columns:
            result.append({
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": column.get("nullable", True),
                "default": column.get("default", None),
                "primary_key": column.get("primary_key", False),
            })
        
        return result
    
    def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Execute an SQL query.

        Args:
            query: SQL query to execute

        Returns:
            Tuple[List[Dict[str, Any]], Optional[str]]: Query results and error message if any
        """
        try:
            # Get the engine from the SQLDatabase
            engine = self._db._engine
            
            # Execute the query
            with engine.connect() as connection:
                result = connection.execute(text(query))
                
                # Convert to list of dictionaries
                rows = [dict(row._mapping) for row in result]
                
            return rows, None
        except SQLAlchemyError as e:
            error_msg = str(e)
            self._logger.error(f"SQL execution error: {error_msg}", exc_info=True)
            return [], error_msg


# Database Tools for LangChain

class ListTablesToolInput(BaseModel):
    """Input for ListTablesTool."""
    pass


class ListTablesTool(BaseTool):
    """Tool for listing all tables in the database."""
    
    name: ClassVar[str] = "list_tables"
    description: ClassVar[str] = "Lists all tables available in the database."
    args_schema = ListTablesToolInput
    
    model_config = {"extra": "allow"}
    
    def __init__(self):
        """Initialize the ListTablesTool."""
        super().__init__()
        self.db_conn = DatabaseConnection()
        self.logger = AgentLogger(agent_name="ListTablesTool", agent_type="Database Tool")
    
    def _run(self, **kwargs) -> List[str]:
        """Run the tool.

        Returns:
            List[str]: List of table names
        """
        self.logger.info("Listing all tables")
        try:
            tables = self.db_conn.get_table_names()
            self.logger.info(f"Found {len(tables)} tables")
            return tables
        except Exception as e:
            self.logger.error(f"Error listing tables: {str(e)}", exc_info=True)
            raise ToolException(f"Error listing tables: {str(e)}")


class GetTableSchemaToolInput(BaseModel):
    """Input for GetTableSchemaTool."""
    
    table_name: str = Field(..., description="Name of the table to get schema for")


class GetTableSchemaTool(BaseTool):
    """Tool for getting the schema of a table."""
    
    name: ClassVar[str] = "get_table_schema"
    description: ClassVar[str] = "Gets the schema (column names, types, etc.) for a specific table."
    args_schema = GetTableSchemaToolInput
    
    model_config = {"extra": "allow"}
    
    def __init__(self):
        """Initialize the GetTableSchemaTool."""
        super().__init__()
        self.db_conn = DatabaseConnection()
        self.logger = AgentLogger(agent_name="GetTableSchemaTool", agent_type="Database Tool")
    
    def _run(self, table_name: str) -> List[Dict[str, Any]]:
        """Run the tool.

        Args:
            table_name: Name of the table

        Returns:
            List[Dict[str, Any]]: List of column information
        """
        self.logger.info(f"Getting schema for table '{table_name}'")
        try:
            schema = self.db_conn.get_table_schema(table_name)
            self.logger.info(f"Retrieved schema with {len(schema)} columns")
            return schema
        except ValueError as e:
            self.logger.error(f"Table not found: {str(e)}")
            raise ToolException(str(e))
        except Exception as e:
            self.logger.error(f"Error getting table schema: {str(e)}", exc_info=True)
            raise ToolException(f"Error getting table schema: {str(e)}")


class ValidateSQLQueryToolInput(BaseModel):
    """Input for ValidateSQLQueryTool."""
    
    query: str = Field(..., description="SQL query to validate")


class ValidateSQLQueryTool(BaseTool):
    """Tool for validating SQL queries."""
    
    name: ClassVar[str] = "validate_sql_query"
    description: ClassVar[str] = "Validates an SQL query without executing it. Checks for syntax and schema validity."
    args_schema = ValidateSQLQueryToolInput
    
    model_config = {"extra": "allow"}
    
    def __init__(self):
        """Initialize the ValidateSQLQueryTool."""
        super().__init__()
        self.db_conn = DatabaseConnection()
        self.logger = AgentLogger(agent_name="ValidateSQLQueryTool", agent_type="Database Tool")
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Run the tool.

        Args:
            query: SQL query to validate

        Returns:
            Dict[str, Any]: Validation result
        """
        self.logger.info(f"Validating SQL query: {query}")
        
        # Basic validation logic
        result = {
            "is_valid": True,
            "error": None,
            "tables_referenced": [],
            "columns_referenced": []
        }
        
        # Extract table names (simple approach, not comprehensive)
        import re
        tables_pattern = r'(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)'
        tables = re.findall(tables_pattern, query, re.IGNORECASE)
        result["tables_referenced"] = list(set(tables))
        
        # Check if tables exist
        valid_tables = self.db_conn.get_table_names()
        for table in result["tables_referenced"]:
            if table not in valid_tables:
                result["is_valid"] = False
                result["error"] = f"Table '{table}' does not exist"
                self.logger.error(f"Validation failed: {result['error']}")
                return result
        
        # For a more comprehensive validation, we would need to parse the SQL
        # Here we'll just return the basic validation
        self.logger.info("SQL query passed basic validation")
        return result


class ExecuteSQLQueryToolInput(BaseModel):
    """Input for ExecuteSQLQueryTool."""
    
    query: str = Field(..., description="SQL query to execute")


class ExecuteSQLQueryTool(BaseTool):
    """Tool for executing SQL queries."""
    
    name: ClassVar[str] = "execute_sql_query"
    description: ClassVar[str] = "Executes an SQL query and returns the results."
    args_schema = ExecuteSQLQueryToolInput
    
    model_config = {"extra": "allow"}
    
    def __init__(self):
        """Initialize the ExecuteSQLQueryTool."""
        super().__init__()
        self.db_conn = DatabaseConnection()
        self.logger = AgentLogger(agent_name="ExecuteSQLQueryTool", agent_type="Database Tool")
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Run the tool.

        Args:
            query: SQL query to execute

        Returns:
            Dict[str, Any]: Query results
        """
        self.logger.info(f"Executing SQL query: {query}")
        
        # Execute the query
        results, error = self.db_conn.execute_query(query)
        
        response = {
            "success": error is None,
            "error": error,
            "row_count": len(results),
            "results": results
        }
        
        if error:
            self.logger.error(f"Query execution failed: {error}")
        else:
            self.logger.info(f"Query executed successfully. Returned {len(results)} rows")
            
        return response


# Factory function to get all database tools
def get_db_tools() -> List[BaseTool]:
    """Get all database tools.

    Returns:
        List[BaseTool]: List of database tools
    """
    return [
        ListTablesTool(),
        GetTableSchemaTool(),
        ValidateSQLQueryTool(),
        ExecuteSQLQueryTool()
    ] 