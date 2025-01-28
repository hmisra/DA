import os
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
import pandas as pd
import logging
from sqlalchemy.engine.url import make_url

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PostgresConnector:
    """Tool for managing PostgreSQL database connections and queries."""
    
    def __init__(self, db_url=None):
        """Initialize database connection parameters from environment variables or URL."""
        logger.info("Initializing PostgresConnector")
        self.db_params = {}
        
        if db_url:
            # Parse URL into parameters
            url = make_url(db_url)
            self.db_params = {
                'host': url.host or 'localhost',
                'port': url.port or 5432,
                'database': url.database,
                'user': url.username,
                'password': url.password
            }
        else:
            # Load from environment variables
            required_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
            param_mapping = {
                'DB_HOST': 'host',
                'DB_PORT': 'port',
                'DB_NAME': 'database',
                'DB_USER': 'user',
                'DB_PASSWORD': 'password'
            }
            
            try:
                for var in required_vars:
                    value = os.getenv(var)
                    if not value:
                        logger.error(f"Required environment variable {var} is not set")
                        raise ValueError(f"Required environment variable {var} is not set")
                    param_name = param_mapping[var]
                    self.db_params[param_name] = value
                    
                # Ensure port is an integer
                try:
                    self.db_params['port'] = int(self.db_params['port'])
                except ValueError:
                    logger.error("DB_PORT must be a valid integer")
                    raise ValueError("DB_PORT must be a valid integer")
            except Exception as e:
                logger.error(f"Failed to initialize from environment variables: {str(e)}")
                raise
        
        try:
            self.engine = self._create_engine()
            logger.info("Successfully initialized database connection")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {str(e)}")
            raise

    def _create_engine(self):
        """Create SQLAlchemy engine for database operations."""
        logger.debug("Creating SQLAlchemy engine")
        try:
            conn_str = (f"postgresql://{self.db_params['user']}:{self.db_params['password']}"
                       f"@{self.db_params['host']}:{self.db_params['port']}"
                       f"/{self.db_params['database']}")
            engine = create_engine(conn_str)
            logger.debug("Successfully created SQLAlchemy engine")
            return engine
        except Exception as e:
            logger.error(f"Failed to create SQLAlchemy engine: {str(e)}")
            raise

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results as a list of dictionaries.
        
        Args:
            query (str): SQL query to execute
            params (tuple, optional): Query parameters for parameterized queries
            
        Returns:
            List[Dict[str, Any]]: Query results as list of dictionaries
            
        Raises:
            Exception: If query execution fails
        """
        logger.info(f"Executing query: {query}")
        try:
            with psycopg2.connect(**self.db_params, cursor_factory=RealDictCursor) as conn:
                with conn.cursor() as cur:
                    if params:
                        cur.execute(query, params)
                    else:
                        cur.execute(query)
                    
                    # For SELECT queries, return results
                    if query.strip().upper().startswith("SELECT"):
                        results = cur.fetchall()
                        results_list = [dict(row) for row in results]
                        logger.info(f"Query returned {len(results_list)} rows")
                        return results_list
                    
                    # For other queries (DDL, DML), just return empty list
                    logger.info("Non-SELECT query executed successfully")
                    return []
                    
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise Exception(f"Query execution failed: {str(e)}")

    def ingest_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """
        Ingest a pandas DataFrame into the database.
        
        Args:
            df (pd.DataFrame): DataFrame to ingest
            table_name (str): Target table name
            if_exists (str): How to behave if table exists ('fail', 'replace', 'append')
            
        Returns:
            bool: True if ingestion was successful
            
        Raises:
            Exception: If data ingestion fails
        """
        logger.info(f"Ingesting DataFrame into table {table_name} (if_exists={if_exists})")
        logger.debug(f"DataFrame shape: {df.shape}")
        try:
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists=if_exists,
                index=False
            )
            logger.info("Successfully ingested DataFrame")
            return True
        except Exception as e:
            logger.error(f"Data ingestion failed: {str(e)}")
            raise Exception(f"Data ingestion failed: {str(e)}")

    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get the schema information for a table.
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            List[Dict[str, str]]: List of column definitions
        """
        logger.info(f"Getting schema for table: {table_name}")
        query = """
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position;
        """
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (table_name,))
                    schema = [dict(row) for row in cur.fetchall()]
                    logger.debug(f"Retrieved schema: {schema}")
                    return schema
        except Exception as e:
            logger.error(f"Failed to get table schema: {str(e)}")
            raise Exception(f"Failed to get table schema: {str(e)}")

    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        logger.info("Testing database connection")
        try:
            with psycopg2.connect(**self.db_params):
                logger.info("Database connection test successful")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False 