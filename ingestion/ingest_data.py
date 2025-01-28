import pandas as pd
from typing import Dict, Any, Optional, Union
import json
import requests
from pathlib import Path
from tools.database import PostgresConnector

class DataIngestionManager:
    """Manages data ingestion from various sources into PostgreSQL."""
    
    def __init__(self):
        """Initialize the data ingestion manager."""
        self.db = PostgresConnector()

    def ingest_csv(
        self,
        file_path: Union[str, Path],
        table_name: str,
        if_exists: str = 'append',
        **csv_kwargs
    ) -> bool:
        """
        Ingest data from a CSV file into PostgreSQL.
        
        Args:
            file_path (Union[str, Path]): Path to CSV file
            table_name (str): Target table name
            if_exists (str): How to behave if table exists
            **csv_kwargs: Additional arguments for pd.read_csv
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df = pd.read_csv(file_path, **csv_kwargs)
            self.db.ingest_dataframe(df, table_name, if_exists)
            return True
        except Exception as e:
            print(f"CSV ingestion failed: {str(e)}")
            return False

    def ingest_json(
        self,
        json_data: Union[str, Path, Dict],
        table_name: str,
        if_exists: str = 'append',
        orient: str = 'records'
    ) -> bool:
        """
        Ingest JSON data into PostgreSQL.
        
        Args:
            json_data (Union[str, Path, Dict]): JSON data or file path
            table_name (str): Target table name
            if_exists (str): How to behave if table exists
            orient (str): Expected JSON structure
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if isinstance(json_data, (str, Path)):
                with open(json_data, 'r') as f:
                    data = json.load(f)
            else:
                data = json_data
                
            df = pd.json_normalize(data) if orient == 'records' else pd.DataFrame(data)
            self.db.ingest_dataframe(df, table_name, if_exists)
            return True
        except Exception as e:
            print(f"JSON ingestion failed: {str(e)}")
            return False

    def ingest_api_data(
        self,
        api_url: str,
        table_name: str,
        if_exists: str = 'append',
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data_path: Optional[str] = None
    ) -> bool:
        """
        Ingest data from an API endpoint into PostgreSQL.
        
        Args:
            api_url (str): API endpoint URL
            table_name (str): Target table name
            if_exists (str): How to behave if table exists
            headers (Optional[Dict[str, str]]): Request headers
            params (Optional[Dict[str, Any]]): Query parameters
            data_path (Optional[str]): JSON path to data array
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data_path:
                for key in data_path.split('.'):
                    data = data[key]
                    
            df = pd.DataFrame(data)
            self.db.ingest_dataframe(df, table_name, if_exists)
            return True
        except Exception as e:
            print(f"API data ingestion failed: {str(e)}")
            return False

    def create_table_schema(
        self,
        table_name: str,
        schema: Dict[str, str],
        drop_existing: bool = False
    ) -> bool:
        """
        Create a table with the specified schema.
        
        Args:
            table_name (str): Name of the table to create
            schema (Dict[str, str]): Column definitions
            drop_existing (bool): Whether to drop existing table
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate CREATE TABLE SQL
            columns = [f"{col} {dtype}" for col, dtype in schema.items()]
            create_sql = f"""
            CREATE TABLE {table_name} (
                {', '.join(columns)}
            );
            """
            
            if drop_existing:
                drop_sql = f"DROP TABLE IF EXISTS {table_name};"
                self.db.execute_query(drop_sql)
                
            self.db.execute_query(create_sql)
            return True
        except Exception as e:
            print(f"Schema creation failed: {str(e)}")
            return False

    def validate_data(self, df: pd.DataFrame, schema: Dict[str, Any]) -> bool:
        """
        Validate DataFrame against expected schema.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            schema (Dict[str, Any]): Expected schema
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required columns
            missing_cols = set(schema.keys()) - set(df.columns)
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return False
            
            # Validate data types
            for col, dtype in schema.items():
                if not df[col].dtype.name == dtype:
                    try:
                        df[col] = df[col].astype(dtype)
                    except Exception:
                        print(f"Column {col} cannot be converted to {dtype}")
                        return False
            
            return True
        except Exception as e:
            print(f"Data validation failed: {str(e)}")
            return False 