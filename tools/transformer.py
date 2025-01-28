from typing import Dict, List, Any, Union, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataTransformer:
    """Tool for data transformation and preprocessing operations."""
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        Initialize the DataTransformer.
        
        Args:
            llm: Optional LangChain LLM instance for advanced transformations
        """
        logger.info("Initializing DataTransformer")
        self.llm = llm
        if llm:
            logger.info("LLM provided for advanced transformations")
        else:
            logger.info("No LLM provided, advanced transformations will be unavailable")

    def preprocess_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert raw data to pandas DataFrame and perform basic preprocessing.
        
        Args:
            data (List[Dict[str, Any]]): Raw data from database
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing")
        logger.debug(f"Input data size: {len(data)} records")
        
        try:
            df = pd.DataFrame(data)
            logger.debug(f"Created DataFrame with shape: {df.shape}")
            
            # Handle missing values
            missing_before = df.isnull().sum().sum()
            df = df.fillna({
                col: 0 if df[col].dtype in ['int64', 'float64'] else 'unknown'
                for col in df.columns
            })
            missing_after = df.isnull().sum().sum()
            logger.info(f"Filled {missing_before - missing_after} missing values")
            
            # Convert date columns
            date_columns = [
                col for col in df.columns
                if any(date_type in col.lower() for date_type in ['date', 'time', 'timestamp'])
            ]
            
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    logger.debug(f"Converted {col} to datetime")
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to datetime: {str(e)}")
                    continue
                    
            logger.info("Data preprocessing completed successfully")
            return df
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise

    def aggregate_data(
        self,
        df: pd.DataFrame,
        group_by: Union[str, List[str]],
        metrics: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Aggregate data based on specified grouping and metrics.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            group_by (Union[str, List[str]]): Column(s) to group by
            metrics (Dict[str, str]): Metrics to calculate (column: operation)
            
        Returns:
            pd.DataFrame: Aggregated DataFrame
        """
        logger.info("Starting data aggregation")
        logger.debug(f"Grouping by: {group_by}")
        logger.debug(f"Calculating metrics: {metrics}")
        
        try:
            if isinstance(group_by, str):
                group_by = [group_by]
                
            result = df.groupby(group_by).agg(metrics).reset_index()
            logger.info(f"Successfully aggregated data, result shape: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"Data aggregation failed: {str(e)}")
            raise

    def calculate_time_series_metrics(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        Calculate time series metrics like rolling averages and trends.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            date_column (str): Name of date column
            value_column (str): Name of value column
            freq (str): Frequency for resampling ('D' for daily, 'W' for weekly, etc.)
            
        Returns:
            pd.DataFrame: DataFrame with time series metrics
        """
        logger.info(f"Calculating time series metrics for {value_column}")
        logger.debug(f"Using frequency: {freq}")
        
        try:
            # Ensure date column is datetime
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
            logger.debug("Set datetime index")
            
            # Create a copy of the value column to avoid modifying the original
            df_agg = pd.DataFrame()
            df_agg[value_column] = df[value_column]
            
            # Resample and calculate metrics
            resampled = df_agg.resample(freq)
            result = pd.DataFrame()
            result[value_column] = resampled[value_column].sum()
            result[f'avg_{value_column}'] = resampled[value_column].mean()
            result[f'min_{value_column}'] = resampled[value_column].min()
            result[f'max_{value_column}'] = resampled[value_column].max()
            
            logger.debug("Calculated basic time series metrics")
            
            # Calculate rolling metrics
            result[f'rolling_avg_7d_{value_column}'] = result[value_column].rolling(window=7).mean()
            result[f'rolling_avg_30d_{value_column}'] = result[value_column].rolling(window=30).mean()
            logger.debug("Calculated rolling averages")
            
            # Calculate growth rates
            result[f'growth_rate_{value_column}'] = result[value_column].pct_change()
            logger.debug("Calculated growth rates")
            
            # Reset index to make date a column again
            result = result.reset_index()
            result = result.rename(columns={'index': date_column})
            
            logger.info("Successfully calculated all time series metrics")
            return result
        except Exception as e:
            logger.error(f"Time series calculation failed: {str(e)}")
            raise

    def generate_insights(self, df: pd.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate insights from the data using LangChain.
        
        Args:
            df (pd.DataFrame): Analyzed DataFrame
            context (Dict[str, Any]): Additional context for analysis
            
        Returns:
            Dict[str, Any]: Generated insights and recommendations
        """
        logger.info("Generating insights from data")
        
        if self.llm is None:
            logger.error("LLM not initialized for insight generation")
            raise ValueError("LLM not initialized for insight generation")
            
        try:
            # Create analysis summary
            summary = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'summary_stats': df.describe().to_dict(),
                'context': context or {}
            }
            logger.debug(f"Created data summary: {summary}")
            
            # Define prompt for insight generation
            prompt = PromptTemplate(
                input_variables=["data_summary"],
                template="""
                Based on the following data summary, generate key insights and recommendations:
                {data_summary}
                
                Please provide:
                1. Key Trends
                2. Notable Patterns
                3. Actionable Recommendations
                """
            )
            
            # Use the new chain syntax
            chain = prompt | self.llm
            insights = chain.invoke({"data_summary": str(summary)})
            logger.debug(f"Generated raw insights: {insights}")
            
            result = {
                'insights': insights,
                'summary_stats': summary['summary_stats'],
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'data_shape': summary['shape']
                }
            }
            
            logger.info("Successfully generated insights")
            return result
        except Exception as e:
            logger.error(f"Insight generation failed: {str(e)}")
            raise Exception(f"Insight generation failed: {str(e)}")

    def validate_data(self, df: pd.DataFrame) -> List[str]:
        """
        Validate data quality and identify potential issues.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            List[str]: List of validation messages
        """
        logger.info("Starting data validation")
        validations = []
        
        try:
            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                for col in missing_counts[missing_counts > 0].index:
                    msg = f"Found {missing_counts[col]} missing values in column '{col}'"
                    logger.warning(msg)
                    validations.append(msg)
            
            # Check for numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                # Check for negative values in typically positive columns
                if any(name in col.lower() for name in ['revenue', 'price', 'units', 'amount']):
                    neg_count = (df[col] < 0).sum()
                    if neg_count > 0:
                        msg = f"Found {neg_count} negative values in column '{col}'"
                        logger.warning(msg)
                        validations.append(msg)
                
                # Check for outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ]
                if not outliers.empty:
                    msg = (f"Found {len(outliers)} outliers in column '{col}' "
                          f"(outside range [{lower_bound:.2f}, {upper_bound:.2f}])")
                    logger.warning(msg)
                    validations.append(msg)
            
            # Check for duplicate rows
            duplicates = df.duplicated()
            if duplicates.any():
                msg = f"Found {duplicates.sum()} duplicate rows in the dataset"
                logger.warning(msg)
                validations.append(msg)
            
            # Check for unexpected values in categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                unique_values = df[col].nunique()
                if unique_values == 1:
                    msg = f"Column '{col}' has only one unique value"
                    logger.warning(msg)
                    validations.append(msg)
                elif unique_values > 100:  # Arbitrary threshold
                    msg = f"Column '{col}' has high cardinality ({unique_values} unique values)"
                    logger.warning(msg)
                    validations.append(msg)
            
            if not validations:
                msg = "All validation checks passed successfully"
                logger.info(msg)
                validations.append(msg)
            
            logger.info(f"Completed data validation with {len(validations)} findings")
            return validations
            
        except Exception as e:
            msg = f"Data validation failed: {str(e)}"
            logger.error(msg)
            validations.append(msg)
            return validations 