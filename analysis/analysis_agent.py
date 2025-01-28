from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from tools.database import PostgresConnector
from tools.transformer import DataTransformer
import pandas as pd
import logging
import re
from langchain_core.language_models import BaseLLM
from prompts.prompts import Prompts

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)8s] %(message)s\n',  # Added newline for better readability
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def log_model_io(logger, title: str, content: str):
    """Log model input/output with clear formatting."""
    separator = "\n" + "="*120 + "\n"
    logger.debug(f"{separator}{title}:{separator}{content}{separator}")

class AnalysisAgent:
    """Agent for executing data analysis tasks."""
    
    def __init__(self, db: PostgresConnector, llm: BaseLLM, transformer: Optional[DataTransformer] = None):
        """Initialize the analysis agent."""
        self.logger = logger
        self.logger.info("Initializing AnalysisAgent")
        self.db = db
        
        # Configure LLM with stop sequences and response format
        if isinstance(llm, OllamaLLM):
            # Set stop sequences to ensure we get complete tags
            llm.stop = ["</o>", "</sql>", "</think>", "</system>", "</user>", "</assistant>"]
            
            # Set temperature to 0 for deterministic outputs
            llm.temperature = 0
            
            # Increase context and response length to ensure complete responses
            llm.num_predict = 2000
            llm.num_ctx = 4096
            
            # Set sampling parameters for more controlled outputs
            llm.top_k = 1  # Only consider the most likely token
            llm.top_p = 0.1  # Very focused sampling
            llm.repeat_penalty = 1.2  # Stronger penalty for repetition
            
            # Set model and format
            llm.model = "deepseek-r1:70b"
            llm.format = "json"
            
            # Additional parameters for better formatting
            llm.mirostat = 2  # Enable Mirostat sampling
            llm.mirostat_tau = 3.0  # Lower value for more focused responses
            llm.mirostat_eta = 0.1  # Lower value for more conservative updates
        
        self.llm = llm
        self.transformer = transformer
        self.prompts = Prompts()
        self.sql_prompt = PromptTemplate(
            input_variables=["query_params", "table_schema"],
            template="""Generate a PostgreSQL SELECT query based on the following parameters:

Query Parameters: {query_params}
Table Schema: {table_schema}

Requirements:
1. Start with SELECT and specify exact columns needed
2. Include appropriate aggregations (SUM, AVG, etc.) for metrics
3. Add GROUP BY for any dimensions
4. Include WHERE clause for time range and filters
5. Add ORDER BY for sorting

Your response MUST be wrapped in <sql></sql> tags.
Return ONLY the SQL query within the tags, no other text.

Example:
<sql>
SELECT 
    product,
    category,
    SUM(revenue) as total_revenue,
    COUNT(*) as count
FROM sales
WHERE date BETWEEN '2024-01-01' AND '2024-03-31'
GROUP BY product, category
ORDER BY total_revenue DESC;
</sql>

Return ONLY your SQL query between <sql></sql> tags with no other text:"""
        )
        self.logger.info("Successfully initialized all components")
        
    def parse_user_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query into structured parameters."""
        self.logger.info(f"Parsing user query: {query}")
        
        # Generate prompt for query parsing
        prompt = self.prompts.get_query_parsing_prompt(query)
        log_model_io(self.logger, "PROMPT TO MODEL", prompt)
        
        # Wrap prompt in list for LangChain LLM
        self.logger.debug("Sending prompt to LLM...")
        response = self.llm.generate([prompt])
        
        if not response.generations:
            self.logger.error("LLM returned no generations")
            raise Exception("Failed to parse query")
            
        raw_response = response.generations[0][0].text
        log_model_io(self.logger, "RAW MODEL RESPONSE", raw_response)
        
        cleaned_response = self._extract_json_from_response(raw_response)
        log_model_io(self.logger, "CLEANED JSON RESPONSE", cleaned_response)
        
        try:
            query_params = json.loads(cleaned_response)
            self.logger.info(f"Successfully parsed query into: {json.dumps(query_params, indent=2)}")
            return query_params
        except Exception as e:
            self.logger.error(f"Failed to parse JSON: {cleaned_response}")
            self.logger.error(f"Error: {str(e)}")
            raise Exception(f"Query parsing failed: {str(e)}")

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON object from LLM response."""
        log_model_io(self.logger, "EXTRACTING JSON FROM RESPONSE", response)
        
        # Extract content between <o> tags, ignoring any thinking/explanations
        pattern = r"<o>(.*?)</o>"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if not match:
            self.logger.error("No <o> tags found in response")
            log_model_io(self.logger, "PROCESSED RESPONSE (NO TAGS FOUND)", response)
            raise Exception("No <o> tags found in response")
            
        json_str = match.group(1).strip()
        log_model_io(self.logger, "EXTRACTED JSON", json_str)
        return json_str

    def generate_sql(self, table_name: str, query_params: Dict[str, Any]) -> str:
        """Generate SQL query based on parsed parameters."""
        self.logger.info(f"Generating SQL for table {table_name} with params: {json.dumps(query_params, indent=2)}")
        
        # Get table schema
        schema = self.db.get_table_schema(table_name)
        self.logger.debug(f"Retrieved schema: {json.dumps(schema, indent=2)}")
        
        if not schema:
            raise Exception(f"Failed to get table schema for {table_name}")
            
        prompt = self.sql_prompt.format(
            query_params=json.dumps(query_params, indent=2),
            table_schema=json.dumps(schema, indent=2)
        )
        log_model_io(self.logger, "SQL PROMPT TO MODEL", prompt)
        
        # Wrap prompt in list for LangChain LLM
        self.logger.debug("Sending prompt to LLM...")
        response = self.llm.generate([prompt])
        
        if not response.generations:
            self.logger.error("LLM returned no generations")
            raise Exception("Failed to generate SQL query")
            
        raw_response = response.generations[0][0].text
        log_model_io(self.logger, "RAW SQL MODEL RESPONSE", raw_response)
        
        sql = self._extract_sql_from_response(raw_response)
        log_model_io(self.logger, "EXTRACTED SQL QUERY", sql)
        
        if not sql.strip().lower().startswith("select"):
            self.logger.error("Generated SQL does not start with SELECT")
            raise Exception("Generated SQL query must start with SELECT")
            
        return sql

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        log_model_io(self.logger, "EXTRACTING SQL FROM RESPONSE", response)
        
        # Extract content between <sql> tags, ignoring any thinking/explanations
        pattern = r"<sql>(.*?)</sql>"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if not match:
            self.logger.error("No <sql> tags found in response")
            log_model_io(self.logger, "PROCESSED RESPONSE (NO TAGS FOUND)", response)
            raise Exception("No <sql> tags found in response")
            
        sql = match.group(1).strip()
        
        if not sql:
            self.logger.error("Empty SQL query extracted")
            raise Exception("Empty SQL query")
            
        if not sql.strip().lower().startswith("select"):
            self.logger.error("Extracted SQL does not start with SELECT")
            raise Exception("Generated SQL query must start with SELECT")
            
        log_model_io(self.logger, "EXTRACTED SQL", sql)
        return sql

    def execute_analysis(self, user_query: str, table_name: str) -> Dict[str, Any]:
        """Execute analysis based on user query."""
        self.logger.info(f"Starting analysis for query: {user_query} on table: {table_name}")
        
        try:
            # Step 1: Parse user query
            self.logger.debug("Step 1: Parsing user query")
            query_params = self.parse_user_query(user_query)
            
            # Step 2: Generate SQL
            self.logger.debug("Step 2: Generating SQL")
            sql_query = self.generate_sql(table_name, query_params)
            
            # Step 3: Execute SQL
            self.logger.debug("Step 3: Executing SQL")
            results = self.db.execute_query(sql_query)
            
            # Step 4: Transform results if needed
            if self.transformer:
                self.logger.debug("Step 4: Transforming results")
                results = self.transformer.transform(results)
                
            return results
        except Exception as e:
            self.logger.error(f"Analysis execution failed: {str(e)}")
            raise Exception(f"Analysis execution failed: {str(e)}")

    def explain_results(self, results: Dict[str, Any]) -> str:
        """Generate natural language explanation of analysis results."""
        self.logger.info("Generating explanation for analysis results")
        prompt = PromptTemplate(
            input_variables=["results"],
            template="""You are a business analyst explaining data analysis results.

For the following results:
{results}

Provide a clear explanation focusing on:
1. Key findings and trends
2. Notable patterns or anomalies
3. Business implications
4. Actionable recommendations

IMPORTANT: Return ONLY the explanation, no other text or formatting.

Return ONLY the explanation:"""
        )
        
        try:
            self.logger.debug(f"Generated explanation prompt:\n{prompt}")
            chain = prompt | self.llm
            raw_explanation = chain.invoke({"results": json.dumps(results, indent=2)})
            self.logger.debug(f"Raw explanation response:\n{raw_explanation}")
            
            # Clean up the response
            explanation = raw_explanation.strip()
            if explanation.startswith('```'):
                explanation = explanation[explanation.find('\n')+1:]
            if explanation.endswith('```'):
                explanation = explanation[:explanation.rfind('\n')]
            explanation = explanation.strip()
            
            self.logger.debug(f"Cleaned explanation:\n{explanation}")
            return explanation
        except Exception as e:
            self.logger.error(f"Results explanation failed: {str(e)}")
            raise Exception(f"Results explanation failed: {str(e)}")

    def validate_results(self, results: Dict[str, Any]) -> List[str]:
        """
        Validate analysis results for potential issues.
        
        Args:
            results (Dict[str, Any]): Analysis results
            
        Returns:
            List[str]: List of validation messages
        """
        self.logger.info("Validating analysis results")
        validations = []
        
        try:
            # Check for empty results
            if not results.get('results'):
                msg = "Warning: No data found for the query"
                self.logger.warning(msg)
                validations.append(msg)
                
            # Check for potential data quality issues
            df = pd.DataFrame(results['results'])
            for col in df.select_dtypes(include=['number']).columns:
                # Check for outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
                if not outliers.empty:
                    msg = f"Warning: Potential outliers detected in {col}"
                    self.logger.warning(msg)
                    validations.append(msg)
                    
            # Check for missing values
            missing = df.isnull().sum()
            if missing.any():
                msg = "Warning: Missing values detected in results"
                self.logger.warning(msg)
                validations.append(msg)
                
            # Add success message if no issues found
            if not validations:
                msg = "Validation passed: No issues detected"
                self.logger.info(msg)
                validations.append(msg)
                
        except Exception as e:
            msg = f"Validation error: {str(e)}"
            self.logger.error(msg)
            validations.append(msg)
            
        return validations 