from typing import List, Dict, Any
import json

class Prompts:
    def get_sql_generation_prompt(self, table_name: str, schema: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
        """Get prompt for SQL generation."""
        metrics = params.get('metrics', [])
        time_range = params.get('time_range', {})
        dimensions = params.get('dimensions', [])
        filters = params.get('filters', {})
        sort = params.get('sort', {})
        
        prompt = f"""Generate a SQL query for the table {table_name} with the following parameters:

Metrics: {metrics}
Time Range: {time_range}
Dimensions: {dimensions}
Filters: {filters}
Sort: {sort}

Table Schema:
{json.dumps(schema, indent=2)}

Requirements:
1. Query MUST start with SELECT
2. Use exact column names from schema (case sensitive)
3. Use BETWEEN for date ranges
4. Include GROUP BY if using aggregations
5. Use table name exactly as provided
6. For dates, use format: date BETWEEN '2024-01-01' AND '2024-12-31'
7. For dimensions, use exact column names from schema
8. For metrics, use SUM() for aggregation
9. For sorting, use ORDER BY with the specified direction

Return ONLY the SQL query, no explanation or other text.
"""
        return prompt 
