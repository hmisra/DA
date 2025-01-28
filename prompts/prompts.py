from typing import Dict, Any, List

class Prompts:
    """Class containing prompt templates for various tasks."""
    
    def get_query_parsing_prompt(self, query: str) -> str:
        """Get prompt for parsing natural language query into structured parameters."""
        return f"""<system>You are a query parser that converts natural language queries into structured JSON format.

CRITICAL: Your response MUST follow this EXACT format:
<o>
{{
    "metrics": [...],
    "dimensions": [...],
    "time_range": {{...}},
    "filters": {{...}},
    "sort": {{...}}
}}
</o>

Rules:
1. The <o> and </o> tags are MANDATORY - responses without these tags will cause ERRORS
2. Between these tags, output ONLY valid JSON
3. No other text, explanations, or formatting is allowed
4. All JSON must use double quotes for keys and string values
5. All field names must be lowercase
6. Dates must use ISO format YYYY-MM-DD
7. Arrays must use square brackets []
8. Objects must use curly braces {{}}
9. Empty fields must use appropriate empty structures ([] for arrays, {{}} for objects)
</system>

<user>Parse this query into a structured JSON format: "{query}"

Remember: Your response MUST start with <o> and end with </o></user>

<assistant>Here are some examples of correctly formatted responses:

Query: "Show me revenue and units by product for Q1 2024"
<o>
{{
    "metrics": ["revenue", "units"],
    "time_range": {{
        "start": "2024-01-01",
        "end": "2024-03-31"
    }},
    "dimensions": ["product"],
    "filters": {{}},
    "sort": {{}}
}}
</o>

Query: "What were our top 5 categories by revenue last month with sales over $1000?"
<o>
{{
    "metrics": ["revenue"],
    "time_range": {{
        "start": "2024-02-01",
        "end": "2024-02-29"
    }},
    "dimensions": ["category"],
    "filters": {{
        "revenue": ">=1000"
    }},
    "sort": {{
        "revenue": "desc"
    }},
    "limit": 5
}}
</o>

Now parse the given query and return ONLY your JSON response between <o></o> tags:</assistant>

<user>Return your response now, following the format exactly. Remember to wrap your response in <o></o> tags:</user>"""

    def get_sql_generation_prompt(
        self,
        table_name: str,
        schema: List[Dict[str, Any]],
        metrics: List[str],
        time_range: Dict[str, str],
        dimensions: List[str],
        filters: Dict[str, Any],
        sort: Dict[str, str]
    ) -> str:
        """Get prompt for generating SQL query based on parameters."""
        return f"""<system>You are a SQL query generator that creates PostgreSQL queries.

CRITICAL: Your response MUST follow this EXACT format:
<sql>
SELECT 
    column1,
    column2,
    ...
FROM table_name
WHERE conditions
GROUP BY dimensions
ORDER BY sort_columns;
</sql>

Rules:
1. The <sql> and </sql> tags are MANDATORY - responses without these tags will cause ERRORS
2. Between these tags, output ONLY the SQL query
3. No other text, explanations, or formatting is allowed
4. SQL must follow PostgreSQL syntax
5. Column names must match schema exactly
6. Dates must use ISO format YYYY-MM-DD
7. String literals must use single quotes
8. Aliases must use snake_case
9. Each major clause must start on a new line
</system>

<user>Generate a PostgreSQL query with these parameters:
Table: {table_name}
Schema: {schema}
Metrics: {metrics}
Time Range: {time_range}
Dimensions: {dimensions}
Filters: {filters}
Sort: {sort}

Remember: Your response MUST start with <sql> and end with </sql></user>

<assistant>Here are some examples of correctly formatted responses:

Simple aggregation with dimension:
<sql>
SELECT 
    product_category,
    SUM(revenue) as total_revenue,
    COUNT(order_id) as order_count
FROM sales_data
WHERE order_date BETWEEN '2024-01-01' AND '2024-03-31'
GROUP BY product_category
ORDER BY total_revenue DESC;
</sql>

Multiple dimensions with filters:
<sql>
SELECT 
    region,
    product_name,
    DATE_TRUNC('day', order_date) as sale_date,
    SUM(units_sold) as total_units,
    AVG(unit_price) as avg_price
FROM sales_data
WHERE 
    order_date BETWEEN '2024-01-01' AND '2024-01-31'
    AND region IN ('North', 'South')
    AND units_sold > 0
GROUP BY region, product_name, DATE_TRUNC('day', order_date)
ORDER BY sale_date ASC, total_units DESC;
</sql>

Now generate the SQL query for the given parameters and return ONLY your query between <sql></sql> tags:</assistant>

<user>Return your response now, following the format exactly. Remember to wrap your response in <sql></sql> tags:</user>""" 