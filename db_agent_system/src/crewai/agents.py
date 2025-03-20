"""
CrewAI agents for the DB Agent System.
Defines specialized agents for SQL development, data analysis, and report generation.
"""
from typing import List, Dict, Any, Optional

from crewai import Agent, Task, LLM
from langchain_core.tools import BaseTool

from src.ollama.llm import get_ollama_llm, OllamaManager
from src.utils.logging import AgentLogger
from src.utils.config import config
from src.crewai.tools import adapt_langchain_tools


class SQLDeveloperAgent:
    """SQL Developer Agent that converts natural language to SQL."""

    def __init__(self, tools: List[BaseTool]):
        """Initialize the SQL Developer Agent.

        Args:
            tools: List of tools available to the agent
        """
        self.logger = AgentLogger(agent_name="SQLDeveloper", agent_type="CrewAI Agent")
        self.logger.info("Initializing SQL Developer Agent")

        # Select the best model for SQL tasks
        ollama_manager = OllamaManager()
        model_name = ollama_manager.select_best_model(
            "Convert natural language to SQL queries with precision"
        )
        
        # Get the LLM
        self.llm = get_ollama_llm(model_name)
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="SQL Developer",
            goal="Convert natural language questions into accurate SQL queries",
            backstory=(
                "You are an expert SQL developer with deep knowledge of database "
                "schema design and query optimization. Your job is to translate "
                "natural language questions into precise SQL queries that extract "
                "the exact information needed."
            ),
            verbose=True,
            llm=self.llm,
            tools=tools,
            allow_delegation=False,
        )
        
        self.logger.info(f"SQL Developer Agent initialized with model {model_name}")

    def create_task(self, query: str, db_schema: Dict[str, Any]) -> Task:
        """Create a task for the SQL Developer Agent.

        Args:
            query: Natural language query
            db_schema: Database schema information

        Returns:
            Task: CrewAI task
        """
        self.logger.info(f"Creating SQL development task for query: {query}")
        
        # Format the schema information
        schema_str = self._format_schema(db_schema)
        
        # Create the task with improved instructions
        task = Task(
            description=(
                f"Convert the following natural language query into a valid SQL query:\n\n"
                f"QUERY: {query}\n\n"
                f"DATABASE SCHEMA:\n{schema_str}\n\n"
                f"IMPORTANT TOOL USAGE INSTRUCTIONS:\n"
                f"- When using the 'list_tables' tool, provide an empty JSON object as input: {{}}\n"
                f"- When using the 'get_table_schema' tool, provide the table name in JSON format: {{\"table_name\": \"[table_name]\"}}\n"
                f"- When using the 'validate_sql_query' tool, provide the query in JSON format: {{\"query\": \"[sql_query]\"}}\n"
                f"- When using the 'execute_sql_query' tool, provide the query in JSON format: {{\"query\": \"[sql_query]\"}}\n"
                f"NEVER repeat the same tool call with the same parameters if it failed - try a different approach.\n\n"
                f"Return only the SQL query without any explanation. Ensure the SQL is valid, "
                f"optimized, and accurately extracts the information requested in the query."
            ),
            expected_output=(
                "A valid SQL query that accurately translates the natural language question. "
                "Include ONLY the SQL query, nothing else."
            ),
            agent=self.agent,
        )
        
        self.logger.info("SQL development task created")
        return task

    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format the schema information for the agent.

        Args:
            schema: Database schema information

        Returns:
            str: Formatted schema string
        """
        result = []
        
        for table_name, columns in schema.items():
            table_str = f"Table: {table_name}\nColumns:"
            
            for column in columns:
                col_str = f"  - {column['name']} ({column['type']})"
                if column.get('primary_key'):
                    col_str += " PRIMARY KEY"
                table_str += f"\n{col_str}"
            
            result.append(table_str)
        
        return "\n\n".join(result)


class DataAnalystAgent:
    """Data Analyst Agent that interprets user intent and analyzes query results."""

    def __init__(self, tools: List[BaseTool]):
        """Initialize the Data Analyst Agent.

        Args:
            tools: List of tools available to the agent
        """
        self.logger = AgentLogger(agent_name="DataAnalyst", agent_type="CrewAI Agent")
        self.logger.info("Initializing Data Analyst Agent")

        # Select the best model for data analysis tasks
        ollama_manager = OllamaManager()
        model_name = ollama_manager.select_best_model(
            "Analyze data and interpret user intent for database queries"
        )
        
        # Get the LLM
        self.llm = get_ollama_llm(model_name)
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="Data Analyst",
            goal="Interpret user intent and analyze database query results",
            backstory=(
                "You are a skilled data analyst with expertise in interpreting complex "
                "data patterns and understanding user needs. Your job is to understand "
                "what users are really asking for and to extract meaningful insights "
                "from query results."
            ),
            verbose=True,
            llm=self.llm,
            tools=tools,
            allow_delegation=True,
        )
        
        self.logger.info(f"Data Analyst Agent initialized with model {model_name}")

    def create_intent_task(self, query: str) -> Task:
        """Create a task for interpreting user intent.

        Args:
            query: Natural language query

        Returns:
            Task: CrewAI task
        """
        self.logger.info(f"Creating intent analysis task for query: {query}")
        
        # Create the task with improved instructions
        task = Task(
            description=(
                f"Analyze the following user query to understand their intent:\n\n"
                f"USER QUERY: {query}\n\n"
                f"Determine what database information they need and any implicit "
                f"requirements (e.g., time periods, sorting, grouping, etc.). "
                f"The goal is to ensure we return exactly what the user is looking for, "
                f"even if they didn't state it explicitly.\n\n"
                f"IMPORTANT TOOL USAGE INSTRUCTIONS:\n"
                f"- When using the 'list_tables' tool, provide an empty JSON object as input: {{}}\n"
                f"- When using the 'get_table_schema' tool, provide the table name in JSON format: {{\"table_name\": \"[table_name]\"}}\n"
                f"- When using the 'validate_sql_query' tool, provide the query in JSON format: {{\"query\": \"[sql_query]\"}}\n"
                f"- When using the 'execute_sql_query' tool, provide the query in JSON format: {{\"query\": \"[sql_query]\"}}\n"
                f"NEVER repeat the same tool call with the same parameters if it failed - try a different approach."
            ),
            expected_output=(
                "A clear interpretation of the user's intent, including explicit and "
                "implicit requirements for the database query. Do NOT include any SQL queries in this step."
            ),
            agent=self.agent,
        )
        
        self.logger.info("Intent analysis task created")
        return task

    def create_analysis_task(self, query: str, sql_query: str, results: List[Dict[str, Any]]) -> Task:
        """Create a task for analyzing query results.

        Args:
            query: Original natural language query
            sql_query: The SQL query that was executed
            results: Query results from the database

        Returns:
            Task: CrewAI task
        """
        self.logger.info(f"Creating results analysis task for {len(results)} rows")
        
        # Format the results for the agent
        results_str = self._format_results(results)
        
        # Create the task with improved instructions
        task = Task(
            description=(
                f"Analyze the following query results to extract key insights:\n\n"
                f"USER QUERY: {query}\n\n"
                f"SQL QUERY: {sql_query}\n\n"
                f"RESULTS:\n{results_str}\n\n"
                f"Provide meaningful analysis of these results that directly answers "
                f"the user's question. Include notable patterns, outliers, or important "
                f"observations. Verify that the results correctly address the user's query.\n\n"
                f"IMPORTANT TOOL USAGE INSTRUCTIONS:\n"
                f"- When using the 'list_tables' tool, provide an empty JSON object as input: {{}}\n"
                f"- When using the 'get_table_schema' tool, provide the table name in JSON format: {{\"table_name\": \"[table_name]\"}}\n"
                f"- When using the 'validate_sql_query' tool, provide the query in JSON format: {{\"query\": \"[sql_query]\"}}\n"
                f"- When using the 'execute_sql_query' tool, provide the query in JSON format: {{\"query\": \"[sql_query]\"}}\n"
                f"NEVER repeat the same tool call with the same parameters if it failed - try a different approach."
            ),
            expected_output=(
                "A comprehensive analysis of the query results that answers the user's "
                "question and highlights important insights from the data."
            ),
            agent=self.agent,
        )
        
        self.logger.info("Results analysis task created")
        return task

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format the query results for the agent.

        Args:
            results: Query results from the database

        Returns:
            str: Formatted results string
        """
        if not results:
            return "No results returned."
        
        # Get the column names from the first result
        columns = list(results[0].keys())
        
        # Format the results as a table
        lines = []
        
        # Header
        header = " | ".join(columns)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Data rows (limit to 20 for readability)
        for i, row in enumerate(results[:20]):
            row_values = [str(row.get(col, "")) for col in columns]
            lines.append(" | ".join(row_values))
        
        # Add a note if there are more rows
        if len(results) > 20:
            lines.append(f"... and {len(results) - 20} more rows")
        
        return "\n".join(lines)


class ReportGeneratorAgent:
    """Report Generator Agent that produces user-friendly summaries."""

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        """Initialize the Report Generator Agent.

        Args:
            tools: Optional list of tools available to the agent
        """
        self.logger = AgentLogger(agent_name="ReportGenerator", agent_type="CrewAI Agent")
        self.logger.info("Initializing Report Generator Agent")

        # Select the best model for report generation tasks
        ollama_manager = OllamaManager()
        model_name = ollama_manager.select_best_model(
            "Generate clear and concise reports from database query results"
        )
        
        # Get the LLM
        self.llm = get_ollama_llm(model_name)
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="Report Generator",
            goal="Create user-friendly, informative reports from query results",
            backstory=(
                "You are a communication expert specializing in data reporting. "
                "Your job is to take complex database query results and transform "
                "them into clear, concise, and insightful reports that any user "
                "can understand, regardless of their technical background."
            ),
            verbose=True,
            llm=self.llm,
            tools=tools or [],
            allow_delegation=False,
        )
        
        self.logger.info(f"Report Generator Agent initialized with model {model_name}")

    def create_task(
        self, 
        query: str, 
        analysis: str, 
        results: List[Dict[str, Any]], 
        format_type: str = "text"
    ) -> Task:
        """Create a task for generating a report.

        Args:
            query: Original natural language query
            analysis: Analysis from the Data Analyst Agent
            results: Query results from the database
            format_type: Type of report format (text, json, etc.)

        Returns:
            Task: CrewAI task
        """
        self.logger.info(f"Creating report generation task in {format_type} format")
        
        # Format the results for the agent
        results_summary = self._format_results_summary(results)
        
        # Determine output format instructions
        format_instructions = self._get_format_instructions(format_type)
        
        # Create the task
        task = Task(
            description=(
                f"Generate a user-friendly report based on the following information:\n\n"
                f"USER QUERY: {query}\n\n"
                f"DATA ANALYSIS: {analysis}\n\n"
                f"RESULTS SUMMARY: {results_summary}\n\n"
                f"Create a clear, concise report that thoroughly answers the user's question "
                f"and presents the information in an easily digestible format. Use plain "
                f"language and avoid technical jargon unless necessary."
                f"\n\n{format_instructions}"
            ),
            expected_output=(
                f"A well-formatted {format_type} report that clearly presents the query results "
                f"in a user-friendly manner."
            ),
            agent=self.agent,
        )
        
        self.logger.info("Report generation task created")
        return task

    def _format_results_summary(self, results: List[Dict[str, Any]]) -> str:
        """Format a summary of the query results for the agent.

        Args:
            results: Query results from the database

        Returns:
            str: Formatted results summary
        """
        if not results:
            return "No results returned."
        
        summary = []
        summary.append(f"Total records: {len(results)}")
        
        # If there are results, add some basic stats
        if results:
            # Get the first few results as examples
            summary.append("\nSample records:")
            for i, record in enumerate(results[:3]):
                summary.append(f"Record {i+1}: {record}")
            
            # Add note if there are more records
            if len(results) > 3:
                summary.append(f"... and {len(results) - 3} more records")
        
        return "\n".join(summary)

    def _get_format_instructions(self, format_type: str) -> str:
        """Get instructions for the requested output format.

        Args:
            format_type: Type of report format

        Returns:
            str: Format instructions
        """
        if format_type.lower() == "json":
            return (
                "Format the report as a valid JSON object with appropriate keys for "
                "summary, key_findings, and detailed_results sections."
            )
        elif format_type.lower() == "markdown":
            return (
                "Format the report in Markdown with appropriate headings, lists, and "
                "emphasis. Include a brief summary at the top, followed by key findings "
                "and detailed results sections."
            )
        else:  # Default to text
            return (
                "Format the report as plain text with clear section headings. Include "
                "a brief summary at the top, followed by key findings and detailed "
                "results sections."
            )

    def create_report_task(self, query: str, sql_query: str, results: List[Dict[str, Any]], analysis: str) -> Task:
        """Alias for create_task with better naming to match our workflow.

        Args:
            query: Original natural language query
            sql_query: The SQL query that was executed
            results: Query results from the database
            analysis: Analysis from the Data Analyst Agent

        Returns:
            Task: CrewAI task
        """
        self.logger.info(f"Creating report task for query results")
        
        # Add SQL query to the description
        description_with_sql = (
            f"Generate a user-friendly report based on the following information:\n\n"
            f"USER QUERY: {query}\n\n"
            f"SQL QUERY EXECUTED: {sql_query}\n\n"
            f"DATA ANALYSIS: {analysis}\n\n"
        )
        
        # Format the results for the agent
        results_summary = self._format_results_summary(results)
        description_with_sql += f"RESULTS SUMMARY: {results_summary}\n\n"
        
        # Create the task using the standard method but with customized description
        task = Task(
            description=(
                description_with_sql +
                f"Create a clear, concise report that thoroughly answers the user's question "
                f"and presents the information in an easily digestible format. Use plain "
                f"language and avoid technical jargon unless necessary.\n\n"
                f"Your report should include:\n"
                f"1. A concise summary of the query and what the user was looking for\n"
                f"2. Key findings from the data analysis\n"
                f"3. A clear answer to the user's question\n"
                f"4. Any relevant insights or recommendations\n\n"
                f"Focus on clarity and readability. The report should be understandable by "
                f"someone with no technical background."
            ),
            expected_output=(
                f"A well-formatted report that clearly presents the query results "
                f"in a user-friendly manner, answering the original query directly and completely."
            ),
            agent=self.agent,
        )
        
        self.logger.info("Report generation task created")
        return task


# Factory functions to create the agents

def create_sql_developer_agent(tools: List[BaseTool]) -> SQLDeveloperAgent:
    """Create a SQL Developer Agent.

    Args:
        tools: List of tools for the agent

    Returns:
        SQLDeveloperAgent: Initialized agent
    """
    # Convert LangChain tools to CrewAI tools
    crewai_tools = adapt_langchain_tools(tools)
    
    # Create the agent with our standard method
    agent = SQLDeveloperAgent(crewai_tools)
    
    # Replace the LLM with a native CrewAI LLM connected to Ollama
    from crewai import LLM
    
    # Get model name from our manager
    ollama_manager = OllamaManager()
    model_name = ollama_manager.select_best_model(
        "Convert natural language to SQL queries with precision"
    )
    
    # Create a native CrewAI LLM for Ollama
    crewai_llm = LLM(
        model=f"ollama/{model_name}",
        base_url="http://localhost:11434"
    )
    
    # Replace the agent's LLM with the CrewAI native one
    agent.agent.llm = crewai_llm
    
    return agent


def create_data_analyst_agent(tools: List[BaseTool]) -> DataAnalystAgent:
    """Create a Data Analyst Agent.

    Args:
        tools: List of tools for the agent

    Returns:
        DataAnalystAgent: Initialized agent
    """
    # Convert LangChain tools to CrewAI tools
    crewai_tools = adapt_langchain_tools(tools)
    
    # Create the agent with our standard method
    agent = DataAnalystAgent(crewai_tools)
    
    # Replace the LLM with a native CrewAI LLM connected to Ollama
    from crewai import LLM
    
    # Get model name from our manager
    ollama_manager = OllamaManager()
    model_name = ollama_manager.select_best_model(
        "Analyze data and interpret user intent for database queries"
    )
    
    # Create a native CrewAI LLM for Ollama
    crewai_llm = LLM(
        model=f"ollama/{model_name}",
        base_url="http://localhost:11434"
    )
    
    # Replace the agent's LLM with the CrewAI native one
    agent.agent.llm = crewai_llm
    
    return agent


def create_report_generator_agent() -> ReportGeneratorAgent:
    """Create a Report Generator Agent.

    Returns:
        ReportGeneratorAgent: Initialized agent
    """
    # Create the agent with our standard method
    agent = ReportGeneratorAgent()
    
    # Replace the LLM with a native CrewAI LLM connected to Ollama
    from crewai import LLM
    
    # Get model name from our manager
    ollama_manager = OllamaManager()
    model_name = ollama_manager.select_best_model(
        "Generate clear and concise reports from database query results"
    )
    
    # Create a native CrewAI LLM for Ollama
    crewai_llm = LLM(
        model=f"ollama/{model_name}",
        base_url="http://localhost:11434"
    )
    
    # Replace the agent's LLM with the CrewAI native one
    agent.agent.llm = crewai_llm
    
    return agent 