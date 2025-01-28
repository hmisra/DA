import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Crew, Agent, Task, Process
import yaml

# Load environment variables
load_dotenv()

class AgenticAnalysisSystem:
    def __init__(self):
        """Initialize the Agentic Analysis System."""
        self.config = self._load_config()
        self.agents = self._create_agents()
        self.crew = self._setup_crew()

    def _load_config(self):
        """Load the YAML configuration file."""
        config_path = Path("crew_config/agents.yaml")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_agents(self):
        """Create Crew AI agents from configuration."""
        agents = {}
        for agent_config in self.config['agents']:
            agent = Agent(
                name=agent_config['name'],
                description=agent_config['description'],
                llm_config={
                    "model": agent_config['llm']['model'],
                    "temperature": agent_config['llm']['temperature']
                }
            )
            agents[agent_config['name']] = agent
        return agents

    def _setup_crew(self):
        """Set up the Crew with configured agents."""
        return Crew(
            agents=list(self.agents.values()),
            process=Process.sequential
        )

    def analyze_query(self, user_query: str) -> dict:
        """
        Process a user query through the full analysis pipeline.
        
        Args:
            user_query (str): Natural language query from user
            
        Returns:
            dict: Analysis results and recommendations
        """
        # Create tasks based on the flow configuration
        tasks = []
        for step in self.config['flows'][0]['steps']:
            agent = self.agents[step['agent']]
            task = Task(
                description=f"Process step: {step['task']} for query: {user_query}",
                agent=agent
            )
            tasks.append(task)

        # Execute the crew's tasks
        result = self.crew.kickoff()
        return result

def main():
    """Main entry point for the application."""
    # Initialize the system
    system = AgenticAnalysisSystem()
    
    # Example query
    query = "What were our top-performing products last quarter by revenue?"
    
    try:
        # Process the query
        results = system.analyze_query(query)
        print("Analysis Results:", results)
    except Exception as e:
        print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main() 