#!/usr/bin/env python
"""
Install and register the Ollama plugin for AgentIQ.
This script sets up the Ollama plugin so it can be used with AgentIQ workflows.
"""
import os
import sys
import shutil
import site
import importlib
from pathlib import Path

def find_aiq_installation():
    """Find the AgentIQ installation directory."""
    # Directly check in known locations first
    for path in sys.path:
        aiq_path = os.path.join(path, 'aiq')
        if os.path.isdir(aiq_path):
            print(f"Found AgentIQ at: {aiq_path}")
            return Path(aiq_path)
    
    # Fallback to importlib approach
    try:
        import aiq
        aiq_dir = Path(aiq.__file__).parent
        print(f"Found AgentIQ via import at: {aiq_dir}")
        return aiq_dir
    except ImportError:
        print("Error: AgentIQ is not installed.")
        return None

def install_ollama_plugin():
    """Install the Ollama plugin to the AgentIQ installation."""
    print("Installing Ollama plugin for AgentIQ...")
    
    # Find the AgentIQ installation
    aiq_dir = find_aiq_installation()
    if not aiq_dir:
        sys.exit(1)
    
    # Set source and target directories
    source_dir = Path("src/agentiq/plugins/ollama")
    target_dir = aiq_dir / "plugins" / "ollama"
    
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} not found.")
        sys.exit(1)
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy files from source to target
    for file_path in source_dir.glob('*.py'):
        if file_path.name != '__pycache__':
            shutil.copy2(file_path, target_dir)
            print(f"Copied {file_path.name} to {target_dir}")
    
    # Create or update __init__.py
    init_file = target_dir / "__init__.py"
    with open(init_file, 'w') as f:
        f.write("""# Ollama LLM Provider for AgentIQ
from aiq.plugins.ollama.ollama_llm import OllamaModelConfig, ollama_llm

__all__ = ["OllamaModelConfig", "ollama_llm"]
""")
    print(f"Created/updated {init_file}")
    
    # Update the register.py file to import our plugin
    register_file = aiq_dir / "llm" / "register.py"
    if register_file.exists():
        with open(register_file, 'r') as f:
            content = f.read()
        
        # Check if our import is already there
        if "from aiq.plugins.ollama import ollama_llm" not in content:
            # Find the last import line
            import_lines = [line for line in content.split('\n') if line.startswith('from') or line.startswith('import')]
            if import_lines:
                last_import = import_lines[-1]
                new_content = content.replace(
                    last_import,
                    f"{last_import}\n# Ollama LLM Provider\nfrom aiq.plugins.ollama import ollama_llm"
                )
                
                with open(register_file, 'w') as f:
                    f.write(new_content)
                print(f"Updated {register_file} to import the Ollama plugin")
            else:
                print(f"Warning: Could not update {register_file} - no import lines found.")
        else:
            print(f"The Ollama plugin is already registered in {register_file}")
    else:
        print(f"Warning: {register_file} not found. Unable to register the Ollama plugin.")
    
    # Update the adapter.py file to use direct imports
    adapter_file = target_dir / "adapter.py"
    if adapter_file.exists():
        with open(adapter_file, 'r') as f:
            content = f.read()
        
        # Replace any references to db_agent_system.src with direct imports
        content = content.replace("from db_agent_system.src", "from")
        
        with open(adapter_file, 'w') as f:
            f.write(content)
        print(f"Updated imports in {adapter_file}")
        
    # Update the ollama_llm.py file to use direct imports
    ollama_file = target_dir / "ollama_llm.py"
    if ollama_file.exists():
        with open(ollama_file, 'r') as f:
            content = f.read()
        
        # Replace any references to db_agent_system.src with direct imports
        content = content.replace("from db_agent_system.src.agentiq.plugins.ollama.adapter", "from aiq.plugins.ollama.adapter")
        
        with open(ollama_file, 'w') as f:
            f.write(content)
        print(f"Updated imports in {ollama_file}")
    
    print("\nOllama plugin installation complete!")
    print("\nYou can now use the Ollama LLM in your AgentIQ workflows:")
    print("""
Example YAML configuration:
```yaml
llms:
  ollama_llm:
    _type: ollama
    model_name: "llama3"
    base_url: "http://localhost:11434"
    temperature: 0.2
    
workflow:
  _type: aiq.agent.tool_calling_agent/tool_calling_agent
  llm: ollama_llm
  tools:
    - aiq.tool/current_datetime
  system_prompt: "You are a helpful assistant."
```
""")

def main():
    """Main entry point."""
    print("Ollama Plugin Installer for AgentIQ")
    print("===================================")
    
    # Install the plugin
    install_ollama_plugin()

if __name__ == "__main__":
    main() 