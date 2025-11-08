"""
Fourier SDK - Project Initialization System

Interactive wizard for initializing new Fourier projects.
Creates folder structure, configuration files, and boilerplate code.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


TEMPLATE_COLORS = {
    'HEADER': '\033[95m',
    'BLUE': '\033[94m',
    'CYAN': '\033[96m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'END': '\033[0m',
    'BOLD': '\033[1m',
}


def colored(text: str, color: str) -> str:
    """Add color to text if terminal supports it"""
    return f"{TEMPLATE_COLORS.get(color, '')}{text}{TEMPLATE_COLORS['END']}"


@dataclass
class ProjectConfig:
    """Configuration for new project"""
    project_name: str
    base_dir: str
    use_agents: bool = True
    use_workflows: bool = False
    use_tools: bool = True
    structure_type: str = "directory"  # "directory" or "file"
    providers: Set[str] = None
    features: Set[str] = None

    def __post_init__(self):
        if self.providers is None:
            self.providers = set()
        if self.features is None:
            self.features = set()


class ProjectInitializer:
    """Handles project initialization"""

    AGENT_TEMPLATE = '''"""
{description}

Auto-generated agent template.
"""

from fourier import Fourier
from agent import Agent, AgentConfig
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Fourier client
client = Fourier(
    api_key=os.getenv("{env_var}"),
    provider="{provider}"
)

# Configure agent
config = AgentConfig(
    name="{agent_name}",
    description="{description}",
    max_iterations=5,
    verbose=True
)

# Create agent
{agent_var} = Agent(
    client=client,
    model="{model}",
    config=config
)

# Export for auto-discovery
__agents__ = {{
    "{agent_name}": {agent_var}
}}
'''

    TOOL_TEMPLATE = '''"""
{description}

Auto-generated tool template.
"""

from fourier.config import tool
from typing import Dict, Any


@tool
def example_tool(param1: str, param2: int) -> str:
    """
    Example tool function.

    Args:
        param1: First parameter
        param2: Second parameter

    Returns:
        Processed result
    """
    return f"Processed: {{param1}} - {{param2}}"


@tool
def fetch_data(query: str) -> Dict[str, Any]:
    """
    Fetch data based on query.

    Args:
        query: Search query

    Returns:
        Data dictionary
    """
    # TODO: Implement your data fetching logic
    return {{
        "query": query,
        "results": [],
        "status": "success"
    }}
'''

    WORKFLOW_TEMPLATE = '''"""
{description}

Auto-generated workflow template.
"""

from workflow import Workflow, WorkflowNode, NodeType


def create_{workflow_name}():
    """Create the {workflow_name} workflow"""
    workflow = Workflow(name="{workflow_name}")

    # TODO: Add your workflow nodes here
    # Example:
    # input_node = WorkflowNode(
    #     node_type=NodeType.INPUT,
    #     name="input",
    #     config={{"schema": {{}}}}
    # )
    # workflow.add_node(input_node)

    return workflow


# Create workflow instance
{workflow_var} = create_{workflow_name}()

# Export for auto-discovery
__workflows__ = {{
    "{workflow_name}": {workflow_var}
}}
'''

    MAIN_TEMPLATE = '''"""
{project_name} - Main Application

Auto-generated main file with Fourier config initialization.
"""

from fourier.config import FourierConfig, get_config
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def initialize():
    """Initialize Fourier configuration"""
    print("\\n" + "=" * 60)
    print("{project_name} - Initializing")
    print("=" * 60 + "\\n")

    # Initialize config with auto-discovery
    config = FourierConfig(
        base_dir=".",
        auto_discover=True
    )

    # Display discovered resources
    resources = config.list_resources()
    print(f"Discovered Resources:")
    print(f"  Agents: {{len(resources['agents'])}} - {{resources['agents']}}")
    print(f"  Workflows: {{len(resources['workflows'])}} - {{resources['workflows']}}")
    print(f"  Tools: {{len(resources['tools'])}} - {{resources['tools']}}")
    print()

    return config


def main():
    """Main application entry point"""
    # Initialize configuration
    config = initialize()

    # Your application logic here
    print("Application started!")
    print("Use get_config() to access resources from anywhere.")
    print()

    # Example: Invoke an agent
    try:
        if config.registry.list_agents():
            agent_name = config.registry.list_agents()[0]
            print(f"Example: Invoking agent '{{agent_name}}'...")
            response = config.invoke_agent(
                agent_name,
                query="Hello! How can you help me?"
            )
            print(f"Response: {{response['response'][:100]}}...")
    except Exception as e:
        logger.error(f"Error: {{e}}")


if __name__ == "__main__":
    main()
'''

    API_TEMPLATE = '''"""
{project_name} - API Server

FastAPI server with Fourier config integration.
Auto-generated API template.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fourier.config import get_config
from dotenv import load_dotenv
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="{project_name} API",
    description="API powered by Fourier SDK",
    version="1.0.0"
)


# Request/Response models
class ChatRequest(BaseModel):
    agent: str
    message: str


class ChatResponse(BaseModel):
    success: bool
    agent: str
    response: str


class ToolRequest(BaseModel):
    tool: str
    params: dict


class ToolResponse(BaseModel):
    success: bool
    tool: str
    result: dict


# Startup event
@app.on_event("startup")
async def startup():
    """Initialize Fourier config on startup"""
    from fourier.config import FourierConfig

    config = FourierConfig(base_dir=".", auto_discover=True)
    resources = config.list_resources()

    logger.info(f"Discovered {{len(resources['agents'])}} agents")
    logger.info(f"Discovered {{len(resources['workflows'])}} workflows")
    logger.info(f"Discovered {{len(resources['tools'])}} tools")


# Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {{"status": "ok", "message": "Fourier API is running"}}


@app.get("/resources")
async def list_resources():
    """List all available resources"""
    config = get_config()
    return config.list_resources()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with an agent"""
    try:
        config = get_config()
        response = config.invoke_agent(request.agent, query=request.message)

        return ChatResponse(
            success=True,
            agent=request.agent,
            response=response['response']
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool", response_model=ToolResponse)
async def execute_tool(request: ToolRequest):
    """Execute a tool"""
    try:
        config = get_config()
        result = config.invoke_tool(request.tool, **request.params)

        return ToolResponse(
            success=True,
            tool=request.tool,
            result={{"result": result}}
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    README_TEMPLATE = '''# {project_name}

Fourier SDK project auto-generated on {date}.

## Project Structure

```
{structure_tree}
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` file and add your API keys:

```bash
{env_example}
```

### 3. Run the Application

```bash
python main.py
```

### 4. (Optional) Run API Server

```bash
# Install FastAPI dependencies first
pip install fastapi uvicorn

# Run server
python api.py
```

Server will be available at: http://localhost:8000

## Usage

### Using the Config System

```python
from fourier.config import get_config

# Get config instance
config = get_config()

# Invoke agent
response = config.invoke_agent("agent_name", query="Hello")

# Use tool
result = config.invoke_tool("tool_name", param="value")
```

### Adding New Resources

#### Add Agent

Create file in `agents/` directory:

```python
from fourier import Fourier
from agent import Agent, AgentConfig

my_agent = Agent(...)

__agents__ = {{"my_agent": my_agent}}
```

#### Add Tool

Create file in `tools/` directory:

```python
from fourier.config import tool

@tool
def my_function(arg):
    return result
```

## Available Resources

{resources_list}

## API Endpoints

- `GET /` - Health check
- `GET /resources` - List all resources
- `POST /chat` - Chat with agent
- `POST /tool` - Execute tool

## Learn More

- [Fourier SDK Documentation](https://github.com/ThinkSDK-AI/SDK-main)
- [Config System Guide](https://github.com/ThinkSDK-AI/SDK-main/blob/main/CONFIG_SYSTEM.md)
- [Installation Guide](https://github.com/ThinkSDK-AI/SDK-main/blob/main/INSTALLATION.md)

## License

MIT
'''

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.base_path = Path(config.base_dir) / config.project_name

    def create_project(self):
        """Create complete project structure"""
        print(colored("\n🚀 Creating Fourier Project", 'BOLD'))
        print(colored("=" * 60, 'BLUE'))

        # Create base directory
        self._create_directory(self.base_path)

        # Create structure based on type
        if self.config.structure_type == "directory":
            self._create_directory_structure()
        else:
            self._create_file_structure()

        # Create common files
        self._create_main_file()
        self._create_env_file()
        self._create_gitignore()
        self._create_requirements()
        self._create_readme()

        # Optional: Create API file
        if 'api' in self.config.features:
            self._create_api_file()

        # Create .fourier_config.json
        self._create_fourier_config()

        print(colored(f"\n✅ Project '{self.config.project_name}' created successfully!", 'GREEN'))
        print(colored(f"📁 Location: {self.base_path}", 'CYAN'))

        self._print_next_steps()

    def _create_directory(self, path: Path):
        """Create a directory"""
        path.mkdir(parents=True, exist_ok=True)
        print(f"  📁 Created: {path.relative_to(self.base_path.parent)}")

    def _create_file(self, path: Path, content: str):
        """Create a file with content"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"  📄 Created: {path.relative_to(self.base_path.parent)}")

    def _create_directory_structure(self):
        """Create directory-based structure"""
        if self.config.use_agents:
            agents_dir = self.base_path / "agents"
            self._create_directory(agents_dir)
            self._create_file(
                agents_dir / "__init__.py",
                '"""Agents module"""\n'
            )
            self._create_example_agent(agents_dir / "example_agent.py")

        if self.config.use_workflows:
            workflows_dir = self.base_path / "workflows"
            self._create_directory(workflows_dir)
            self._create_file(
                workflows_dir / "__init__.py",
                '"""Workflows module"""\n'
            )
            self._create_example_workflow(workflows_dir / "example_workflow.py")

        if self.config.use_tools:
            tools_dir = self.base_path / "tools"
            self._create_directory(tools_dir)
            self._create_file(
                tools_dir / "__init__.py",
                '"""Tools module"""\n'
            )
            self._create_example_tools(tools_dir / "example_tools.py")

    def _create_file_structure(self):
        """Create file-based structure"""
        if self.config.use_agents:
            self._create_example_agent(self.base_path / "agents.py")

        if self.config.use_workflows:
            self._create_example_workflow(self.base_path / "workflows.py")

        if self.config.use_tools:
            self._create_example_tools(self.base_path / "tools.py")

    def _create_example_agent(self, path: Path):
        """Create example agent file"""
        # Get first provider or default to groq
        provider = list(self.config.providers)[0] if self.config.providers else "groq"
        provider_map = {
            "groq": ("GROQ_API_KEY", "llama-3.1-8b-instant"),
            "anthropic": ("ANTHROPIC_API_KEY", "claude-3-5-sonnet-20241022"),
            "openai": ("OPENAI_API_KEY", "gpt-4"),
            "together": ("TOGETHER_API_KEY", "meta-llama/Llama-3-8b-chat-hf"),
            "bedrock": ("AWS_ACCESS_KEY_ID", "claude-3-5-sonnet"),
        }

        env_var, model = provider_map.get(provider, ("GROQ_API_KEY", "llama-3.1-8b-instant"))

        content = self.AGENT_TEMPLATE.format(
            description="Example agent for demonstration",
            env_var=env_var,
            provider=provider,
            agent_name="example_agent",
            model=model,
            agent_var="example_agent"
        )
        self._create_file(path, content)

    def _create_example_workflow(self, path: Path):
        """Create example workflow file"""
        content = self.WORKFLOW_TEMPLATE.format(
            description="Example workflow for demonstration",
            workflow_name="example_workflow",
            workflow_var="example_workflow"
        )
        self._create_file(path, content)

    def _create_example_tools(self, path: Path):
        """Create example tools file"""
        content = self.TOOL_TEMPLATE.format(
            description="Example tools for demonstration"
        )
        self._create_file(path, content)

    def _create_main_file(self):
        """Create main.py"""
        content = self.MAIN_TEMPLATE.format(
            project_name=self.config.project_name
        )
        self._create_file(self.base_path / "main.py", content)

    def _create_api_file(self):
        """Create api.py with FastAPI template"""
        content = self.API_TEMPLATE.format(
            project_name=self.config.project_name
        )
        self._create_file(self.base_path / "api.py", content)

    def _create_env_file(self):
        """Create .env file"""
        env_lines = ["# Fourier SDK Environment Configuration\n\n"]

        # Add provider-specific keys
        provider_keys = {
            "groq": "GROQ_API_KEY=your_groq_api_key_here",
            "together": "TOGETHER_API_KEY=your_together_api_key_here",
            "openai": "OPENAI_API_KEY=your_openai_api_key_here",
            "anthropic": "ANTHROPIC_API_KEY=your_anthropic_api_key_here",
            "perplexity": "PERPLEXITY_API_KEY=your_perplexity_api_key_here",
            "nebius": "NEBIUS_API_KEY=your_nebius_api_key_here",
            "bedrock": "AWS_ACCESS_KEY_ID=your_aws_access_key_here\nAWS_SECRET_ACCESS_KEY=your_aws_secret_key_here\nAWS_REGION=us-east-1",
        }

        for provider in self.config.providers:
            if provider in provider_keys:
                env_lines.append(f"# {provider.title()}\n")
                env_lines.append(provider_keys[provider] + "\n\n")

        if not self.config.providers:
            env_lines.append("# Add your API keys here\n")

        self._create_file(self.base_path / ".env", "".join(env_lines))

    def _create_gitignore(self):
        """Create .gitignore"""
        content = """# Fourier SDK
.env
.fourier_config.json
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.vscode/
.idea/
*.swp
*.swo
.DS_Store
"""
        self._create_file(self.base_path / ".gitignore", content)

    def _create_requirements(self):
        """Create requirements.txt"""
        reqs = ["# Fourier SDK Dependencies\n\n"]

        # Base requirements
        reqs.append("-e git+https://github.com/ThinkSDK-AI/SDK-main.git#egg=fourier-sdk\n")
        reqs.append("# Or if using pip: fourier-sdk\n\n")

        # Provider-specific
        if "bedrock" in self.config.providers:
            reqs.append("# AWS Bedrock\nboto3>=1.28.0\n\n")

        # Optional features
        if "search" in self.config.features:
            reqs.append("# Web Search\nbeautifulsoup4>=4.12.0\n\n")

        if "api" in self.config.features:
            reqs.append("# API Server\nfastapi>=0.104.0\nuvicorn>=0.24.0\n\n")

        self._create_file(self.base_path / "requirements.txt", "".join(reqs))

    def _create_readme(self):
        """Create README.md"""
        from datetime import datetime

        # Build structure tree
        tree_lines = [f"{self.config.project_name}/"]
        if self.config.structure_type == "directory":
            if self.config.use_agents:
                tree_lines.append("├── agents/")
                tree_lines.append("│   └── example_agent.py")
            if self.config.use_workflows:
                tree_lines.append("├── workflows/")
                tree_lines.append("│   └── example_workflow.py")
            if self.config.use_tools:
                tree_lines.append("├── tools/")
                tree_lines.append("│   └── example_tools.py")
        else:
            if self.config.use_agents:
                tree_lines.append("├── agents.py")
            if self.config.use_workflows:
                tree_lines.append("├── workflows.py")
            if self.config.use_tools:
                tree_lines.append("├── tools.py")

        tree_lines.extend([
            "├── main.py",
            "├── .env",
            "├── .gitignore",
            "├── requirements.txt",
            "└── README.md"
        ])

        # Build env example
        env_example = "\n".join([
            f"{provider.upper()}_API_KEY=your_key_here"
            for provider in self.config.providers
        ]) or "# Add your API keys"

        # Build resources list
        resources = []
        if self.config.use_agents:
            resources.append("- **Agents**: example_agent")
        if self.config.use_workflows:
            resources.append("- **Workflows**: example_workflow")
        if self.config.use_tools:
            resources.append("- **Tools**: example_tool, fetch_data")

        content = self.README_TEMPLATE.format(
            project_name=self.config.project_name,
            date=datetime.now().strftime("%Y-%m-%d"),
            structure_tree="\n".join(tree_lines),
            env_example=env_example,
            resources_list="\n".join(resources) if resources else "None yet - add your own!"
        )
        self._create_file(self.base_path / "README.md", content)

    def _create_fourier_config(self):
        """Create .fourier_config.json"""
        config_data = {
            "project_name": self.config.project_name,
            "structure_type": self.config.structure_type,
            "providers": list(self.config.providers),
            "features": list(self.config.features),
            "paths": {}
        }

        if self.config.structure_type == "directory":
            if self.config.use_agents:
                config_data["paths"]["agents_dir"] = "./agents"
            if self.config.use_workflows:
                config_data["paths"]["workflows_dir"] = "./workflows"
            if self.config.use_tools:
                config_data["paths"]["tools_dir"] = "./tools"
        else:
            if self.config.use_agents:
                config_data["paths"]["agents_file"] = "./agents.py"
            if self.config.use_workflows:
                config_data["paths"]["workflows_file"] = "./workflows.py"
            if self.config.use_tools:
                config_data["paths"]["tools_file"] = "./tools.py"

        self._create_file(
            self.base_path / ".fourier_config.json",
            json.dumps(config_data, indent=2)
        )

    def _print_next_steps(self):
        """Print next steps for user"""
        print(colored("\n📋 Next Steps:", 'BOLD'))
        print(colored("=" * 60, 'BLUE'))
        print()
        print(f"1. Navigate to your project:")
        print(colored(f"   cd {self.config.project_name}", 'CYAN'))
        print()
        print("2. Install dependencies:")
        print(colored("   pip install -r requirements.txt", 'CYAN'))
        print()
        print("3. Configure your API keys:")
        print(colored("   edit .env", 'CYAN'))
        print()
        print("4. Run your application:")
        print(colored("   python main.py", 'CYAN'))
        print()
        if 'api' in self.config.features:
            print("5. (Optional) Start API server:")
            print(colored("   python api.py", 'CYAN'))
            print()
        print(colored("📚 Documentation:", 'BOLD'))
        print("   - README.md in your project")
        print("   - Config System: CONFIG_SYSTEM.md")
        print("   - Installation: INSTALLATION.md")
        print()


def run_interactive_init(base_dir: str = ".") -> Optional[ProjectConfig]:
    """Run interactive initialization wizard"""
    print(colored("\n" + "=" * 60, 'BLUE'))
    print(colored("  Fourier SDK - Project Initialization Wizard", 'BOLD'))
    print(colored("=" * 60 + "\n", 'BLUE'))

    # Project name
    default_name = "my_fourier_project"
    project_name = input(f"Project name [{default_name}]: ").strip() or default_name

    # Structure type
    print(colored("\n📁 Choose project structure:", 'BOLD'))
    print("  1. Directory-based (agents/, tools/, workflows/)")
    print("  2. File-based (agents.py, tools.py, workflows.py)")
    structure_choice = input("Choice [1]: ").strip() or "1"
    structure_type = "directory" if structure_choice == "1" else "file"

    # What to include
    print(colored("\n🔧 What do you want to include?", 'BOLD'))
    use_agents = input("Agents? [Y/n]: ").strip().lower() != 'n'
    use_workflows = input("Workflows? [y/N]: ").strip().lower() == 'y'
    use_tools = input("Tools? [Y/n]: ").strip().lower() != 'n'

    # Providers
    print(colored("\n🤖 Select LLM providers (comma-separated):", 'BOLD'))
    print("  1. Groq")
    print("  2. Anthropic")
    print("  3. OpenAI")
    print("  4. Together AI")
    print("  5. Bedrock")
    print("  6. All")
    provider_choice = input("Providers [1]: ").strip() or "1"

    provider_map = {
        "1": {"groq"},
        "2": {"anthropic"},
        "3": {"openai"},
        "4": {"together"},
        "5": {"bedrock"},
        "6": {"groq", "anthropic", "openai", "together", "bedrock"}
    }

    if ',' in provider_choice:
        providers = set()
        for choice in provider_choice.split(','):
            providers.update(provider_map.get(choice.strip(), set()))
    else:
        providers = provider_map.get(provider_choice, {"groq"})

    # Features
    print(colored("\n✨ Additional features:", 'BOLD'))
    use_api = input("Generate API server template? [y/N]: ").strip().lower() == 'y'
    use_search = input("Include web search? [y/N]: ").strip().lower() == 'y'

    features = set()
    if use_api:
        features.add("api")
    if use_search:
        features.add("search")

    # Confirm
    print(colored("\n📋 Summary:", 'BOLD'))
    print(f"  Project: {project_name}")
    print(f"  Structure: {structure_type}")
    print(f"  Agents: {'Yes' if use_agents else 'No'}")
    print(f"  Workflows: {'Yes' if use_workflows else 'No'}")
    print(f"  Tools: {'Yes' if use_tools else 'No'}")
    print(f"  Providers: {', '.join(providers)}")
    print(f"  Features: {', '.join(features) if features else 'None'}")
    print()

    confirm = input("Create project? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print(colored("❌ Cancelled", 'YELLOW'))
        return None

    return ProjectConfig(
        project_name=project_name,
        base_dir=base_dir,
        use_agents=use_agents,
        use_workflows=use_workflows,
        use_tools=use_tools,
        structure_type=structure_type,
        providers=providers,
        features=features
    )


def initialize_project(base_dir: str = ".", interactive: bool = True) -> bool:
    """
    Initialize a new Fourier project.

    Args:
        base_dir: Base directory for project creation
        interactive: If True, run interactive wizard

    Returns:
        True if successful, False otherwise
    """
    try:
        if interactive:
            config = run_interactive_init(base_dir)
            if config is None:
                return False
        else:
            # Non-interactive with defaults
            config = ProjectConfig(
                project_name="my_fourier_project",
                base_dir=base_dir
            )

        # Create project
        initializer = ProjectInitializer(config)
        initializer.create_project()

        return True

    except KeyboardInterrupt:
        print(colored("\n\n❌ Cancelled by user", 'YELLOW'))
        return False
    except Exception as e:
        print(colored(f"\n\n❌ Error: {e}", 'RED'))
        import traceback
        traceback.print_exc()
        return False
