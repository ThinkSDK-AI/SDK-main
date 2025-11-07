"""
Workflow Examples

Demonstrates how to create and execute node-based workflows with agents,
assistants, and other processing nodes.
"""

import os
from fourier import Fourier
from agent import Agent, AgentConfig
from assistant import Assistant, AssistantConfig
from workflow import Workflow
from dotenv import load_dotenv

load_dotenv()


def simple_workflow_example():
    """Simple linear workflow."""
    print("=== Simple Linear Workflow Example ===\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        return

    client = Fourier(api_key=api_key, provider="groq")

    # Create assistant
    assistant = Assistant(
        client=client,
        name="SimpleAssistant",
        model="mixtral-8x7b-32768"
    )

    # Create workflow
    workflow = Workflow(name="SimpleWorkflow")

    # Add nodes
    input_node = workflow.add_input_node("Start")
    assistant_node = workflow.add_assistant_node(assistant, "Assistant")
    output_node = workflow.add_output_node("Result")

    # Connect nodes
    workflow.connect(input_node.node_id, assistant_node.node_id)
    workflow.connect(assistant_node.node_id, output_node.node_id)

    # Visualize
    print(workflow.visualize())

    # Execute
    print("\nExecuting workflow...")
    result = workflow.execute("What is the capital of France?", verbose=True)

    print(f"\n=== Result ===")
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    print(f"Execution time: {result['execution_time']:.2f}s")
    print(f"Nodes executed: {result['nodes_executed']}")


def agent_workflow_example():
    """Workflow with agent node."""
    print("\n\n=== Agent Workflow Example ===\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        return

    client = Fourier(api_key=api_key, provider="groq")

    # Create agent with tool
    agent = Agent(
        client=client,
        name="MathAgent",
        model="mixtral-8x7b-32768",
        config=AgentConfig(verbose=False)
    )

    # Register calculator tool
    def calculator(operation: str, a: float, b: float) -> float:
        ops = {"add": a + b, "multiply": a * b, "subtract": a - b, "divide": a / b if b != 0 else 0}
        return ops.get(operation, 0)

    agent.register_tool(
        name="calculator",
        description="Perform arithmetic operations",
        parameters={
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "a": {"type": "number"},
                "b": {"type": "number"}
            }
        },
        function=calculator,
        required=["operation", "a", "b"]
    )

    # Create workflow
    workflow = Workflow(name="AgentWorkflow")

    # Add nodes
    input_node = workflow.add_input_node()
    agent_node = workflow.add_agent_node(agent, "MathAgent")
    output_node = workflow.add_output_node()

    # Connect
    workflow.connect(input_node.node_id, agent_node.node_id)
    workflow.connect(agent_node.node_id, output_node.node_id)

    # Execute
    print(workflow.visualize())
    print("\nExecuting workflow...")

    result = workflow.execute("Calculate 25 multiplied by 4", verbose=True)

    print(f"\n=== Result ===")
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    print(f"Execution time: {result['execution_time']:.2f}s")


def transform_workflow_example():
    """Workflow with transform nodes."""
    print("\n\n=== Transform Workflow Example ===\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        return

    client = Fourier(api_key=api_key, provider="groq")
    assistant = Assistant(client=client, model="mixtral-8x7b-32768")

    # Create workflow
    workflow = Workflow(name="TransformWorkflow")

    # Add nodes
    input_node = workflow.add_input_node("UserInput")

    # Transform: Convert to uppercase
    uppercase_node = workflow.add_transform_node(
        lambda x: x.upper(),
        "ToUppercase"
    )

    # Transform: Add prefix
    prefix_node = workflow.add_transform_node(
        lambda x: f"Question: {x}",
        "AddPrefix"
    )

    assistant_node = workflow.add_assistant_node(assistant, "Assistant")

    # Transform: Extract first sentence
    extract_node = workflow.add_transform_node(
        lambda x: x.split('.')[0] + '.' if '.' in x else x,
        "ExtractFirstSentence"
    )

    output_node = workflow.add_output_node("FinalOutput")

    # Connect nodes
    workflow.connect(input_node.node_id, uppercase_node.node_id)
    workflow.connect(uppercase_node.node_id, prefix_node.node_id)
    workflow.connect(prefix_node.node_id, assistant_node.node_id)
    workflow.connect(assistant_node.node_id, extract_node.node_id)
    workflow.connect(extract_node.node_id, output_node.node_id)

    # Execute
    print(workflow.visualize())
    print("\nExecuting workflow...")

    result = workflow.execute("what is the capital of france?", verbose=True)

    print(f"\n=== Result ===")
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    print(f"Execution time: {result['execution_time']:.2f}s")


def condition_workflow_example():
    """Workflow with conditional branching."""
    print("\n\n=== Conditional Workflow Example ===\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        return

    client = Fourier(api_key=api_key, provider="groq")

    # Create assistants for different paths
    short_assistant = Assistant(
        client=client,
        model="mixtral-8x7b-32768",
        config=AssistantConfig(
            system_prompt="Give very brief, one-sentence answers."
        )
    )

    detailed_assistant = Assistant(
        client=client,
        model="mixtral-8x7b-32768",
        config=AssistantConfig(
            system_prompt="Give detailed, comprehensive answers with examples."
        )
    )

    # Create workflow
    workflow = Workflow(name="ConditionalWorkflow")

    # Add nodes
    input_node = workflow.add_input_node()

    # Condition: Check if query is short
    condition_node = workflow.add_condition_node(
        lambda x: len(str(x)) < 20,
        "IsShortQuery"
    )

    short_assistant_node = workflow.add_assistant_node(short_assistant, "BriefResponse")
    detailed_assistant_node = workflow.add_assistant_node(detailed_assistant, "DetailedResponse")

    output_node = workflow.add_output_node()

    # Connect
    workflow.connect(input_node.node_id, condition_node.node_id)

    # Set condition branches
    condition_node.true_branch = short_assistant_node.node_id
    condition_node.false_branch = detailed_assistant_node.node_id

    workflow.connect(short_assistant_node.node_id, output_node.node_id)
    workflow.connect(detailed_assistant_node.node_id, output_node.node_id)

    # Execute with short query
    print(workflow.visualize())
    print("\n=== Test 1: Short Query ===")
    result1 = workflow.execute("What is AI?", verbose=True)
    print(f"Output: {result1['output']}\n")

    # Execute with long query
    print("=== Test 2: Long Query ===")
    result2 = workflow.execute("Can you explain what artificial intelligence is and how it works?", verbose=True)
    print(f"Output: {result2['output']}")


def multi_agent_workflow_example():
    """Complex workflow with multiple agents."""
    print("\n\n=== Multi-Agent Workflow Example ===\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        return

    client = Fourier(api_key=api_key, provider="groq")

    # Create specialized agents
    researcher = Agent(
        client=client,
        name="Researcher",
        model="mixtral-8x7b-32768",
        config=AgentConfig(
            system_prompt="You are a researcher. Provide factual, well-researched information.",
            verbose=False
        )
    )

    analyzer = Agent(
        client=client,
        name="Analyzer",
        model="mixtral-8x7b-32768",
        config=AgentConfig(
            system_prompt="You are an analyzer. Break down information and provide structured analysis.",
            verbose=False
        )
    )

    summarizer = Assistant(
        client=client,
        name="Summarizer",
        model="mixtral-8x7b-32768",
        config=AssistantConfig(
            system_prompt="You are a summarizer. Create brief, concise summaries."
        )
    )

    # Create workflow
    workflow = Workflow(name="ResearchPipeline")

    # Add nodes
    input_node = workflow.add_input_node("Topic")
    researcher_node = workflow.add_agent_node(researcher, "Research")
    analyzer_node = workflow.add_agent_node(analyzer, "Analyze")
    summarizer_node = workflow.add_assistant_node(summarizer, "Summarize")
    output_node = workflow.add_output_node("Report")

    # Connect
    workflow.connect(input_node.node_id, researcher_node.node_id)
    workflow.connect(researcher_node.node_id, analyzer_node.node_id)
    workflow.connect(analyzer_node.node_id, summarizer_node.node_id)
    workflow.connect(summarizer_node.node_id, output_node.node_id)

    # Execute
    print(workflow.visualize())
    print("\nExecuting multi-agent workflow...")

    result = workflow.execute("Quantum computing", verbose=True)

    print(f"\n=== Final Report ===")
    print(f"Success: {result['success']}")
    print(f"Output:\n{result['output']}")
    print(f"\nExecution time: {result['execution_time']:.2f}s")
    print(f"Nodes executed: {result['nodes_executed']}")


def workflow_persistence_example():
    """Save and load workflows."""
    print("\n\n=== Workflow Persistence Example ===\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        return

    client = Fourier(api_key=api_key, provider="groq")
    assistant = Assistant(client=client, model="mixtral-8x7b-32768")

    # Create workflow
    workflow = Workflow(name="SavedWorkflow")

    input_node = workflow.add_input_node()
    assistant_node = workflow.add_assistant_node(assistant, "Assistant")
    output_node = workflow.add_output_node()

    workflow.connect(input_node.node_id, assistant_node.node_id)
    workflow.connect(assistant_node.node_id, output_node.node_id)

    # Execute
    result1 = workflow.execute("Hello!", verbose=False)
    print(f"First execution: {result1['output']}")

    # Save workflow
    workflow.save("workflow.json")
    print("\nWorkflow saved to workflow.json")

    # Load and show structure
    import json
    with open("workflow.json", 'r') as f:
        loaded_data = json.load(f)

    print(f"\nLoaded workflow structure:")
    print(f"  Name: {loaded_data['name']}")
    print(f"  Nodes: {len(loaded_data['nodes'])}")
    print(f"  Executions: {loaded_data['execution_count']}")


def main():
    """Run all workflow examples."""
    print("="*60)
    print("FourierSDK Workflow Examples")
    print("="*60)

    try:
        simple_workflow_example()
        agent_workflow_example()
        transform_workflow_example()
        condition_workflow_example()
        multi_agent_workflow_example()
        workflow_persistence_example()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Examples Complete")
    print("="*60)


if __name__ == "__main__":
    main()
