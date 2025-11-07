#!/bin/bash
# FourierSDK CLI Examples
# Demonstrates various CLI commands and workflows

set -e  # Exit on error

echo "========================================="
echo "FourierSDK CLI Examples"
echo "========================================="
echo ""

# Check if API key is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "Error: GROQ_API_KEY not set"
    echo "Please run: export GROQ_API_KEY='your-api-key'"
    exit 1
fi

echo "1. Quick Chat Example"
echo "---------------------"
python cli.py chat "What is the capital of France?" --verbose
echo ""

echo "2. Chat with Thinking Mode"
echo "----------------------------"
python cli.py chat "What are the latest developments in AI?" \
    --thinking-mode \
    --verbose
echo ""

echo "3. Create an Agent"
echo "-------------------"
python cli.py create-agent \
    --name ExampleAgent \
    --provider groq \
    --model mixtral-8x7b-32768 \
    --system-prompt "You are a helpful assistant" \
    --save
echo ""

echo "4. Create Research Agent with Thinking Mode"
echo "---------------------------------------------"
python cli.py create-agent \
    --name ResearchBot \
    --provider groq \
    --thinking-mode \
    --thinking-depth 2 \
    --system-prompt "You are a research assistant" \
    --save
echo ""

echo "5. List All Agents"
echo "-------------------"
python cli.py list-agents --details
echo ""

echo "6. Add MCP Tools (if directory exists)"
echo "---------------------------------------"
if [ -d "./mcp_tools" ]; then
    python cli.py add-mcp --directory ./mcp_tools --agent ExampleAgent
    echo "MCP tools added to ExampleAgent"
else
    echo "Skipping: ./mcp_tools directory not found"
fi
echo ""

echo "7. Run Saved Agent"
echo "-------------------"
python cli.py run \
    --agent ExampleAgent \
    --query "Hello, how can you help me?" \
    --verbose
echo ""

echo "8. List MCP Tools"
echo "------------------"
python cli.py list-mcp
echo ""

echo "9. Show Configuration"
echo "----------------------"
python cli.py config --show
echo ""

echo "10. Run Multiple Queries with Different Agents"
echo "------------------------------------------------"
python cli.py run --agent ExampleAgent --query "Tell me a joke"
echo ""
python cli.py run --agent ResearchBot --query "Explain quantum entanglement" --verbose
echo ""

echo "11. Interactive Mode Demo"
echo "--------------------------"
echo "For interactive mode, run:"
echo "  python cli.py interactive"
echo ""
echo "Then try these commands:"
echo "  fourier> agent ExampleAgent"
echo "  fourier> chat Hello!"
echo "  fourier> list-agents"
echo "  fourier> exit"
echo ""

echo "12. Cleanup (Optional)"
echo "----------------------"
echo "To delete test agents:"
echo "  python cli.py delete-agent ExampleAgent --force"
echo "  python cli.py delete-agent ResearchBot --force"
echo ""

echo "========================================="
echo "Examples Complete!"
echo "========================================="
echo ""
echo "For more information, see CLI.md"
