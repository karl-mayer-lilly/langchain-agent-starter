from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import os
from dotenv import load_dotenv

# Example 1: Using OpenAI's GPT-4o model with a custom endpoint
load_dotenv()  # Load environment variables from .env file

model = init_chat_model(
    model="gpt-4o",  # Specify which OpenAI model to use
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),  # Load API key from .env file
    base_url=os.getenv("CUSTOM_ENDPOINT_URL"),  # Load custom endpoint URL from .env file
    temperature=0
)

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model=model,  
    tools=[get_weather],  
    prompt="You are a helpful assistant"  
)

# # Run the agent
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
# )

config = {"configurable": {"thread_id": "abc123"}}
for step in agent.stream(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# Example 2: Using a custom MCP server for math operations and weather queries
async def run_mcp_agent():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Replace with absolute path to your math_server.py file
                "args": ["math_mcp_server.py"],
                "transport": "stdio",
            },
            "weather": {
                "command": "python",
                # Replace with absolute path to your math_server.py file
                "args": ["weather_mcp_server.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(
        model=model, 
        tools=tools
    )

    async for step in agent.astream(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
    
    async for step in agent.astream(
        {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
        
asyncio.run(run_mcp_agent())