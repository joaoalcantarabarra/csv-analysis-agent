import re
from typing import List
import os

from langchain.tools import Tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompts.prompt_csv_agent import generate_csv_agent_prompt
from src.tools.available_csvs import ListAvailableCSVsTool
from src.tools.visualization_tools import VisualizationToolkit


def create_custom_tools(
    output_dir: str = 'graphics'
) -> List[Tool]:
    """Create custom tools for the CSV agent"""

    python_tool = PythonAstREPLTool()

    viz_tools = VisualizationToolkit(output_dir=output_dir)
    csv_list_tool = ListAvailableCSVsTool()

    tools = [python_tool, csv_list_tool, *viz_tools.get_tools()]

    return tools


def create_csv_agent(
    session_id: str = 'agente_csv',
    output_dir: str = 'graphics',
) -> object:
    """
    Create a CSV agent with LangGraph.

    Args:
        session_id (str): The session ID for the agent.
        output_dir (str): The directory to save graphics.
        data_dir (str): The directory containing the CSV data.

    Returns:
        object: The configured agent instance.
    """
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY não configurada. "
            "Configure a API Key através da interface ou arquivo .env"
        )

    tools = create_custom_tools(output_dir=output_dir)

    checkpointer = InMemorySaver()

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2,
        google_api_key=api_key
    )

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=generate_csv_agent_prompt(),
        checkpointer=checkpointer,
        name=session_id,
        debug=True
    )

    return agent

def analyze_csv(
    agent, query: str, thread_id: str = "default_thread"
) -> str:
    """
    Perform CSV data analysis based on the user’s query.

    Args:
        agent (Agent): Configured agent instance.
        query (str): User's question or query about the data.

    Returns:
        str: The analysis response from the agent.
    """
    inputs = {
        "messages": [{"role": "user", "content": query}]
    }
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    result = agent.invoke(inputs, config=config)

    return result["messages"][-1].content


def output_parser(response: str) -> str:
    """
    Parse the agent's response to extract the generated output.

    Args:
        response (str): The response from the agent.

    Returns:
        str: The generated output.
    """
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
