import os
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def search(query: str) -> str:
    """Search the web for information

    Args:
        query: The query to search for

    Returns:
        The search result
    """

    print(f"Searching for {query}")
    return tavily_client.search(query)


# Use a model that supports tool calling (e.g. llama3.2, llama3.1, mistral).
# gemma3:270m does not support tools.
llm = ChatOllama(model="llama3.2")
tools = [search]
agent = create_agent(model=llm, tools=tools)

messages = [HumanMessage(content="What is the weather in Tokyo?")]
result = agent.invoke({"messages": messages})
print(result)


def main():
    print("Hello from langchain-agent!")


if __name__ == "__main__":
    main()
