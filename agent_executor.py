from langchain.agents import initialize_agent, AgentType
from tools import make_retrieval_tool, make_summary_tool

def build_agent(llm, store, actions):
    retrieval_tool = make_retrieval_tool(store, actions)
    summarize_tool = make_summary_tool(llm)

    tools = [retrieval_tool, summarize_tool]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent
