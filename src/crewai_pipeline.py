from crewai import Agent, Task, Crew
from langchain.tools import Tool

def make_reranked_retrieval_tool(store, actions):
    def reranked_retrieval(query: str) -> str:
        raw_results = store.similarity_search(query, k=4)
        reranked = actions.get_document_rerank(3, query, raw_results)
        return "\n\n".join([doc.page_content for doc in reranked])

    return Tool(
        name="RerankedRetriever",
        func=reranked_retrieval,
        description="Retrieves and reranks documents from the vector store using a similarity search followed by a custom reranker."
    )

def build_crew(llm, retrieval_tool, user_query):
    # Agents
    retriever_agent = Agent(
        role="Retriever",
        goal="Retrieve the most relevant document context",
        backstory="Expert in semantic search and information retrieval.",
        tools=[retrieval_tool],
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    summarizer_agent = Agent(
        role="Summarizer",
        goal="Summarize the retrieved content for analysis",
        backstory="Skilled at condensing technical content into understandable summaries.",
        verbose=True,
        llm=llm
    )

    answer_agent = Agent(
        role="Answer Generator",
        goal="Answer the user's question accurately",
        backstory="Experienced in interpreting document content and generating insights.",
        verbose=True,
        llm=llm
    )

    # Tasks
    task1 = Task(
        description=f"Retrieve the most relevant information for the query: '{user_query}'",
        expected_output="Relevant document excerpts",
        agent=retriever_agent
    )

    task2 = Task(
        description="Summarize the content retrieved by the first task.",
        expected_output="A summary of the key points from the document.",
        agent=summarizer_agent,
        context=[task1]
    )

    task3 = Task(
        description=f"Using the summary, answer the user’s original query: '{user_query}'",
        expected_output="A clear and well-grounded answer to the question.",
        agent=answer_agent,
        context=[task2]
    )

    crew = Crew(
        agents=[retriever_agent, summarizer_agent, answer_agent],
        tasks=[task1, task2, task3],
        verbose=True
    )

    # Return crew and task objects so you can run each step manually
    return crew, task1, task2, task3