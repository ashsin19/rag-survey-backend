from langchain.tools import Tool
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Tool 1: Retrieval
def make_retrieval_tool(store, actions):
    def retrieve(query: str) -> str:
        results = store.similarity_search(query, k=4)
        reranked = actions.get_document_rerank(3, query, results)
        print(f"Reranked results: {reranked}")
        return "\n\n".join([doc for doc in reranked])
    
    return Tool(
        name="RerankedRetrieval",
        func=retrieve,
        description="Retrieves the most relevant content using similarity search and reranking."
    )

# Tool 2: Summarizer
def make_summarizer_tool(llm):
    def summarize(text: str) -> str:
        docs = [Document(page_content=text)]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        return chain.run(docs)
    
    return Tool(
        name="Summarizer",
        func=summarize,
        description="Summarizes the input document into key points."
    )

# Tool 3: Answer Generator
def make_answer_tool(llm):
    def answer(input_str: str) -> str:
        # input_str should include summary + original query separated by a marker
        parts = input_str.split("[QUESTION]")
        summary = parts[0].strip()
        question = parts[1].strip() if len(parts) > 1 else ""

        prompt = PromptTemplate.from_template(
            "Given this summary:\n{summary}\n\nAnswer the question:\n{question}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run({"summary": summary, "question": question})
    
    return Tool(
        name="AnswerGenerator",
        func=answer,
        description="Generates a final answer from summary + query."
    )
