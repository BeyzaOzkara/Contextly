from typing import List
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
from .deps import qdrant, embeddings, llm, DEFAULT_COLLECTION
from typing import TypedDict

class RagState(TypedDict):
    question: str
    k: int
    collection: str
    context: List[Document]
    answer: str

def retrieve_node(state: RagState) -> RagState:
    vs = Qdrant(client=qdrant, collection_name=state["collection"], embeddings=embeddings)
    # include scores so we can return them
    docs_scores = vs.similarity_search_with_score(state["question"], k=state["k"])
    # keep only docs in context, stash scores in metadata for returning
    docs = []
    for doc, score in docs_scores:
        md = dict(doc.metadata or {})
        md["_score"] = float(score)
        doc.metadata = md
        docs.append(doc)
    state["context"] = docs
    return state

def generate_node(state: RagState) -> RagState:
    context_text = "\n\n".join([d.page_content for d in state["context"]])
    system = (
        "You are a helpful assistant that answers ONLY using the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    messages = [
        ("system", system),
        ("human", f"Context:\n{context_text}\n\nQuestion: {state['question']}")
    ]
    resp = llm.invoke(messages)
    state["answer"] = resp.content
    return state

def build_graph():
    g = StateGraph(RagState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    return g.compile()
