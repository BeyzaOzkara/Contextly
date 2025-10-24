from fastapi import FastAPI, HTTPException
from .schemas import AskRequest, AskResponse, Source, IngestRequest
from .deps import DEFAULT_COLLECTION
from .graph import build_graph
from langchain_community.vectorstores import Qdrant
from .deps import qdrant, embeddings

app = FastAPI(title="LangGraph RAG API")
workflow = build_graph()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest(req: IngestRequest):
    from .ingest import upsert
    upsert(req.path, req.collection or DEFAULT_COLLECTION)
    return {"ok": True}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    collection = req.collection or DEFAULT_COLLECTION

    # Ensure collection exists / non-empty
    try:
        vs = Qdrant(client=qdrant, collection_name=collection, embeddings=embeddings)
        _ = vs.similarity_search("ping", k=1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Collection '{collection}' not ready: {e}")

    state = {
        "question": req.question,
        "k": req.k,
        "collection": collection,
        "context": [],
        "answer": "",
    }
    result = workflow.invoke(state)

    docs = result.get("context", [])
    sources = [
        Source(
            text=d.page_content[:1000],
            metadata=d.metadata,
            score=(d.metadata or {}).get("_score"),
        )
        for d in docs
    ]
    return AskResponse(answer=result["answer"], sources=sources)
