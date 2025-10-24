import argparse
from typing import List
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from .deps import qdrant, embeddings, DEFAULT_COLLECTION
from .utils import make_splitter, load_docs
from qdrant_client.http.models import VectorParams, Distance

def upsert(path: str, collection: str):
    raw_docs = load_docs(path)
    splitter = make_splitter()
    chunks: List[Document] = splitter.split_documents(raw_docs)

    # Detect embedding dimension dynamically (works for OpenAI or Ollama)
    dim = len(embeddings.embed_query("dimension probe"))

    # Create collection if missing
    existing = [c.name for c in qdrant.get_collections().collections]
    if collection not in existing:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    # Build the vector store and add documents
    vs = Qdrant(
        client=qdrant,
        collection_name=collection,
        embeddings=embeddings,
    )
    vs.add_documents(chunks)

    print(f"Ingested {len(chunks)} chunks into '{collection}'.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()
    upsert(args.path, args.collection)
