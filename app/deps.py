import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")

USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

if USE_OLLAMA:
    # Local, free
    # from langchain_community.chat_models import ChatOllama
    # from langchain_community.embeddings import OllamaEmbeddings
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_K_M"),
        temperature=0.1,
    )
    embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
else:
    # OpenAI (paid API)
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
