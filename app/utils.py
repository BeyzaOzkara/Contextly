from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .deps import CHUNK_SIZE, CHUNK_OVERLAP


def make_splitter():
    return RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", ".", " "]
    )


# Simple loaders for md/txt/pdf
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader


def load_docs(path: str):
    base = Path(path)
    docs = []
    for p in base.rglob("*"):
        if p.suffix.lower() in {".md", ".txt"}:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
        elif p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
    return docs