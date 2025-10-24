# 🧠 Contextly

A lightweight, local-first document Q&A assistant built with
**LangGraph**, **FastAPI**, and **Qdrant** --- powered by free local
models via **Ollama**.

------------------------------------------------------------------------

## 🚀 Features

-   Retrieval-Augmented Generation (RAG) pipeline using LangGraph.
-   Local Qdrant vector database for semantic search.
-   FastAPI backend with `/ingest` and `/ask` endpoints.
-   Ollama integration for free local LLMs and embeddings (e.g.,
    `nomic-embed-text`, `qwen2.5`).
-   Docker-ready with Qdrant and optional Ollama services.

------------------------------------------------------------------------

## 📂 Project Structure

    Contextly/
    ├─ app/
    │  ├─ deps.py
    │  ├─ graph.py
    │  ├─ ingest.py
    │  ├─ main.py
    │  ├─ schemas.py
    │  └─ utils.py
    ├─ data/sample/example.md
    ├─ .env.example
    ├─ requirements.txt
    ├─ docker-compose.yml
    └─ README.md

------------------------------------------------------------------------

## ⚙️ Setup Instructions

1.  Start Qdrant:

    ``` bash
    docker compose up -d qdrant
    ```

2.  Create virtual environment:

    ``` bash
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

3.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

4.  Copy `.env.example` → `.env` and set `USE_OLLAMA=true`.

5.  Run Ollama server and pull models:

    ``` bash
    ollama pull nomic-embed-text
    ollama pull qwen2.5:3b-instruct-q4_K_M
    ```

6.  Ingest data:

    ``` bash
    python -m app.ingest --path data\sample --collection docs_ollama
    ```

7.  Run the API:

    ``` bash
    uvicorn app.main:app --reload --port 8000
    ```

8.  Test at <http://localhost:8000/docs>

------------------------------------------------------------------------

## 💡 Example Request

``` bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" ^
  -d '{"question":"What is this project for?","collection":"docs_ollama"}'
```

------------------------------------------------------------------------

## 🧩 Technologies

LangGraph • FastAPI • Qdrant • Ollama • Docker • Python 3.10+

------------------------------------------------------------------------

## 👩‍💻 Author

Created by **Beyza Nur Özkara**.
