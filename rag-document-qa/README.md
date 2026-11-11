# LLM-Based Document Q&A (RAG)
LangChain + FAISS + OpenAI, served by FastAPI in Docker, with MLflow monitoring and drift detection.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
uvicorn app.main:app --reload          # or: docker compose up --build
curl -F "file=@docs/sample.txt" localhost:8000/ingest
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What is RAG?"}'
mlflow ui                               # view latency / drift metrics
```
