import shutil, time, tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from .rag import RAGEngine
from . import monitoring

app = FastAPI(title="LLM Document Q&A (RAG)")
engine = RAGEngine()


class Question(BaseModel):
    question: str
    k: int = 4


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = "." + file.filename.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
    n = engine.ingest(tmp.name)
    return {"filename": file.filename, "chunks_indexed": n}


@app.post("/ask")
def ask(q: Question):
    start = time.time()
    try:
        result = engine.ask(q.question, q.k)
    except ValueError as e:
        raise HTTPException(400, str(e))
    latency = time.time() - start
    drift = monitoring.log_query(q.question, result["answer"], latency,
                                 engine.embed_query(q.question), result["sources"][0]["distance"])
    return {**result, "latency_s": round(latency, 3), "drift_pvalue": drift}
