"""RAG pipeline: LangChain + FAISS + OpenAI."""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_DIR = Path("faiss_index")
PROMPT = """Answer using ONLY the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""


class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self.store = None
        if INDEX_DIR.exists():
            self.store = FAISS.load_local(str(INDEX_DIR), self.embeddings, allow_dangerous_deserialization=True)

    def ingest(self, path: str) -> int:
        loader = PyPDFLoader(path) if path.lower().endswith(".pdf") else TextLoader(path, encoding="utf-8")
        chunks = self.splitter.split_documents(loader.load())
        if self.store is None:
            self.store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.store.add_documents(chunks)
        self.store.save_local(str(INDEX_DIR))
        return len(chunks)

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    def ask(self, question: str, k: int = 4) -> dict:
        if self.store is None:
            raise ValueError("No documents ingested yet.")
        docs = self.store.similarity_search_with_score(question, k=k)
        context = "\n\n".join(d.page_content for d, _ in docs)
        answer = self.llm.invoke(PROMPT.format(context=context, question=question)).content
        return {
            "answer": answer,
            "sources": [{"source": d.metadata.get("source"), "page": d.metadata.get("page"), "distance": float(s)} for d, s in docs],
        }
