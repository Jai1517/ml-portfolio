<div align="center">

# Jayakumar Sivanandham — ML & GenAI Portfolio

**AI/ML Engineer · MLOps · LLMs & RAG · Cloud**
MSc Computer Science, Ravensbourne University London · United Kingdom

[LinkedIn](https://linkedin.com/in/jayakumar-sivan) · [Email](mailto:jayakumar1505@outlook.com)

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c) ![TensorFlow](https://img.shields.io/badge/TensorFlow-ff6f00) ![AWS](https://img.shields.io/badge/AWS-SageMaker-232f3e) ![GCP](https://img.shields.io/badge/GCP-Vertex%20AI-4285f4) ![Azure](https://img.shields.io/badge/Azure-ML-0078d4) ![Docker](https://img.shields.io/badge/Docker-2496ed) ![MLflow](https://img.shields.io/badge/MLflow-0194e2)

</div>

## Projects

| Project | What it demonstrates | Stack |
|---|---|---|
| [**LLM Document Q&A (RAG)**](rag-document-qa) | Production-ready, document-grounded Q&A; Dockerised FastAPI service; MLflow monitoring and data drift detection | LangChain, FAISS, OpenAI, FastAPI, Docker, MLflow |
| [**End-to-End MLOps – Multi-Cloud**](mlops-multicloud) | One scikit-learn pipeline deployed to AWS, GCP and Azure; DVC versioning; MLflow lifecycle; DevSecOps CI | scikit-learn, DVC, MLflow, EC2, S3, SageMaker, Vertex AI, Azure ML, GitHub Actions |
| [**Agentic AI Chatbot**](agentic-chatbot) | Autonomous task execution via tool use and a plan → act → review prompt chain | OpenAI GPT, function calling |
| [**Image Classification CNN – SageMaker**](cnn-sagemaker) | CNNs in TensorFlow and PyTorch, evaluation, SageMaker deployment | TensorFlow, PyTorch, SageMaker |
| [**Sentiment Analysis (NLP)**](sentiment-nlp) | NLP pipeline with feature-store-ready output; precision/recall/F1 evaluation | NLTK, spaCy, TF-IDF, Parquet |

## Architecture at a glance

```
Data ──DVC──► Train (scikit-learn / TF / PyTorch) ──MLflow──► Model registry
                                                      │
                     ┌────────────────┬───────────────┼───────────────┐
                     ▼                ▼               ▼               ▼
                AWS SageMaker    Vertex AI        Azure ML       EC2 / Docker
                                                                  (FastAPI)

Docs ─► chunk ─► embeddings ─► FAISS ─► retrieve top-k ─► GPT ─► grounded answer ─► MLflow drift metrics
```

## Skills

**AI/ML:** Machine Learning, Deep Learning, NLP, LLMs, RAG, Agentic AI, Data Drift Detection, Feature Stores
**Platforms:** SageMaker, Vertex AI, Azure ML, OpenAI API, AutoML, OCR, Speech-to-Text
**MLOps/DevOps:** MLflow, DVC, Docker, CI/CD, Git, DevSecOps
**Frameworks:** PyTorch, TensorFlow, FastAPI, Pandas, NumPy, scikit-learn, NLTK, spaCy

## Run locally

```bash
git clone https://github.com/<your-username>/ml-portfolio.git
cd ml-portfolio
python -m venv .venv && source .venv/bin/activate
cp .env.example .env          # add your OPENAI_API_KEY
code ml-portfolio.code-workspace
```
Each project folder has its own README with exact run commands.
