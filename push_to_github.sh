#!/usr/bin/env bash
# Usage: ./push_to_github.sh <github-username>
set -e
USER="${1:?Usage: ./push_to_github.sh <github-username>}"
git init -b main
git add .
git commit -m "Initial commit: ML & GenAI portfolio"
if command -v gh >/dev/null; then
  gh repo create "$USER/ml-portfolio" --public --source=. --push \
    --description "ML, MLOps & GenAI portfolio: RAG, multi-cloud MLOps, agentic AI, CNNs, NLP"
else
  git remote add origin "https://github.com/$USER/ml-portfolio.git"
  git push -u origin main
fi
echo "Portfolio: https://github.com/$USER/ml-portfolio"
