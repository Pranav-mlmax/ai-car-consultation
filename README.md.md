# 🚗 AutoAdvisor India

> RAG-powered AI car consultant for the Indian market — get your top 3 picks based on budget, body type, fuel, and priorities.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Flask](https://img.shields.io/badge/Flask-REST_API-black) ![LangChain](https://img.shields.io/badge/LangChain-RAG-green) ![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-orange)

## Tech Stack

| Layer | Tech |
|---|---|
| LLM | GPT-4o-mini via OpenRouter |
| RAG | LangChain + ChromaDB + HuggingFace Embeddings |
| Backend | Flask REST API |
| Frontend | Vanilla JS + HTML/CSS |
| Deploy | Render |

## How It Works

1. User inputs budget, car type, fuel preference, usage, and priorities
2. Preferences are used to retrieve relevant chunks from a curated Indian car market knowledge base (2026 data) via ChromaDB vector search
3. Retrieved context + user profile is passed to GPT-4o-mini via a structured prompt
4. Model returns top 3 car recommendations with prices, pros/cons, and community sentiment

## Run Locally

```bash
git clone https://github.com/yourusername/car-consultant-india
cd car-consultant-india

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Add your OpenRouter API key to .env

python app.py
```

Visit `http://localhost:5000`

## Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service → Connect repo
3. Set environment variable: `OPENROUTER_API_KEY=your_key`
4. Render auto-detects `render.yaml` and deploys

## Project Structure

```
├── app.py              # Flask app + RAG chain
├── market_data.py      # Indian car market knowledge base (2026)
├── templates/
│   └── index.html      # Frontend UI
├── requirements.txt
├── render.yaml         # Render deployment config
└── .env.example
```

## Get an OpenRouter API Key

Sign up at [openrouter.ai](https://openrouter.ai) — free tier available.
