from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from market_data import MARKET_DATA
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=2048
)

docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(
    [Document(page_content=MARKET_DATA)]
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
retriever = Chroma.from_documents(docs, embeddings).as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
You are a senior Indian car consultant with deep knowledge from Team-BHP, r/CarsIndia and r/IndianCarsUnder10Lakhs communities.

Customer Profile:
{user_summary}

Verified Market Data:
{rag_context}

Instructions:
- Budget is in INR. If given in lakhs convert it (e.g. 12 lakhs = 12,00,000 INR).
- Recommend exactly top 3 cars matching this customer.
- For each car include: model and variant, ex-showroom price, on-road price (add 18%), why it fits this customer specifically, 2 Pros, 1 Con, Team-BHP take, Reddit community opinion.
- End with a final verdict: one pick with a clear reason.
- Use real prices from the market data. Be direct and specific. No filler.
""")

chain = prompt | llm | StrOutputParser()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.json
    prefs = data.get("preferences", {})

    user_summary = "\n".join([f"{k}: {v}" for k, v in prefs.items()])
    rag_context = "\n".join([d.page_content for d in retriever.invoke(prefs.get("body_type", "") + " " + prefs.get("budget", ""))])

    result = chain.invoke({"user_summary": user_summary, "rag_context": rag_context})
    return jsonify({"recommendation": result})


if __name__ == "__main__":
    app.run(debug=True)
