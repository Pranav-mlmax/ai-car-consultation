from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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

prompt = ChatPromptTemplate.from_template("""
You are a senior Indian car consultant with deep knowledge from Team-BHP, r/CarsIndia and r/IndianCarsUnder10Lakhs communities.

Customer Profile:
{user_summary}

Indian Car Market Data (2026):
{market_data}

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
    result = chain.invoke({"user_summary": user_summary, "market_data": MARKET_DATA})
    return jsonify({"recommendation": result})


if __name__ == "__main__":
    app.run(debug=True)
