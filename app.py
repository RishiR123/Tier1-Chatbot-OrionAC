from flask import Flask, request, jsonify
from flask import render_template
from dotenv import load_dotenv
import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as gemini

# Load API key from Render environment variable
api = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
gemini.configure(api_key=api)

# Load model
model = gemini.GenerativeModel("gemini-1.5-flash")

# Initialize Flask app
app = Flask(__name__)

# Load knowledge base once at startup
loader = PyPDFLoader(file_path="data.pdf")
docs = loader.load()

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api,
    task_type="retrieval_query"
)

# Vector DB
text_data = [doc.page_content for doc in docs]
vectordb = Chroma.from_texts(texts=text_data, embedding=embeddings, persist_directory="rova_db")

# Root route
@app.route('/')
def home():
    return render_template('index.html')

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query")
    results = vectordb.similarity_search(query, k=3)
    context = "\n\n".join([r.page_content for r in results])

    prompt = f"User's Query: {query}\n\nContext:\n{context}\n\nAnswer in a human-like way."
    response = model.generate_content(prompt)
    answer = response.candidates[0].content.parts[0].text

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

