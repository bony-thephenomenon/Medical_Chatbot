from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

# RAG imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# your helper files
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

# vector DB (Chroma)
from langchain_community.vectorstores import Chroma

# LLM (Groq)
from langchain_groq import ChatGroq


app = Flask(__name__)

# 🔹 Embeddings
embedding = download_hugging_face_embeddings()

# 🔹 Vector DB
db = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# 🔹 LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 🔹 Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# 🔹 Chains
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


# 🔹 Routes
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]

    try:
        response = rag_chain.invoke({"input": user_input})
        answer = response.get("answer", "No response generated.")
    except Exception as e:
        answer = f"Error: {str(e)}"

    return str(answer)


# 🔹 Run app (Render compatible)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))