from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True)

# RAG imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# your helper files
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

# vector DB (Chroma)
from langchain_community.vectorstores import Chroma

# LLM (Groq)
from langchain_groq import ChatGroq

load_dotenv()

app = Flask(__name__)



embedding = download_hugging_face_embeddings()

db = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

retriever = db.as_retriever(search_kwargs={"k": 3})


llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

qa_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, qa_chain)



@app.route("/")
def index():
    return render_template("chat.html")




@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]

    # 🔥 Get chat history
    chat_history = memory.load_memory_variables({}).get("chat_history", "")

    # 🔥 Combine history + current input
    full_input = f"""
You are a medical assistant.

Conversation history:
{chat_history}

Current user question:
{user_input}

IMPORTANT:
If the user asks follow-up questions like "it", "this", etc.,
refer to the last disease discussed in the conversation.

Answer ONLY about that disease.
"""
    # 🔥 Get response
    response = rag_chain.invoke({"input": full_input})
    answer = response["answer"]

    # 🔥 Save to memory
    memory.save_context(
        {"input": user_input},
        {"output": answer}
    )

    return str(answer)
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))