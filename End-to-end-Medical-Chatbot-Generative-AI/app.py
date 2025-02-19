from flask import Flask, render_template, jsonify, request
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Flask App
app = Flask(__name__)

# Hardcoded API Keys
PINECONE_API_KEY = "pcsk_6EaG3Q_S1PbWepX4ygjoe6BVvJ1PUeZwHTGipCJWr1ugv9dZnijTB6ViHLRUmRpeBjfcaa"
OPENAI_API_KEY = "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"

# Set Environment Variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Updated class

# Pinecone Index
index_name = "medical"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# OpenAI LLM
llm = OpenAI(temperature=0.4, max_tokens=500)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"Input: {msg}")
    response = rag_chain.invoke({"input": msg})
    print(f"Response: {response['answer']}")
    return str(response["answer"])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6789)  # Use port 6789 or any free port
