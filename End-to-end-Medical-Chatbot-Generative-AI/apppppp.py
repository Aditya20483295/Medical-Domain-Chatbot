from flask import Flask, request
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

app = Flask(__name__)

# Hardcoded API Keys
PINECONE_API_KEY = "pcsk_6EaG3Q_S1PbWepX4ygjoe6BVvJ1PUeZwHTGipCJWr1ugv9dZnijTB6ViHLRUmRpeBjfcaa"
OPENAI_API_KEY = "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"

# Set Environment Variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
system_prompt = "You are a helpful medical assistant. Use the provided context to answer the user's query."
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        ("assistant", "Based on the following context: {context}\nAnswer the user's query."),
    ]
)

# Chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Chatbot</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(120deg, #6a11cb, #2575fc);
                min-height: 100vh;
                font-family: 'Poppins', sans-serif;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
            }
            .chat-container {
                background: rgba(0, 0, 0, 0.6);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 40px;
                width: 100%;
                max-width: 700px;
                box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.3);
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                font-weight: bold;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
            }
            .form-control {
                background-color: rgba(255, 255, 255, 0.8);
                border-radius: 20px;
                padding: 15px;
            }
            .btn-primary, .btn-speak {
                border-radius: 20px;
                padding: 10px 20px;
                transition: all 0.3s ease;
            }
            .btn-primary:hover {
                background-color: #00c6ff;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            }
            .btn-speak {
                background-color: #d63384;
                color: white;
            }
            .btn-speak:hover {
                background-color: #ff007a;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            }
            .response-box {
                background: white;
                color: black;
                padding: 20px;
                border-radius: 15px;
                margin-top: 20px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            }
            .response-header {
                font-weight: bold;
                margin-bottom: 10px;
            }
        </style>
        <script>
            function startListening() {
                const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                recognition.lang = 'en-US';
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;

                recognition.start();

                recognition.onresult = function(event) {
                    const voiceInput = event.results[0][0].transcript;
                    document.getElementById("userInput").value = voiceInput;
                };

                recognition.onerror = function(event) {
                    alert("Speech recognition error: " + event.error);
                };
            }
            function handleAnimation() {
                const button = document.getElementById("sendBtn");
                button.classList.add("sending");
                setTimeout(() => button.classList.remove("sending"), 300);
            }
        </script>
    </head>
    <body>
        <div class="chat-container">
            <h1>Medical Chatbot</h1>
            <form id="chatForm" method="POST" action="/get" onsubmit="handleAnimation()">
                <div class="mb-3">
                    <input type="text" id="userInput" name="msg" class="form-control" placeholder="Type your query or use voice input" required>
                </div>
                <div class="d-flex justify-content-between">
                    <button id="sendBtn" type="submit" class="btn btn-primary">Send</button>
                    <button type="button" class="btn btn-speak" onclick="startListening()">🎤 Speak</button>
                </div>
            </form>
        </div>
    </body>
    </html>
    """

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    answer = response["answer"]
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Response</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background: linear-gradient(120deg, #2575fc, #6a11cb);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                font-family: 'Poppins', sans-serif;
            }}
            .response-container {{
                background: rgba(0, 0, 0, 0.6);
                border-radius: 15px;
                padding: 40px;
                max-width: 600px;
                width: 100%;
                text-align: center;
                box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.3);
            }}
            .btn-back {{
                background: #ff7eb3;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 10px 20px;
                transition: all 0.3s ease;
            }}
            .btn-back:hover {{
                background: #ff4a8d;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            }}
        </style>
    </head>
    <body>
        <div class="response-container">
            <h2>Chatbot Response</h2>
            <p class="response-box"><span class="response-header">Response:</span> {answer}</p>
            <a href="/" class="btn btn-back">Back to Chat</a>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6789)
