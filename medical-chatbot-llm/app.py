# Main Application (Using Flask) - app.py

from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt  # Assuming you defined `system_prompt` in prompt.py

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Initialize Flask app
app = Flask(__name__)  # ✅ FIXED: Was _name_

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Make sure environment variables are available to dependencies
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize HuggingFace Embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index name
index_name = "healthybot"

# Load existing Pinecone vector store
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Options: gemini-pro, gemini-1.5-pro, gemini-1.5-flash
    temperature=0.4,
    max_tokens=500,
    google_api_key=GOOGLE_API_KEY
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create Q&A Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create RAG Chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# Route: Home
@app.route("/")
def index():
    return render_template("healthybot.html")  # Ensure this HTML exists in /templates/


# Route: Chat Handler
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    
    return str(response["answer"])


# Run the Flask app
if __name__ == "__main__":  # ✅ FIXED: Was _main_
    app.run(host="0.0.0.0", port=8080, debug=True)
