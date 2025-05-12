from flask import Flask,render_template,jsonify,request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 
from dotenv import load_dotenv
from src.prompt import *
import os
app = Flask(__name__)
load_dotenv()
PINECONE_API_KEY = os.getenv("pcsk_4PfaKg_FRiDYNQN7CVYCM9MdiYhxd72asb9szV9FBhfpSCwmhs3DH4V2tTx3LAXZicgvpH")
GEMINI_API_KEY = os.getenv("AIzaSyC3WKIrRjXOnPSuD-h94KtYol3XXYb6Jbg")

embeddings =download_hugging_face_embeddings()
index_name="medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings,
)

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs={"k":3})
#llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyC3WKIrRjXOnPSuD-h94KtYol3XXYb6Jbg")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # ✅ Use a valid model name
    google_api_key="AIzaSyC3WKIrRjXOnPSuD-h94KtYol3XXYb6Jbg"
)

# Create a prompt WITHOUT system instructions (Gemini Free does not allow them)
prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        "You are a helpful assistant. Use only the context below to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{input}"
    ),
])


# Create document combination chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create RAG retrieval chain
rag_chain = create_retrieval_chain(retriever, document_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["GET","POST"])
def chat():
    msg= request.form["msg"]
    input=msg
    print(input)
    response= rag_chain.invoke({"input":msg})
    print("Response : ",response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)