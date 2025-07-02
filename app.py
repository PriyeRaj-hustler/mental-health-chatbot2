import os

from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
from dotenv import load_dotenv
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
# Initialize components
def initialize_components():
    # LLM
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,  # Replace with your key
        model_name="llama-3.3-70b-versatile"
    )

    # VectorDB
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        loader = DirectoryLoader("./data/", glob="*.pdf", loader_cls=PyPDFLoader)
        texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
        vector_db = Chroma.from_documents(
            texts,
            HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            persist_directory=db_path
        )
    else:
        vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )

    # QA Chain
    prompt_template = """You are a compassionate mental health chatbot. Respond using:
    {context}
    User: {question}
    Chatbot:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )

# Initialize once
qa_chain = initialize_components()

# Corrected chat function
def respond(message, history):
    try:
        response = qa_chain.invoke({"query": message})
        return response["result"]
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Launch interface
demo = gr.ChatInterface(
    respond,
    title="Mental Health Chatbot",
    description="Ask me anything about mental health support",
    examples=["How to cope with anxiety?", "What are signs of depression?"],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()