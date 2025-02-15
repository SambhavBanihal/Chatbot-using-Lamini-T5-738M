import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
import chromadb
from chromadb.config import Settings

# ✅ Directories
DOCS_DIR = "docs"
DB_DIR = "db"

# ✅ Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# ✅ Set your local model path
MODEL_PATH = r"C:\Users\HP\Downloads\final\LaMini-T5-738M"

# ✅ Check if CUDA (GPU) is available; otherwise, use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ✅ Load model with proper device settings
try:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
    ).to(device)
    print("✅ Model loaded successfully on", device)
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

@st.cache_resource
def llm_pipeline():
    """Loads the text generation pipeline"""
    pipe = pipeline(
        "text2text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def qa_llm():
    """Sets up the retrieval-based QA system"""
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

def process_answer(instruction):
    """Processes the answer using the local model"""
    qa = qa_llm()
    try:
        generated_text = qa(instruction)
        return generated_text["result"], generated_text
    except Exception as e:
        return "❌ Error generating response", {"error": str(e)}

def ingest_pdf(uploaded_file):
    """Save uploaded PDF and create embeddings"""
    file_path = os.path.join(DOCS_DIR, uploaded_file.name)
    
    # Save file to docs/ directory
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ {uploaded_file.name} uploaded successfully!")

    # Load PDF and create embeddings
    documents = PDFMinerLoader(file_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(settings=Settings(persist_directory=DB_DIR))
    collection = client.get_or_create_collection(name="pdf_documents")

    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

    for i, text in enumerate(texts):
        if str(i) not in existing_ids:  # Avoid duplicate inserts
            collection.add(
                ids=[str(i)],
                documents=[text.page_content],
                metadatas=[text.metadata],
            )

    st.success("✅ Embeddings successfully created and stored in ChromaDB!")

def main():
    """Streamlit UI"""
    st.title("📄 Chat-Bot EDAS built by Sambhav Banihal (Intern) --Mentor Name:  Mr. Saurabh Kumar Chief Manager (Programming)Exploration Data Services, ExploraProject Title: Responsive Pdf Chatbot")
    
    with st.expander("ℹ About the App"):
        st.markdown(
            "This is a Generative AI-powered Chat-bot built for EDAS ONGC that responds to questions about your PDF files."
        )

    # 🔹 File Uploader
    uploaded_file = st.file_uploader("📤 Upload a PDF to Store & Search", type=["pdf"])
    if uploaded_file is not None:
        ingest_pdf(uploaded_file)

    # 🔹 Q&A Section
    question = st.text_area("📝 Enter your Question")
    
    if st.button("🔍 Ask"):
        st.info("🛠 Processing your question...")
        answer, metadata = process_answer(question)
        st.success("✅ Answer:")
        st.write(answer)
        st.json(metadata)

if __name__ == "__main__":
    main()
