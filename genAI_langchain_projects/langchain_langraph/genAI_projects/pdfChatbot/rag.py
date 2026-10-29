from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS


def get_rag():
    #load key for hugging face
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

    #1. load the doc
    loader = PyPDFLoader("/Users/sanvijanvi/Documents/SJ_HomeWork/Novatech_HR_Policy_Handbook_2026.pdf")
    pdfDoc = loader.load()

    #2. Chunking docs
    seperators = ["\n\n","\n"," ",""]
    pdfSplitter = RecursiveCharacterTextSplitter(separators=seperators,chunk_size = 1000, chunk_overlap = 100)
    pdfChunks = pdfSplitter.split_documents(pdfDoc)

    #3. Embeddings
    pdfEmbeder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #4.  vector db
    vectorStore = FAISS.from_documents(documents=pdfChunks, embedding=pdfEmbeder)

    # 5. retriever
    base_retriever = vectorStore.as_retriever(search_type = "similarity",search_kwargs = {"k":1})
    
    return base_retriever
