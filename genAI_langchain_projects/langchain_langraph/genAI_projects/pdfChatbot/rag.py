from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os, json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import re
from functools import lru_cache

load_dotenv()


# load the doc and config file
script_dir = os.path.dirname(os.path.abspath(__file__))
policy_doc_path = os.path.join(script_dir,"data","Novatech_HR_Policy_Handbook_2026.pdf")
rag_config_path = os.path.join(script_dir,"data","rag_config.json")

try :
    pdfDoc = PyPDFLoader(policy_doc_path).load()
    with open(file=rag_config_path,mode="r",encoding="utf-8") as f:
       rag_config = json.load(f)
except FileNotFoundError as e:
    raise SystemExit(f"File not found :{e.filename}")
except json.JSONDecodeError:
    raise SystemExit(f"Config json not able to parse")

def load_embedder(cfg):
    if cfg is None:
        return None
    if cfg.get("provider")  == "openai":
        return OpenAIEmbeddings(model=cfg["model"])
    if cfg.get("provider")  == "huggingface":
        return HuggingFaceEmbeddings(model_name = cfg["model"],encoding_kwargs= {"normalize_embeddings":True})
    return ValueError(f"Unidentified provider:{cfg['provider']}")
               
@lru_cache(maxsize=1)
def get_rag():

    # Chunking docs
    pdfSplitter = RecursiveCharacterTextSplitter(chunk_size = rag_config["chunk_size"], chunk_overlap = rag_config["chunk_overlap"])
    pdfChunks = pdfSplitter.split_documents(pdfDoc)

    def make_bm25():
        bm25 = BM25Retriever.from_documents(
            pdfChunks, preprocess_func=lambda text: re.findall(r"\w+", text.lower())
        )
        bm25.k = rag_config["k"]
        return bm25

    if rag_config["retriever_type"] == "bm25":
        return make_bm25() 

    vectorStore = FAISS.from_documents(documents=pdfChunks, embedding=load_embedder(rag_config["embedding"]))
    dense = vectorStore.as_retriever(search_kwargs = {"k":rag_config["k"]})


    if rag_config["retriever_type"] == "hybrid":
        hybrid = EnsembleRetriever(retrievers=[dense,make_bm25()],weights=rag_config["weights"])
        return hybrid
   
    
    return dense
