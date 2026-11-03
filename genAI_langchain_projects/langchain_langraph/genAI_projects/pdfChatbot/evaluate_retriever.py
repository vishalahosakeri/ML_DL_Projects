import json
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import re
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir,"questions.json")

try:
    loader = PyPDFLoader(os.path.join(script_dir,"data","Novatech_HR_Policy_Handbook_2026.pdf"))
    pdfDoc = loader.load()
    with open(json_path,"r",encoding="utf-8") as file:
        test_set = json.load(file)
except FileNotFoundError as e:
    raise SystemExit(f"Missing file:{e.filename}")
except json.JSONDecodeError:
    print("JSON parse error")

embedders = {
    "MiniLM (local)": HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    "bge-small (local)": HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    ),
    "OpenAI 3-small (API)": OpenAIEmbeddings(model="text-embedding-3-small"),
}
model_registry = {
    "MiniLM (local)":       {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "bge-small (local)":    {"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"},
    "OpenAI 3-small (API)": {"provider": "openai",      "model": "text-embedding-3-small"},
}


sizes = [(300, 30), (500, 50), (1000, 100)]
weight_options = [[0.7, 0.3], [0.5, 0.5], [0.3, 0.7]]
results = []

def evaluate_retriever(retriever, test_set):
    hit3 = hit5 = rr_total = 0
    for case in test_set:
        results = retriever.invoke(case["question"])
        pages = [d.metadata["page"]+1 for d in results]

        hit3 += any(p for p in pages[:3] if p in case["expected_pages"] )
        hit5 += any(p for p in pages[:5] if p in case["expected_pages"] )

        rank = [i+1 for i, p in enumerate(pages) if p in case["expected_pages"] ]
        rr_total += 1/rank[0] if rank else 0

    n = len(test_set)

    return {"hit3":hit3/n,"hit5":hit5/n,"mrr":round(rr_total/n,3)}

def log(size, overlap, embedding, retriever_type, weights, retriever):
    scores = evaluate_retriever(retriever=retriever, test_set= test_set)
    results.append({"chunk_size":size,"chunk_overlap":overlap,"emebdding":embedding,"retriever_type":retriever_type, "weights":weights,
                    **scores})
    
for size, overlap in sizes :
    chunks = RecursiveCharacterTextSplitter(chunk_size = size, chunk_overlap = overlap).split_documents(documents=pdfDoc)
    bm25_retriever = BM25Retriever.from_documents(documents=chunks, preprocess_func= lambda text : re.findall(r"\w+",text.lower()))
    bm25_retriever.k =5

    log(size, overlap, "-", "bm25", "-", bm25_retriever)
    for name, embed in embedders.items():
        base_retriever = (FAISS.from_documents(documents= chunks, embedding=embed)).as_retriever(search_kwargs={"k": 5})
        log(size, overlap, name, "dense","-",base_retriever)

        for w in weight_options:
            ensemble_retriever = EnsembleRetriever(retrievers=[base_retriever,bm25_retriever],weights=w)
            log(size, overlap, name, "hybrid",w,ensemble_retriever)

df = pd.DataFrame(results).sort_values(["hit3","mrr","hit5"],ascending=False)
df.to_csv(path_or_buf= os.path.join(script_dir,"data","retrieval_experiments.csv"),index= False)

best = df.iloc[0]

#### after writing observations to csv, choose the suitable one and prepare the config
final_config = {
    "chunk_size": int(best["chunk_size"]),
    "chunk_overlap": int(best["chunk_overlap"]),
    "embedding": model_registry.get(best["emebdding"]),
    "retriever_type": best["retriever_type"],
    "weights": best["weights"],
    "k": 5,
    "scores":{
        "hit@3":float(best["hit3"]),
        "hit@5":float(best["hit5"]),
        "mrr":float(best["mrr"])
    },
    "reson":"Why OpenAI dense: perfect Hit@3 with the highest MRR among those, and a simpler retriever.Why 500 over 1000: a tie on all metrics, so the smaller chunk size sends less unrelated text to the LLM."

}

with open(os.path.join(script_dir,"data","rag_config.json"),"w") as f :
    json.dump(final_config,f,indent=2)