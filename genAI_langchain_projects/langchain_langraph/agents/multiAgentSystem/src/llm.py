import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# models = Groq().models.list()
# print([m.id for m in models.data])
#llama-3.1-8b-instant
def getLLm():
    llm_groq = init_chat_model(model="groq:qwen/qwen3.8-27b",
        temperature=0,
        max_tokens=800,
        max_retries=3,
        timeout=60)
    return llm_groq
