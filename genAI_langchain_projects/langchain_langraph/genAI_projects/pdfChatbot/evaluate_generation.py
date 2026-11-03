import os, json, time
from rag import get_rag
from langchain_groq.chat_models import ChatGroq
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
import pandas as pd

stcript_dir = os.path.dirname(os.path.abspath(__file__))
questions_path = os.path.join(stcript_dir,"questions.json")

try :
    with open(file=questions_path,mode="r") as f:
        test_set = json.load(f)
except FileNotFoundError as e:
    raise SystemExit(f"File not found:{e.filename}")
except json.JSONDecodeError:
    raise SystemExit("JSON can not be parsed")

retriever = get_rag()

answer_llm = ChatGroq(model="qwen/qwen3.8-27b",temperature=0)
judge_llm  = ChatOpenAI(model="gpt-5.4-mini",temperature=0)

answer_prompt = """
    Give answer for the question {question} using context {context} in HR policy context . If HR policy context dont have answer, 
    say you dont know. 
"""
#generate answer
def generate_answer(question):
    context = retriever.invoke(question)
    answer = answer_llm.invoke(answer_prompt.format(question = question, context = context))
    return context, answer

class Grade(BaseModel):
    faithfull : int = Field(description="1-5: is answer in the represents retreived context?")
    relevance : int = Field(description="1-5: is answer relevant to the asked question?")
    correctness : int = Field(description="1-5: does answer match the reference answer given")
    reason : str = Field(description="Give one line reason.")

grade_llm = judge_llm.with_structured_output(Grade)

judge_prompt = """
    You are grading the answer for RAG system using following inputs

    question : {question}
    system_answer : {answer}
    reference : {reference}
    context : {context}
"""

def evaluate_geneate(test_set):
    rows = []
    for case in test_set:
        print(case["id"])
        context, answer = generate_answer(case["question"])
        grade = grade_llm.invoke(judge_prompt.format(question=case["question"],answer=answer,reference=case["reference"],context = context))
        rows.append({"id":case["id"],"type":case["type"],"question":case["question"],"faithfull":grade.faithfull,"relevance":grade.relevance,
                     "correctness":grade.correctness,"reason":grade.reason})
        time.sleep(2)
    return pd.DataFrame(rows)


df = evaluate_geneate(test_set)
df.to_csv(path_or_buf=os.path.join(stcript_dir,"data","generation_eval.csv"),index=False)
metric = ["faithfull","relevance","correctness"]
print("********")
print(df[metric].mean().round(2))
print("********")
print(df.groupby("type")[metric].mean().round(2))
