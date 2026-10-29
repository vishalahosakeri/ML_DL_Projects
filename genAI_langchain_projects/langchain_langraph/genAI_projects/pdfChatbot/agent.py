from tools import leave_balance,leave_encashment,search_hr_policy,salary_after_lwp
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
llm = init_chat_model("groq:openai/gpt-oss-20b", temperature=0)

def getAgent():
    
    SYSTEM_PROMPT = """You are the HR assistant for Novatech Solutions employees.
    
    Tools:
    - search_hr_policy: any question about Novatech HR policies
    - leave_balance: remaining earned, casual or sick leave
    - leave_encashment: earned leave encashment at exit
    - salary_after_lwp: salary after Leave Without Pay deduction
    
    Rules:
    - Never do arithmetic yourself. Always use the calculator tools.
    - Answer policy questions only from the search results and mention the page number.
    - If the handbook does not cover the question, say so and suggest contacting
    the HR Helpdesk (hrhelpdesk@novatech-example.in).
    - For general questions unrelated to HR, answer briefly without tools."""
 
    agent = create_agent(
    model=llm,
    tools=[search_hr_policy, leave_balance, leave_encashment, salary_after_lwp],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),   # remembers the conversation per thread_id
    )

    return agent
