from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tools import web_search, scrape_url
from llm import getLLm

#get llm
llm = getLLm()

def build_agent(system_msg:str,tools:list)->AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ('system',system_msg),
        ('human','{input}'),
        ('placeholder','{agent_scratchpad}')
    ])
    agent = create_tool_calling_agent(llm,tools, prompt)
    return AgentExecutor(agent=agent , tools=tools,max_iterations=6,verbose=False)

search_agent = build_agent('''You are search agent. Use the given web search tool and find the most relevant and up to date
                           information for given topic. Return the most relevant results, each with url, title and the one line summary 
''',[web_search])

read_agent = build_agent('''Youa are reader agent. Use the given scrape url tool to read information from 2 to 3 urls given by search result.
                         Give the results like key findings, arguments and information grouped by url ''',[scrape_url])




