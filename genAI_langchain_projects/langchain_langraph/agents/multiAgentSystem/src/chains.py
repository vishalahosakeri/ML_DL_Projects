from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from llm import getLLm

llm = getLLm()
writePrompt = ChatPromptTemplate.from_messages([
    ('system','You are writer, writing report for given topic.Write clear and structured report(intro, key findings, conclusion, sources) using '
    'OnlY materials given'),
    ('human','Topic :{topic}\nsearch results :{search_results}\nscraped content:{scraped_content}')
])

write_chain = writePrompt | llm| StrOutputParser()

criticPrompt = writePrompt = ChatPromptTemplate.from_messages([
    ('system','You are strict research critic.Check the report given. Start with score: out of 10, then ONLY 3 gaps, 3 biases - one line each. Keep the whole review under 250 words. '),
    ('human','Topic :{topic}\nSource material :{scraped_content}\nreport:{report}')
])

critc_chain = criticPrompt | llm | StrOutputParser()
