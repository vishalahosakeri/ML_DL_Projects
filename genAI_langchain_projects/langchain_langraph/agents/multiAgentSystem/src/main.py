from agents import search_agent, read_agent
from chains import write_chain, critc_chain
from researchState import ResearchState

def run_pipeline(topic:str):
    state : ResearchState = {"topic":topic}

    out = search_agent.invoke({"input":f"Find recent information about {topic}"})
    state["search_results"] = out["output"]
    # print("Search results :",state["search_results"] )

    out = read_agent.invoke({"input":f"topic: {topic} \n search result : {state['search_results']}"})
    state["scraped_content"] = out["output"]
    # print("scraped_content :",state["scraped_content"] )

    report = write_chain.invoke(state)
    state["report"] = report
    # print(report)

    state["critique"] = critc_chain.invoke(state)
    print(state["critique"])
run_pipeline("AI latest news")
