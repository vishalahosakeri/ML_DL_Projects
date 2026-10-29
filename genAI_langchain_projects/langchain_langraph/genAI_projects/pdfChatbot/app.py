from agent import getAgent

agent = getAgent()

def ask(question: str, thread_id: str = "demo"):
    """Ask the agent a question and print the final answer."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    print("\n" + "=" * 70)
    print("Q:", question)
    print("A:", result["messages"][-1].content)

if __name__ == "__main__":
    # Demo 1: policy question -> RAG tool
    ask("How many days of paternity leave do I get, and by when must I take it?")
    # Demo 2: calculation -> leave tool
    ask("I carried forward 25 sick leaves and used 4 this year. How many are left?")
    # Demo 3: memory follow-up (same thread_id) -> salary tool
    ask("If I take 2 more days as LWP and my CTC is 9 LPA, what will my salary be?")
    # Demo 4: mixed question -> both RAG and calculator
    ask("How many casual leaves can I take in a row? Also, I've used 5 casual leaves, how many are left?")