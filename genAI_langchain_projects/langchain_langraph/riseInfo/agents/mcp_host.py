from langchain_mcp_adapters.client import MultiServerMCPClient
import sys
from pathlib import Path
import asyncio

SERVER_PATH = Path(__file__).parent/"mcp_server.py"


async def main():
    client = MultiServerMCPClient({
    "company":{
        "transport":"stdio",
        "command":sys.executable,
        "args" : [str(SERVER_PATH)]
    }}
    )

    resposne = await client.get_tools()
    for t in resposne:
        print(t.name , "*******",t.description)

    add_tool = next( t for t in resposne if t.name == "add_calc")
    result = await add_tool.ainvoke({"a":10,"b":10})
    print("Result = ",result)

    #resources
    resources = await client.get_resources(server_name="company")
    print(resources[0].as_string())

    #prompt
    prompt = await client.get_prompt(server_name="company",prompt_name="leave_prompt",arguments={"empId":"123"})
    print(prompt)
asyncio.run(main())