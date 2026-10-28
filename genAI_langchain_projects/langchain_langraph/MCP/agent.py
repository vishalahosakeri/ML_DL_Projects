from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import sys
from pathlib import Path

SERVER_PATH = Path(__file__).parent / "server.py"

async def main():

    client = MultiServerMCPClient({
        "company": {
            "transport": "stdio",
            "command": sys.executable,
            "args":  [str(SERVER_PATH)],
        }
    })

    tools = await client.get_tools()
    for t in tools:
        print(f"{t.name}: {t.description}")

    add_tool = next(t for t in tools if t.name == "add_calc")
    result = await add_tool.ainvoke({"a": 2, "b": 3})
    print("2 + 3 =", int(result[0]["text"]))

    resource = await client.get_resources("company")
    print(f"******{resource[0].as_string() if resource else None}")

    promt = await client.get_prompt(prompt_name="prompt_template",arguments={"employee_id":"123"},server_name="company")
    print(promt,"********")

if __name__ == "__main__":
    asyncio.run(main())
