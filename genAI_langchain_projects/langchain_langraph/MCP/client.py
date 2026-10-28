import asyncio
import sys
from pathlib import Path
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client

server_path = Path(__file__).parent/"server.py"

async def main():

    # strat server
    server_paramters = StdioServerParameters(command=sys.executable,args=[str(server_path)])

    # establish channel 
    async with stdio_client(server_paramters) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()

            resposne = await session.list_tools()

            for t in resposne.tools:
                print(t.name,"****",t.description)

            result = await session.call_tool(name="add_calc",arguments={"a":5,"b":10})

            print(result.content[0].text)

asyncio.run(main())