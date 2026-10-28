
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Company tools resource and prompt")

@mcp.tool()
def add_calc(a : int, b: int)->int :
    '''This tool is for adding the integer numbers and returning the result'''
    return a+b

@mcp.resource(uri="company://policy/leave")
def leave_resource()->str:
    '''This resource gives detailed information on leave'''
    return "This is leave policy based on expereience and level of employee."

@mcp.prompt()
def leave_prompt(empId)->str:
    '''This prompt gives generic prompt for employee leave info.'''
    return (f"Explain the leave policy to employee {empId}. "
        "Use clear and simple language.")


if __name__=="__main__":
    mcp.run(transport="stdio")


