from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Company tools")

@mcp.tool()
def add_calc(a: int, b: int)->int:
    """ This tool is calculator for adding numbers.Returns the addition"""
    return a+b

@mcp.resource("company://policies/leave")
def resourceDetails()->str:
    """gives details about document"""
    return """Details of resource"""

@mcp.prompt()
def prompt_template(employee_id)->str:
    """create prompt for leave policy """
    return (f"Explain the leave policy to employee {employee_id}. "
        "Use clear and simple language.")

if __name__ == "__main__":
    mcp.run(transport="stdio")