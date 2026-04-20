import asyncio
import json
import os
import sys
from dotenv import load_dotenv

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

load_dotenv()

# Import our existing functions from previous layers
sys.path.append(os.path.dirname(__file__))
from queries import (
    get_customer_profile,
    get_order_details,
    get_complaint_history,
    get_recent_orders
)
from rag_pipeline import retrieve_relevant_chunks

# ─────────────────────────────────────────────
# CREATE THE MCP SERVER
# This is the server instance that any
# MCP-compatible client can connect to
# ─────────────────────────────────────────────

server = Server("quickbite-support-tools")

# ─────────────────────────────────────────────
# TOOL DISCOVERY
# When a client connects and asks "what tools
# do you have?", this function responds with
# the full list of available tools and their
# schemas. This is how MCP clients know what
# they can call without any hardcoding.
# ─────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_customer_profile",
            description="""Retrieves full customer profile including segment 
label (high_value, standard, at_risk), lifetime orders, average order value, 
and complaint count. Call this first for any support interaction.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "mobile_number": {
                        "type": "string",
                        "description": "Customer mobile number"
                    }
                },
                "required": ["mobile_number"]
            }
        ),
        types.Tool(
            name="get_order_details",
            description="""Retrieves full order details including all items, 
quantities, prices, delivery address, order status, and timestamp. 
Call when customer references a specific order.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID in format ORD followed by numbers"
                    }
                },
                "required": ["order_id"]
            }
        ),
        types.Tool(
            name="get_recent_orders",
            description="""Retrieves the most recent orders for a customer 
when no specific order ID is provided. Use this to identify which order 
the customer is likely referring to.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "mobile_number": {
                        "type": "string",
                        "description": "Customer mobile number"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent orders to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["mobile_number"]
            }
        ),
        types.Tool(
            name="get_complaint_history",
            description="""Retrieves last 10 support tickets for a customer 
showing complaint categories, resolutions, and feedback scores. 
Call when customer mentions previous complaints or seems frustrated.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "mobile_number": {
                        "type": "string",
                        "description": "Customer mobile number"
                    }
                },
                "required": ["mobile_number"]
            }
        ),
        types.Tool(
            name="search_knowledge_base",
            description="""Searches policy knowledge base for relevant 
resolution guidelines using semantic similarity. Use descriptive queries 
like 'cold food refund policy high value customer'.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Descriptive search query for the policy needed"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default 2)",
                        "default": 2
                    }
                },
                "required": ["query"]
            }
        )
    ]

# ─────────────────────────────────────────────
# TOOL EXECUTION
# When a client calls a tool, this function
# receives the tool name and arguments,
# executes the right function, and returns
# the result in MCP's standard format.
# ─────────────────────────────────────────────

@server.call_tool()
async def call_tool(
    name: str,
    arguments: dict
) -> list[types.TextContent]:

    try:
        if name == "get_customer_profile":
            result = get_customer_profile(arguments["mobile_number"])
            if not result:
                result = {"error": "Customer not found"}

        elif name == "get_order_details":
            result = get_order_details(arguments["order_id"])
            if not result:
                result = {"error": "Order not found"}

        elif name == "get_recent_orders":
            limit = arguments.get("limit", 5)
            result = get_recent_orders(arguments["mobile_number"], limit)

        elif name == "get_complaint_history":
            history = get_complaint_history(arguments["mobile_number"])
            result = {
                "complaint_count": len(history),
                "complaints": history
            }

        elif name == "search_knowledge_base":
            n_results = arguments.get("n_results", 2)
            chunks = retrieve_relevant_chunks(
                arguments["query"],
                n_results=n_results
            )
            result = {
                "chunks_retrieved": len(chunks),
                "policies": [
                    {
                        "title": c["title"],
                        "content": c["text"],
                        "similarity": c["similarity"]
                    }
                    for c in chunks
                ]
            }

        else:
            result = {"error": f"Unknown tool: {name}"}

        # MCP returns results as TextContent
        # JSON string so any client can parse it
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]

    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

# ─────────────────────────────────────────────
# START THE SERVER
# stdio_server means communication happens
# through standard input/output — the standard
# MCP transport for local tool servers.
# Claude Desktop connects to it this way.
# ─────────────────────────────────────────────

async def main():
    print("QuickBite Support MCP Server starting...", file=sys.stderr)
    print("Tools available:", file=sys.stderr)
    print("  - get_customer_profile", file=sys.stderr)
    print("  - get_order_details", file=sys.stderr)
    print("  - get_recent_orders", file=sys.stderr)
    print("  - get_complaint_history", file=sys.stderr)
    print("  - search_knowledge_base", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())