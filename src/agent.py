import os
import json
from datetime import datetime
from dotenv import load_dotenv
import anthropic

load_dotenv()

# Import our existing query functions from Layer 1
import sys
sys.path.append(os.path.dirname(__file__))
from queries import (
    get_customer_profile,
    get_order_details,
    get_complaint_history,
    get_ticket_by_order
)
from rag_pipeline import retrieve_relevant_chunks

# ─────────────────────────────────────────────
# TOOL DEFINITIONS
# These tell the model what tools exist,
# what they do, and when to use them.
# The description is the most important part.
# ─────────────────────────────────────────────

AGENT_TOOLS = [
    {
        "name": "get_customer_profile",
        "description": """Retrieves the full customer profile including their 
segment label (high_value, standard, or at_risk), lifetime order count, 
average order value, and complaint count in last 10 orders.
ALWAYS call this first for any complaint — you need to know the customer 
segment before deciding on resolution. The segment determines which policy 
clause applies.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "mobile_number": {
                    "type": "string",
                    "description": "Customer's mobile number"
                }
            },
            "required": ["mobile_number"]
        }
    },
    {
        "name": "get_order_details",
        "description": """Retrieves full details of a specific order including 
all items ordered, quantities, prices, order status, delivery address, 
and order timestamp. Call this when the customer references a specific 
order or when you need to verify what was ordered.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID (format: ORD followed by numbers)"
                }
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "get_complaint_history",
        "description": """Retrieves the last 10 support tickets for a customer, 
showing complaint categories, resolutions applied, and feedback scores.
Call this when: customer mentions a previous complaint, customer seems 
frustrated suggesting history, or you need context on repeat issues.
Essential for escalation decisions.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "mobile_number": {
                    "type": "string",
                    "description": "Customer's mobile number"
                }
            },
            "required": ["mobile_number"]
        }
    },
    {
        "name": "search_knowledge_base",
        "description": """Searches the policy knowledge base for relevant 
resolution guidelines. Returns the most semantically similar policy 
documents to your query. Call this to find the correct resolution policy 
for any complaint type. Use descriptive queries like 
'cold food refund policy high value customer' or 
'repeat complaint escalation procedure'.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Descriptive search query for the policy needed"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of policy chunks to retrieve (default 2)",
                    "default": 2
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": """Escalates the ticket to a human support agent. 
Call this when: customer explicitly requests human agent, 
complaint involves repeat issue with same category, 
customer segment is at_risk with complex complaint, 
or automated resolution is not appropriate.
Provide a detailed briefing so the human agent has full context.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "mobile_number": {
                    "type": "string",
                    "description": "Customer mobile number"
                },
                "reason": {
                    "type": "string",
                    "description": "Clear reason for escalation"
                },
                "briefing": {
                    "type": "string",
                    "description": """Full context briefing for human agent 
including customer segment, complaint history, 
order details, and recommended resolution"""
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level: high, medium, or low",
                    "enum": ["high", "medium", "low"]
                }
            },
            "required": ["mobile_number", "reason", "briefing", "priority"]
        }
    }
]

# ─────────────────────────────────────────────
# TOOL EXECUTOR
# When the model decides to call a tool,
# this function actually runs it and returns
# the result back to the model.
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Executes the requested tool and returns
    the result as a JSON string.
    The model reads this result and decides
    what to do next.
    """

    if tool_name == "get_customer_profile":
        result = get_customer_profile(tool_input["mobile_number"])
        if not result:
            return json.dumps({"error": "Customer not found"})
        return json.dumps(result)

    elif tool_name == "get_order_details":
        result = get_order_details(tool_input["order_id"])
        if not result:
            return json.dumps({"error": "Order not found"})
        return json.dumps(result)

    elif tool_name == "get_complaint_history":
        result = get_complaint_history(tool_input["mobile_number"])
        return json.dumps({
            "complaint_count": len(result),
            "complaints": result
        })

    elif tool_name == "search_knowledge_base":
        n_results = tool_input.get("n_results", 2)
        chunks = retrieve_relevant_chunks(
            tool_input["query"],
            n_results=n_results
        )
        return json.dumps({
            "chunks_retrieved": len(chunks),
            "policies": [
                {
                    "title": c["title"],
                    "content": c["text"],
                    "similarity": c["similarity"]
                }
                for c in chunks
            ]
        })

    elif tool_name == "escalate_to_human":
        # In production this would create a ticket in a
        # helpdesk system like Zendesk or Freshdesk.
        # For our project we simulate it.
        escalation_id = f"ESC{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return json.dumps({
            "escalation_id": escalation_id,
            "status": "escalated",
            "assigned_to": "Senior Support Team",
            "estimated_response": "Within 2 hours",
            "briefing_received": True,
            "priority": tool_input.get("priority", "medium")
        })

    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

# ─────────────────────────────────────────────
# THE AGENT — REACT LOOP
# Reason → Act → Observe → Reason again
# Continues until model decides to stop
# calling tools and generate final response
# ─────────────────────────────────────────────

def run_agent(
    customer_query: str,
    mobile_number: str,
    max_iterations: int = 10
) -> dict:
    """
    The core agentic loop.
    
    The model reasons about which tools to call,
    we execute them, return results, and the model
    continues reasoning until it has enough
    information to generate a final response.
    
    max_iterations prevents infinite loops —
    a safety mechanism for production systems.
    """

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # System prompt — defines the agent's role,
    # available context, and reasoning instructions
    system_prompt = f"""You are an intelligent customer support agent for QuickBite, 
a QSR food delivery brand. You have access to tools that let you look up 
customer information, order details, complaint history, and company policies.

Your goal is to resolve customer complaints fairly and efficiently by:
1. Always checking the customer profile first to understand their segment
2. Gathering all relevant context before deciding on resolution
3. Applying the correct policy based on customer segment
4. Escalating when the customer requests it or when policy requires it
5. Being empathetic and professional in all responses

The customer's mobile number is: {mobile_number}

Think step by step. Use tools to gather information before responding.
Do not make assumptions — verify with tools first."""

    # Initial message from customer
    messages = [
        {"role": "user", "content": customer_query}
    ]

    # Tracking for observability
    tools_called = []
    tool_results = []
    total_input_tokens = 0
    total_output_tokens = 0
    iterations = 0
    start_time = datetime.now()

    # ─────────────────────────────────────────
    # REACT LOOP
    # Each iteration: model reasons → calls tool
    # → we execute → return result → model reasons again
    # Loop ends when model stops calling tools
    # and generates final text response
    # ─────────────────────────────────────────

    while iterations < max_iterations:
        iterations += 1

        # Call the model with current conversation
        # and all available tools
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            tools=AGENT_TOOLS,
            messages=messages
        )

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Check why the model stopped generating
        # stop_reason tells us what to do next
        if response.stop_reason == "end_turn":
            # Model finished — extract final text response
            final_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_response = block.text
                    break
            break

        elif response.stop_reason == "tool_use":
            # Model wants to call one or more tools
            # Add model's reasoning to conversation
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Process each tool call the model requested
            tool_results_for_message = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id

                    print(f"  → Agent calling: {tool_name}")
                    if tool_input:
                        for k, v in tool_input.items():
                            print(f"     {k}: {v}")

                    # Execute the tool
                    result = execute_tool(tool_name, tool_input)

                    tools_called.append(tool_name)
                    tool_results.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "result": json.loads(result)
                    })

                    # Collect tool result for the message
                    tool_results_for_message.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result
                    })

            # Return all tool results to the model
            # so it can continue reasoning
            messages.append({
                "role": "user",
                "content": tool_results_for_message
            })

        else:
            # Unexpected stop reason — break safely
            final_response = "I was unable to process your request. Please contact support."
            break

    latency_ms = int(
        (datetime.now() - start_time).total_seconds() * 1000
    )

    return {
        "response": final_response,
        "tools_called": tools_called,
        "tool_results": tool_results,
        "iterations": iterations,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "latency_ms": latency_ms
    }

# ─────────────────────────────────────────────
# TEST: Run agent on complex scenarios
# ─────────────────────────────────────────────

def test_agent():

    # Get real mobile numbers from database
    import sqlite3
    db_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'support.db'
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get one customer from each segment
    high_value = conn.execute("""
        SELECT c.mobile_number, c.name
        FROM customers c
        JOIN customer_segments cs ON c.mobile_number = cs.mobile_number
        WHERE cs.segment_label = 'high_value'
        LIMIT 1
    """).fetchone()

    standard = conn.execute("""
        SELECT c.mobile_number, c.name
        FROM customers c
        JOIN customer_segments cs ON c.mobile_number = cs.mobile_number
        WHERE cs.segment_label = 'standard'
        LIMIT 1
    """).fetchone()

    at_risk = conn.execute("""
        SELECT c.mobile_number, c.name
        FROM customers c
        JOIN customer_segments cs ON c.mobile_number = cs.mobile_number
        WHERE cs.segment_label = 'at_risk'
        LIMIT 1
    """).fetchone()

    conn.close()

    test_cases = []

    if high_value:
        test_cases.append({
            "name": high_value["name"],
            "mobile": high_value["mobile_number"],
            "query": "My food arrived completely cold again. "
                    "This keeps happening. I want a refund.",
            "label": "HIGH VALUE — Repeat cold food complaint"
        })

    if standard:
        test_cases.append({
            "name": standard["name"],
            "mobile": standard["mobile_number"],
            "query": "There are items missing from my order. "
                    "I ordered nuggets and fries but only got the burger.",
            "label": "STANDARD — Missing items"
        })

    if at_risk:
        test_cases.append({
            "name": at_risk["name"],
            "mobile": at_risk["mobile_number"],
            "query": "I complained last week about cold food and nothing "
                    "was done. Same thing happened again. "
                    "I want to speak to a manager right now.",
            "label": "AT RISK — Repeat complaint, escalation requested"
        })

    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {test['label']}")
        print(f"Customer: {test['name']} ({test['mobile']})")
        print(f"Query: {test['query']}")
        print(f"{'='*60}")
        print("Agent reasoning:")

        result = run_agent(
            customer_query=test["query"],
            mobile_number=test["mobile"]
        )

        print(f"\nFINAL RESPONSE:")
        print(result["response"])
        print(f"\nMETRICS:")
        print(f"  Tools called:    {result['tools_called']}")
        print(f"  Iterations:      {result['iterations']}")
        print(f"  Total tokens:    {result['total_tokens']}")
        print(f"  Latency:         {result['latency_ms']}ms")

if __name__ == "__main__":
    test_agent()