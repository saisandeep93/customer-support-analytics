import os
import sqlite3
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_agent

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'support.db')
CHROMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_langchain')

# ─────────────────────────────────────────────
# VECTORSTORE — loaded once at module level
# so every search_knowledge_base call reuses it
# ─────────────────────────────────────────────

_vectorstore: Chroma | None = None

def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            collection_name="knowledge_base_lc",
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vectorstore

# ─────────────────────────────────────────────
# TOOL DEFINITIONS
# @tool turns a plain function into a LangChain
# tool; the docstring becomes the description
# the model uses to decide when to call it.
# ─────────────────────────────────────────────

@tool
def get_customer_profile(mobile_number: str) -> str:
    """Retrieves the full customer profile including their segment label
    (high_value, standard, or at_risk), lifetime order count, average order
    value, and complaint count in last 10 orders.
    ALWAYS call this first for any complaint — you need to know the customer
    segment before deciding on resolution. The segment determines which policy
    clause applies."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("""
        SELECT c.mobile_number, c.name, c.email, c.registration_date,
               c.channel_preference,
               cs.total_orders_lifetime, cs.complaint_count_last_10,
               cs.avg_order_value, cs.days_since_first_order, cs.segment_label
        FROM customers c
        LEFT JOIN customer_segments cs ON c.mobile_number = cs.mobile_number
        WHERE c.mobile_number = ?
    """, (mobile_number,)).fetchone()
    conn.close()
    if not row:
        return json.dumps({"error": "Customer not found"})
    return json.dumps(dict(row))


@tool
def get_order_details(order_id: str) -> str:
    """Retrieves full details of a specific order including all items ordered,
    quantities, prices, order status, delivery address, and order timestamp.
    Call this when the customer references a specific order or when you need
    to verify what was ordered."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    order = conn.execute("""
        SELECT o.order_id, o.mobile_number, o.order_timestamp, o.channel,
               o.delivery_address, o.total_amount, o.order_status
        FROM orders o WHERE o.order_id = ?
    """, (order_id,)).fetchone()
    if not order:
        conn.close()
        return json.dumps({"error": "Order not found"})
    items = conn.execute("""
        SELECT p.product_name, p.category, oli.quantity, oli.unit_price
        FROM order_line_items oli
        JOIN products p ON oli.product_id = p.product_id
        WHERE oli.order_id = ?
    """, (order_id,)).fetchall()
    conn.close()
    return json.dumps({**dict(order), "line_items": [dict(i) for i in items]})


@tool
def get_recent_orders(mobile_number: str) -> str:
    """Retrieves the 5 most recent orders for a customer with order ID,
    timestamp, total amount, status, and item count.
    Call this when the customer does not provide an order ID but you need
    to identify which order they are referring to."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT o.order_id, o.order_timestamp, o.total_amount, o.order_status,
               COUNT(oli.line_item_id) as item_count
        FROM orders o
        LEFT JOIN order_line_items oli ON o.order_id = oli.order_id
        WHERE o.mobile_number = ?
        GROUP BY o.order_id
        ORDER BY o.order_timestamp DESC
        LIMIT 5
    """, (mobile_number,)).fetchall()
    conn.close()
    return json.dumps({"recent_orders": [dict(r) for r in rows]})


@tool
def get_complaint_history(mobile_number: str) -> str:
    """Retrieves the last 10 support tickets for a customer, showing complaint
    categories, resolutions applied, and feedback scores.
    Call this when: customer mentions a previous complaint, customer seems
    frustrated suggesting history, or you need context on repeat issues.
    Essential for escalation decisions."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT ticket_id, order_id, complaint_category, complaint_text,
               created_at, resolution_status, resolution_type,
               customer_feedback_score
        FROM support_tickets
        WHERE mobile_number = ?
        ORDER BY created_at DESC
        LIMIT 10
    """, (mobile_number,)).fetchall()
    conn.close()
    complaints = [dict(r) for r in rows]
    return json.dumps({"complaint_count": len(complaints), "complaints": complaints})


@tool
def search_knowledge_base(query: str) -> str:
    """Searches the policy knowledge base for relevant resolution guidelines.
    Returns the most semantically similar policy documents to your query.
    Call this to find the correct resolution policy for any complaint type.
    Use descriptive queries like 'cold food refund policy high value customer'
    or 'repeat complaint escalation procedure'."""
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=2)
    policies = []
    for doc, distance in results:
        policies.append({
            "title": doc.metadata["title"],
            "content": doc.page_content,
            "similarity": round(1 - distance, 4),
        })
    return json.dumps({"chunks_retrieved": len(policies), "policies": policies})


TOOLS = [
    get_customer_profile,
    get_order_details,
    get_recent_orders,
    get_complaint_history,
    search_knowledge_base,
]

# ─────────────────────────────────────────────
# AGENT RUNNER
# create_react_agent builds the ReAct graph.
# We recreate it per call so the system prompt
# can embed the customer's mobile number, exactly
# mirroring how agent.py injects it.
# ─────────────────────────────────────────────

def run_agent(customer_query: str, mobile_number: str) -> dict:

    system_prompt = f"""You are an intelligent customer support agent for QuickBite, \
a QSR food delivery brand. You have access to tools that let you look up \
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

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    agent = create_agent(llm, TOOLS, system_prompt=system_prompt)

    start = datetime.now()
    state = agent.invoke({"messages": [HumanMessage(content=customer_query)]})
    latency_ms = int((datetime.now() - start).total_seconds() * 1000)

    messages = state["messages"]

    tools_called: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    final_response = ""

    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue

        # Accumulate token usage
        if msg.usage_metadata:
            total_input_tokens += msg.usage_metadata.get("input_tokens", 0)
            total_output_tokens += msg.usage_metadata.get("output_tokens", 0)

        # Collect tool calls made by this message
        for tc in msg.tool_calls or []:
            tools_called.append(tc["name"])
            print(f"  → Agent calling: {tc['name']}")
            for k, v in tc.get("args", {}).items():
                print(f"     {k}: {v}")

        # The last AIMessage with text content is the final answer
        if isinstance(msg.content, str) and msg.content:
            final_response = msg.content
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    final_response = block["text"]

    return {
        "response": final_response,
        "tools_called": tools_called,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "latency_ms": latency_ms,
    }

# ─────────────────────────────────────────────
# TEST: standard customer (missing items) +
#       at_risk customer (escalation request)
# ─────────────────────────────────────────────

def test_agent():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

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

    if standard:
        test_cases.append({
            "name": standard["name"],
            "mobile": standard["mobile_number"],
            "query": (
                "There are items missing from my order. "
                "I ordered nuggets and fries but only got the burger."
            ),
            "label": "STANDARD — Missing items",
        })

    if at_risk:
        test_cases.append({
            "name": at_risk["name"],
            "mobile": at_risk["mobile_number"],
            "query": (
                "I complained last week about cold food and nothing "
                "was done. Same thing happened again. "
                "I want to speak to a manager right now."
            ),
            "label": "AT RISK — Repeat complaint, escalation requested",
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
            mobile_number=test["mobile"],
        )

        print(f"\nFINAL RESPONSE:")
        print(result["response"])
        print(f"\nMETRICS:")
        print(f"  Tools called:    {result['tools_called']}")
        print(f"  Input tokens:    {result['total_input_tokens']}")
        print(f"  Output tokens:   {result['total_output_tokens']}")
        print(f"  Total tokens:    {result['total_tokens']}")
        print(f"  Latency:         {result['latency_ms']}ms")


if __name__ == "__main__":
    test_agent()
