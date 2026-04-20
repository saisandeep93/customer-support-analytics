import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'support.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row  # returns rows as dictionaries
    return conn

# ─────────────────────────────────────────────
# QUERY 1: Get full customer profile by mobile number
# Used by agent when a customer contacts support
# ─────────────────────────────────────────────
def get_customer_profile(mobile_number: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    
    result = cursor.execute('''
        SELECT 
            c.mobile_number,
            c.name,
            c.email,
            c.registration_date,
            c.channel_preference,
            cs.total_orders_lifetime,
            cs.complaint_count_last_10,
            cs.avg_order_value,
            cs.days_since_first_order,
            cs.segment_label
        FROM customers c
        LEFT JOIN customer_segments cs 
            ON c.mobile_number = cs.mobile_number
        WHERE c.mobile_number = ?
    ''', (mobile_number,)).fetchone()
    
    conn.close()
    
    if result:
        return dict(result)
    return {}

# ─────────────────────────────────────────────
# QUERY 2: Get order details with line items
# Used by agent to understand what the customer ordered
# ─────────────────────────────────────────────
def get_order_details(order_id: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    
    order = cursor.execute('''
        SELECT 
            o.order_id,
            o.mobile_number,
            o.order_timestamp,
            o.channel,
            o.delivery_address,
            o.total_amount,
            o.order_status
        FROM orders o
        WHERE o.order_id = ?
    ''', (order_id,)).fetchone()
    
    if not order:
        conn.close()
        return {}
    
    line_items = cursor.execute('''
        SELECT 
            p.product_name,
            p.category,
            oli.quantity,
            oli.unit_price
        FROM order_line_items oli
        JOIN products p ON oli.product_id = p.product_id
        WHERE oli.order_id = ?
    ''', (order_id,)).fetchall()
    
    conn.close()
    
    return {
        **dict(order),
        'line_items': [dict(item) for item in line_items]
    }

# ─────────────────────────────────────────────
# QUERY 3: Get complaint history for a customer
# Used by agent to detect repeat complainers
# ─────────────────────────────────────────────
def get_complaint_history(mobile_number: str) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    
    tickets = cursor.execute('''
        SELECT 
            t.ticket_id,
            t.order_id,
            t.complaint_category,
            t.complaint_text,
            t.created_at,
            t.resolution_status,
            t.resolution_type,
            t.customer_feedback_score
        FROM support_tickets t
        WHERE t.mobile_number = ?
        ORDER BY t.created_at DESC
        LIMIT 10
    ''', (mobile_number,)).fetchall()
    
    conn.close()
    return [dict(t) for t in tickets]

# ─────────────────────────────────────────────
# QUERY 4: Check if an order already has a ticket
# Used by agent to avoid duplicate tickets
# ─────────────────────────────────────────────
def get_ticket_by_order(order_id: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    
    ticket = cursor.execute('''
        SELECT *
        FROM support_tickets
        WHERE order_id = ?
    ''', (order_id,)).fetchone()
    
    conn.close()
    return dict(ticket) if ticket else {}

# ─────────────────────────────────────────────
# QUERY 5: Get high value customers with open tickets
# Analytics query — used in dashboard
# ─────────────────────────────────────────────
def get_high_value_open_tickets() -> list:
    conn = get_connection()
    cursor = conn.cursor()
    
    results = cursor.execute('''
        SELECT 
            c.name,
            c.mobile_number,
            cs.segment_label,
            cs.total_orders_lifetime,
            cs.avg_order_value,
            t.ticket_id,
            t.complaint_category,
            t.created_at,
            t.resolution_status
        FROM support_tickets t
        JOIN customers c ON t.mobile_number = c.mobile_number
        JOIN customer_segments cs ON t.mobile_number = cs.mobile_number
        WHERE cs.segment_label = 'high_value'
        AND t.resolution_status IN ('open', 'in_progress')
        ORDER BY cs.avg_order_value DESC
    ''').fetchall()
    
    conn.close()
    return [dict(r) for r in results]

# ─────────────────────────────────────────────
# QUERY 6: Resolution summary by complaint category
# Analytics query — used in dashboard
# ─────────────────────────────────────────────
def get_resolution_summary() -> list:
    conn = get_connection()
    cursor = conn.cursor()
    
    results = cursor.execute('''
        SELECT 
            complaint_category,
            COUNT(*) as total_tickets,
            SUM(CASE WHEN resolution_status = 'resolved' THEN 1 ELSE 0 END) as resolved,
            SUM(CASE WHEN resolution_status = 'escalated' THEN 1 ELSE 0 END) as escalated,
            ROUND(AVG(customer_feedback_score), 2) as avg_feedback_score,
            ROUND(
                100.0 * SUM(CASE WHEN resolution_status = 'resolved' THEN 1 ELSE 0 END) 
                / COUNT(*), 1
            ) as resolution_rate_pct
        FROM support_tickets
        GROUP BY complaint_category
        ORDER BY total_tickets DESC
    ''').fetchall()
    
    conn.close()
    return [dict(r) for r in results]

# ─────────────────────────────────────────────
# TEST: Run all queries and print results
# ─────────────────────────────────────────────
def test_all_queries():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get a real mobile number and order ID from the database
    sample_customer = cursor.execute(
        'SELECT mobile_number FROM customers LIMIT 1'
    ).fetchone()['mobile_number']
    
    sample_order = cursor.execute(
        'SELECT order_id FROM orders LIMIT 1'
    ).fetchone()['order_id']
    
    conn.close()

    print("=" * 60)
    print("QUERY 1: Customer Profile")
    print("=" * 60)
    profile = get_customer_profile(sample_customer)
    for key, value in profile.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("QUERY 2: Order Details")
    print("=" * 60)
    order = get_order_details(sample_order)
    for key, value in order.items():
        if key != 'line_items':
            print(f"  {key}: {value}")
    print("  line_items:")
    for item in order.get('line_items', []):
        print(f"    - {item['product_name']} x{item['quantity']} @ Rs{item['unit_price']}")

    print("\n" + "=" * 60)
    print("QUERY 3: Complaint History")
    print("=" * 60)
    history = get_complaint_history(sample_customer)
    if history:
        for ticket in history:
            print(f"  {ticket['ticket_id']} | {ticket['complaint_category']} | {ticket['resolution_status']}")
    else:
        print("  No complaints found for this customer")

    print("\n" + "=" * 60)
    print("QUERY 5: High Value Customers with Open Tickets")
    print("=" * 60)
    high_value = get_high_value_open_tickets()
    if high_value:
        for row in high_value[:5]:
            print(f"  {row['name']} | {row['complaint_category']} | {row['resolution_status']}")
    else:
        print("  No high value open tickets found")

    print("\n" + "=" * 60)
    print("QUERY 6: Resolution Summary by Category")
    print("=" * 60)
    summary = get_resolution_summary()
    for row in summary:
        print(f"  {row['complaint_category']:20} | "
              f"Total: {row['total_tickets']:3} | "
              f"Resolved: {row['resolved']:3} | "
              f"Rate: {row['resolution_rate_pct']}% | "
              f"Avg Score: {row['avg_feedback_score']}")

def get_recent_orders(mobile_number: str, limit: int = 5) -> list:
    """Get most recent orders for a customer.
    Used by agent when customer doesn't provide order ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    orders = cursor.execute('''
        SELECT 
            o.order_id,
            o.order_timestamp,
            o.total_amount,
            o.order_status,
            COUNT(oli.line_item_id) as item_count
        FROM orders o
        LEFT JOIN order_line_items oli ON o.order_id = oli.order_id
        WHERE o.mobile_number = ?
        GROUP BY o.order_id
        ORDER BY o.order_timestamp DESC
        LIMIT ?
    ''', (mobile_number, limit)).fetchall()
    
    conn.close()
    return [dict(o) for o in orders]

if __name__ == "__main__":
    test_all_queries()