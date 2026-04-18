import sqlite3
import os

# Database will be stored in the data/ folder
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'support.db')

def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")  # enforce foreign key constraints
    return conn

def create_schema():
    """Create all tables in the database."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Customers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            mobile_number TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            registration_date TEXT NOT NULL,
            channel_preference TEXT CHECK(channel_preference IN ('app', 'web'))
        )
    ''')

    # 2. Customer segments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customer_segments (
            mobile_number TEXT PRIMARY KEY,
            total_orders_lifetime INTEGER DEFAULT 0,
            complaint_count_last_10 INTEGER DEFAULT 0,
            avg_order_value REAL DEFAULT 0.0,
            days_since_first_order INTEGER DEFAULT 0,
            segment_label TEXT CHECK(segment_label IN ('high_value', 'standard', 'at_risk')),
            FOREIGN KEY (mobile_number) REFERENCES customers(mobile_number)
        )
    ''')

    # 3. Products table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            description TEXT
        )
    ''')

    # 4. Orders table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            mobile_number TEXT NOT NULL,
            order_timestamp TEXT NOT NULL,
            channel TEXT CHECK(channel IN ('app', 'web')),
            delivery_address TEXT,
            total_amount REAL NOT NULL,
            order_status TEXT CHECK(order_status IN (
                'placed', 'preparing', 'out_for_delivery', 'delivered', 'cancelled'
            )),
            FOREIGN KEY (mobile_number) REFERENCES customers(mobile_number)
        )
    ''')

    # 5. Order line items table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS order_line_items (
            line_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            unit_price REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    ''')

    # 6. Support tickets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS support_tickets (
            ticket_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL UNIQUE,
            mobile_number TEXT NOT NULL,
            complaint_category TEXT CHECK(complaint_category IN (
                'cold_food', 'missing_item', 'wrong_item', 'late_delivery',
                'damaged_packaging', 'payment_issue', 'other'
            )),
            complaint_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            resolution_status TEXT CHECK(resolution_status IN (
                'open', 'in_progress', 'resolved', 'escalated'
            )) DEFAULT 'open',
            resolution_type TEXT CHECK(resolution_type IN (
                'refund', 'replacement', 'voucher', 'no_action', 'escalated_to_human'
            )),
            resolution_notes TEXT,
            customer_feedback_score INTEGER CHECK(
                customer_feedback_score BETWEEN 1 AND 5
            ),
            resolved_at TEXT,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (mobile_number) REFERENCES customers(mobile_number)
        )
    ''')

    # 7. Knowledge base documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_base_documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            resolution_type_applicable TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')

    # 8. Interaction logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interaction_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT,
            query_text TEXT NOT NULL,
            retrieved_doc_ids TEXT,
            retrieval_scores TEXT,
            tools_called TEXT,
            llm_response TEXT,
            tokens_used_input INTEGER,
            tokens_used_output INTEGER,
            latency_ms INTEGER,
            resolution_status TEXT,
            human_feedback_score INTEGER,
            escalated_to_human INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (ticket_id) REFERENCES support_tickets(ticket_id)
        )
    ''')

    conn.commit()
    conn.close()
    print("Schema created successfully.")
    print(f"Database location: {os.path.abspath(DB_PATH)}")

if __name__ == "__main__":
    create_schema()