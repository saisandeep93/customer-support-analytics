import sqlite3
import os
import random
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'support.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def seed_products(cursor):
    products = [
        ('P001', 'Classic Burger', 'burger', 199.0, 'Beef patty with lettuce, tomato, and cheese'),
        ('P002', 'Spicy Chicken Burger', 'burger', 219.0, 'Crispy spicy chicken with coleslaw'),
        ('P003', 'Veggie Burger', 'burger', 179.0, 'Plant-based patty with fresh vegetables'),
        ('P004', 'French Fries - Regular', 'sides', 89.0, 'Crispy golden fries, regular size'),
        ('P005', 'French Fries - Large', 'sides', 119.0, 'Crispy golden fries, large size'),
        ('P006', 'Coleslaw', 'sides', 69.0, 'Creamy coleslaw with fresh cabbage'),
        ('P007', 'Chocolate Milkshake', 'beverages', 149.0, 'Thick chocolate milkshake'),
        ('P008', 'Coca Cola - Regular', 'beverages', 79.0, 'Chilled Coca Cola, regular size'),
        ('P009', 'Chicken Nuggets - 6pc', 'snacks', 159.0, 'Crispy chicken nuggets, 6 pieces'),
        ('P010', 'Family Meal Bundle', 'bundles', 599.0, '2 burgers, 2 fries, 2 drinks'),
    ]
    cursor.executemany('''
        INSERT OR IGNORE INTO products 
        (product_id, product_name, category, price, description)
        VALUES (?, ?, ?, ?, ?)
    ''', products)
    print(f"  Seeded {len(products)} products")

def seed_customers(cursor):
    names = [
        'Priya Sharma', 'Rahul Verma', 'Ananya Singh', 'Karthik Nair',
        'Divya Menon', 'Arjun Patel', 'Sneha Reddy', 'Vikram Iyer',
        'Pooja Gupta', 'Amit Kumar', 'Lakshmi Rao', 'Rohit Joshi',
        'Meera Pillai', 'Suresh Bhat', 'Kavya Krishnan', 'Nikhil Shah',
        'Anjali Desai', 'Ravi Shankar', 'Deepa Nambiar', 'Arun Mehta',
        'Shruti Saxena', 'Manoj Tiwari', 'Nisha Kapoor', 'Vijay Pandey',
        'Rekha Sinha', 'Ganesh Murthy', 'Sunita Yadav', 'Prasad Kulkarni',
        'Radha Venkat', 'Sanjay Mishra', 'Usha Rajan', 'Dinesh Choudhary',
        'Malathi Subramanian', 'Harish Gowda', 'Padma Swamy', 'Girish Hegde',
        'Vani Narayanan', 'Mohan Prakash', 'Savitha Rao', 'Rajesh Kamath',
        'Chitra Balaji', 'Naveen Kumar', 'Bhavana Reddy', 'Sunil Patil',
        'Latha Krishnamurthy', 'Vinod Shetty', 'Kamala Devi', 'Ashok Nayak',
        'Sudha Ramesh', 'Prakash Bhandari'
    ]
    
    channels = ['app', 'web']
    base_date = datetime(2023, 1, 1)
    mobiles = []

    for i, name in enumerate(names):
        mobile = f"9{''.join([str(random.randint(0,9)) for _ in range(9)])}"
        email = f"{name.split()[0].lower()}{i}@email.com"
        reg_date = base_date + timedelta(days=random.randint(0, 365))
        channel = random.choice(channels)
        
        cursor.execute('''
            INSERT OR IGNORE INTO customers 
            (mobile_number, name, email, registration_date, channel_preference)
            VALUES (?, ?, ?, ?, ?)
        ''', (mobile, name, email, reg_date.strftime('%Y-%m-%d'), channel))
        mobiles.append(mobile)
    
    print(f"  Seeded {len(names)} customers")
    return mobiles

def seed_orders_and_lineitems(cursor, mobiles):
    products = cursor.execute('SELECT product_id, price FROM products').fetchall()
    statuses = ['delivered', 'delivered', 'delivered', 'cancelled']
    channels = ['app', 'web']
    
    base_date = datetime(2024, 1, 1)
    order_ids = []
    order_count = 0
    lineitem_count = 0

    for mobile in mobiles:
        num_orders = random.randint(2, 8)
        for j in range(num_orders):
            order_id = f"ORD{str(len(order_ids)+1).zfill(4)}"
            order_date = base_date + timedelta(days=random.randint(0, 364))
            channel = random.choice(channels)
            status = random.choice(statuses)
            address = f"{random.randint(1,200)}, MG Road, Bengaluru"

            # Pick 1-3 random products for this order
            order_products = random.sample(products, random.randint(1, 3))
            total = sum(p[1] for p in order_products)

            cursor.execute('''
                INSERT OR IGNORE INTO orders
                (order_id, mobile_number, order_timestamp, channel, 
                 delivery_address, total_amount, order_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (order_id, mobile, order_date.strftime('%Y-%m-%d %H:%M:%S'),
                  channel, address, total, status))

            for product_id, price in order_products:
                qty = random.randint(1, 2)
                cursor.execute('''
                    INSERT INTO order_line_items
                    (order_id, product_id, quantity, unit_price)
                    VALUES (?, ?, ?, ?)
                ''', (order_id, product_id, qty, price))
                lineitem_count += 1

            order_ids.append((order_id, mobile))
            order_count += 1

    print(f"  Seeded {order_count} orders and {lineitem_count} line items")
    return order_ids

def seed_customer_segments(cursor, mobiles):
    for mobile in mobiles:
        total_orders = cursor.execute(
            'SELECT COUNT(*) FROM orders WHERE mobile_number = ?', (mobile,)
        ).fetchone()[0]
        
        avg_value = cursor.execute(
            'SELECT AVG(total_amount) FROM orders WHERE mobile_number = ?', (mobile,)
        ).fetchone()[0] or 0
        
        complaint_count = random.randint(0, 3)
        days_since_first = random.randint(30, 700)

        if total_orders >= 10 and avg_value >= 300 and complaint_count <= 1:
            segment = 'high_value'
        elif complaint_count >= 3:
            segment = 'at_risk'
        else:
            segment = 'standard'

        cursor.execute('''
            INSERT OR REPLACE INTO customer_segments
            (mobile_number, total_orders_lifetime, complaint_count_last_10,
             avg_order_value, days_since_first_order, segment_label)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (mobile, total_orders, complaint_count, round(avg_value, 2),
              days_since_first, segment))

    print(f"  Seeded customer segments for {len(mobiles)} customers")

def seed_support_tickets(cursor, order_ids):
    categories = [
        'cold_food', 'missing_item', 'wrong_item',
        'late_delivery', 'damaged_packaging', 'payment_issue'
    ]
    complaint_texts = {
        'cold_food': 'My food arrived completely cold. The burger was cold and fries were soggy.',
        'missing_item': 'I ordered 3 items but only received 2. My chicken nuggets are missing.',
        'wrong_item': 'I received the wrong order. I ordered a veggie burger but got a chicken burger.',
        'late_delivery': 'My order was supposed to arrive in 30 minutes but came after 90 minutes.',
        'damaged_packaging': 'The packaging was completely crushed. My burger was squashed.',
        'payment_issue': 'I was charged twice for my order. Please refund the extra charge.'
    }
    resolution_types = ['refund', 'replacement', 'voucher', 'no_action']
    
    # Only raise tickets for ~40% of delivered orders
    delivered_orders = [
        (oid, mob) for oid, mob in order_ids
        if cursor.execute(
            'SELECT order_status FROM orders WHERE order_id=?', (oid,)
        ).fetchone()[0] == 'delivered'
    ]
    
    ticket_orders = random.sample(
        delivered_orders, int(len(delivered_orders) * 0.4)
    )
    
    ticket_count = 0
    for order_id, mobile in ticket_orders:
        ticket_id = f"TKT{str(ticket_count+1).zfill(4)}"
        category = random.choice(categories)
        created_at = datetime(2024, 6, 1) + timedelta(days=random.randint(0, 180))
        resolved_at = created_at + timedelta(hours=random.randint(1, 48))
        resolution = random.choice(resolution_types)
        feedback = random.randint(1, 5)

        cursor.execute('''
            INSERT OR IGNORE INTO support_tickets
            (ticket_id, order_id, mobile_number, complaint_category,
             complaint_text, created_at, resolution_status, resolution_type,
             resolution_notes, customer_feedback_score, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (ticket_id, order_id, mobile, category,
              complaint_texts[category],
              created_at.strftime('%Y-%m-%d %H:%M:%S'),
              'resolved', resolution,
              f"Applied {resolution} as per customer segment policy.",
              feedback,
              resolved_at.strftime('%Y-%m-%d %H:%M:%S')))
        ticket_count += 1

    print(f"  Seeded {ticket_count} support tickets")

def seed_knowledge_base(cursor):
    docs = [
        (
            'KB001',
            'Cold Food - Refund Policy',
            '''When a customer reports cold food delivery, follow this resolution policy:
            
HIGH VALUE customers (segment: high_value): Issue immediate full refund with no questions asked. 
Send apology message. No investigation required.

STANDARD customers: Offer choice of replacement order or 50% refund voucher for next order. 
Ask for photo evidence if order value exceeds Rs 500.

AT RISK customers (repeat complainers): Escalate to human agent for manual review. 
Check complaint history before applying resolution. Maximum one refund per month.

Resolution timeframe: All cold food complaints must be resolved within 2 hours.''',
            'cold_food',
            'refund,replacement,voucher',
            datetime.now().strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        ),
        (
            'KB002',
            'Missing Item - Resolution Guide',
            '''When a customer reports a missing item from their order:

Step 1: Verify the order details and confirm which item is missing.
Step 2: Check if the item was marked as unavailable at time of order.

HIGH VALUE customers: Immediately redeliver the missing item or issue full item refund.
STANDARD customers: Issue refund for the missing item value only.
AT RISK customers: Verify claim against order records. Escalate if pattern detected.

Note: Missing item claims must be raised within 2 hours of delivery.
Evidence required: Order confirmation screenshot for claims above Rs 300.''',
            'missing_item',
            'refund,replacement',
            datetime.now().strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        ),
        (
            'KB003',
            'Wrong Item Delivered - Resolution Guide',
            '''When a customer receives the wrong item:

Immediate action: Apologise and acknowledge the error.

HIGH VALUE customers: Redeliver correct item within 30 minutes if possible, 
plus issue a Rs 100 goodwill voucher.
STANDARD customers: Redeliver correct item or issue full refund for wrong item.
AT RISK customers: Issue refund only. No redelivery. Escalate if customer insists.

Wrong item complaints must include photo evidence for orders above Rs 400.''',
            'wrong_item',
            'refund,replacement,voucher',
            datetime.now().strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        ),
        (
            'KB004',
            'Late Delivery - Compensation Policy',
            '''Late delivery is defined as delivery exceeding 45 minutes from confirmed order time.

Under 60 minutes late:
- Apologise and offer Rs 50 voucher for next order.

60-90 minutes late:
- Offer Rs 100 voucher or 20% refund on order value.

Over 90 minutes late:
- HIGH VALUE: Full refund plus Rs 150 goodwill voucher.
- STANDARD: 50% refund or full voucher equivalent.
- AT RISK: 30% refund. No additional goodwill gesture.

All late delivery claims must be verified against delivery timestamp in system.''',
            'late_delivery',
            'refund,voucher',
            datetime.now().strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        ),
        (
            'KB005',
            'Damaged Packaging - Resolution Guide',
            '''When a customer reports damaged packaging:

Assessment criteria:
- Minor damage (cosmetic only): Apologise, offer Rs 50 voucher.
- Moderate damage (food affected): Offer replacement or 50% refund.
- Severe damage (food inedible): Full refund or full replacement.

HIGH VALUE customers: Always treat as severe. Full refund or replacement, no questions.
STANDARD customers: Assess based on photo evidence.
AT RISK customers: Require photo evidence. Apply standard tier only.

Packaging complaints accepted within 1 hour of delivery only.''',
            'damaged_packaging',
            'refund,replacement,voucher',
            datetime.now().strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        ),
        (
            'KB006',
            'Payment Issues - Double Charge Resolution',
            '''When a customer reports a payment issue or double charge:

Step 1: Verify transaction records immediately.
Step 2: Confirm whether double charge occurred.

If double charge confirmed:
- Initiate refund within 24 hours regardless of customer segment.
- Refund timeline: 5-7 business days to original payment method.
- Send confirmation message with refund reference number.

If double charge not confirmed:
- Share transaction details with customer.
- Escalate to finance team if customer disputes.

Payment issues always get priority handling regardless of customer segment.
Never dismiss a payment complaint without verification.''',
            'payment_issue',
            'refund',
            datetime.now().strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        ),
        (
            'KB007',
            'Customer Segment Definitions',
            '''Customer segments determine resolution priority and compensation levels.

HIGH VALUE segment criteria:
- Total lifetime orders: 10 or more
- Average order value: Rs 300 or more  
- Complaint count in last 10 orders: 1 or fewer
- These customers receive premium resolution with no questions asked.

STANDARD segment criteria:
- Does not meet high value criteria
- Complaint count in last 10 orders: 2 or fewer
- These customers receive standard resolution per policy guidelines.

AT RISK segment criteria:
- Complaint count in last 10 orders: 3 or more
- Regardless of order history or value
- These customers require human review before resolution.
- Flag for potential abuse pattern investigation.

Segment labels are updated weekly based on rolling order and complaint data.''',
            'general',
            'general',
            datetime.now().strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        ),
    ]

    cursor.executemany('''
        INSERT OR IGNORE INTO knowledge_base_documents
        (doc_id, title, content, category, resolution_type_applicable,
         created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', docs)
    print(f"  Seeded {len(docs)} knowledge base documents")

def seed_all():
    conn = get_connection()
    cursor = conn.cursor()
    
    print("Seeding database...")
    seed_products(cursor)
    mobiles = seed_customers(cursor)
    order_ids = seed_orders_and_lineitems(cursor, mobiles)
    seed_customer_segments(cursor, mobiles)
    seed_support_tickets(cursor, order_ids)
    seed_knowledge_base(cursor)
    
    conn.commit()
    conn.close()
    print("\nDatabase seeded successfully.")

if __name__ == "__main__":
    seed_all()