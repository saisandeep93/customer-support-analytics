import os
import json
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import anthropic

load_dotenv()

CHROMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')

# ─────────────────────────────────────────────
# INITIALISE CLIENTS
# ─────────────────────────────────────────────

def get_chroma_faq_collection():
    """
    Separate ChromaDB collection for FAQs.
    Different from knowledge_base collection in rag_pipeline.py
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="faqs",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def get_anthropic_client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────────
# FAQ DATA
# In production this would come from a database
# or CMS. For our project we define them here.
# ─────────────────────────────────────────────

FAQS = [
    {
        "faq_id": "FAQ001",
        "question": "What are your store hours?",
        "answer": "Our stores are open from 10am to 11pm daily, including weekends and public holidays. Some locations may have extended hours — check the app for your nearest store.",
        "category": "general"
    },
    {
        "faq_id": "FAQ002",
        "question": "Which areas do you deliver to?",
        "answer": "We deliver within a 5km radius of each store location. Enter your address in the app to check if delivery is available in your area.",
        "category": "delivery"
    },
    {
        "faq_id": "FAQ003",
        "question": "What payment methods do you accept?",
        "answer": "We accept UPI, credit cards, debit cards, net banking, and cash on delivery. Digital wallets including Paytm and PhonePe are also supported.",
        "category": "payment"
    },
    {
        "faq_id": "FAQ004",
        "question": "How do I download your app?",
        "answer": "Search for QuickBite on the Apple App Store or Google Play Store. The app is free to download. You can also order via our website at quickbite.com.",
        "category": "general"
    },
    {
        "faq_id": "FAQ005",
        "question": "How long does delivery take?",
        "answer": "Standard delivery takes 25 to 40 minutes depending on your location and store demand. You can track your order in real time through the app.",
        "category": "delivery"
    },
    {
        "faq_id": "FAQ006",
        "question": "Can I cancel my order after placing it?",
        "answer": "Orders can be cancelled within 2 minutes of placing them. After that, the order goes to preparation and cannot be cancelled. Contact support if you need help.",
        "category": "orders"
    },
    {
        "faq_id": "FAQ007",
        "question": "Do you have vegetarian options?",
        "answer": "Yes, we have a full vegetarian menu including veggie burgers, plant-based nuggets, and sides. All vegetarian items are marked with a green symbol in the app.",
        "category": "menu"
    },
    {
        "faq_id": "FAQ008",
        "question": "Is there a minimum order value for delivery?",
        "answer": "The minimum order value for delivery is Rs 149. Orders below this amount are available for pickup only.",
        "category": "delivery"
    },
    {
        "faq_id": "FAQ009",
        "question": "How do I apply a promo code or voucher?",
        "answer": "Enter your promo code in the cart screen before checkout. The discount will be applied automatically. One promo code can be used per order.",
        "category": "payment"
    },
    {
        "faq_id": "FAQ010",
        "question": "Do you have a loyalty programme?",
        "answer": "Yes, QuickBite Rewards lets you earn 1 point for every Rs 10 spent. Points can be redeemed for free items and discounts. Join through the app.",
        "category": "general"
    },
    {
        "faq_id": "FAQ011",
        "question": "What happens if my delivery is late?",
        "answer": "If your delivery exceeds the estimated time, you are eligible for compensation under our late delivery policy. Please contact support with your order ID.",
        "category": "delivery"
    },
    {
        "faq_id": "FAQ012",
        "question": "Can I change my delivery address after ordering?",
        "answer": "Delivery addresses cannot be changed after an order is placed. Please ensure your address is correct before confirming your order.",
        "category": "orders"
    },
    {
        "faq_id": "FAQ013",
        "question": "Do you cater for large group or bulk orders?",
        "answer": "Yes, we accept bulk orders for groups and events. Contact our catering team at catering@quickbite.com at least 24 hours in advance.",
        "category": "orders"
    },
    {
        "faq_id": "FAQ014",
        "question": "How do I track my order?",
        "answer": "Open the app and go to My Orders. Select your current order to see real-time tracking including preparation status and delivery partner location.",
        "category": "orders"
    },
    {
        "faq_id": "FAQ015",
        "question": "Are your ingredients allergen free?",
        "answer": "Our menu contains allergens including gluten, dairy, eggs, and nuts. Full allergen information is available on each product page in the app. Contact us if you have specific requirements.",
        "category": "menu"
    },
]

# ─────────────────────────────────────────────
# STEP 0: INDEX FAQs INTO CHROMADB
# Runs once — stores FAQ questions as vectors
# with answers in metadata
# ─────────────────────────────────────────────

def index_faqs():
    """
    Stores each FAQ question as a vector in ChromaDB.
    The answer lives in metadata — not embedded,
    just returned alongside the matching question.
    """
    print("Indexing FAQs...")
    collection = get_chroma_faq_collection()

    # Clear existing FAQs
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])

    ids = []
    questions = []
    metadatas = []

    for faq in FAQS:
        ids.append(faq['faq_id'])
        questions.append(faq['question'])
        metadatas.append({
            "answer": faq['answer'],
            "category": faq['category'],
            "faq_id": faq['faq_id']
        })

    collection.add(
        ids=ids,
        documents=questions,   # only questions get embedded
        metadatas=metadatas    # answers stored as metadata
    )

    print(f"  Indexed {len(FAQS)} FAQs into ChromaDB")

# ─────────────────────────────────────────────
# STEP 1: QUERY CLASSIFIER
# Uses Haiku to decide: SIMPLE or COMPLEX
# This is the router — the heart of Layer 3
# ─────────────────────────────────────────────

def classify_query(query: str) -> tuple:
    """
    Sends the query to Haiku with a classification prompt.
    Returns (classification, confidence, tokens_used)
    classification is either 'SIMPLE' or 'COMPLEX'
    """
    client = get_anthropic_client()

    classification_prompt = f"""You are a query classifier for a food delivery customer support system.

Classify the following customer query as either SIMPLE or COMPLEX.

SIMPLE queries:
- Ask for general information (hours, delivery areas, payment methods)
- Have the same answer for every customer
- Do not require looking up order details or customer history
- Examples: store hours, how to download app, menu questions, promo code help

COMPLEX queries:
- Involve a specific complaint or problem with an order
- Require checking order details, customer history, or account information
- Need judgment about compensation or resolution
- Examples: cold food, missing item, wrong order, late delivery complaint, payment dispute

Respond with exactly one word: SIMPLE or COMPLEX

Customer query: "{query}"

Classification:"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": classification_prompt}]
    )

    classification = message.content[0].text.strip().upper()
    tokens_used = message.usage.input_tokens + message.usage.output_tokens

    # Validate — default to COMPLEX if unexpected response
    if classification not in ['SIMPLE', 'COMPLEX']:
        classification = 'COMPLEX'

    return classification, tokens_used

# ─────────────────────────────────────────────
# STEP 2: FAQ RETRIEVAL
# For SIMPLE queries — find the best matching FAQ
# and return its answer directly
# ─────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.55

def retrieve_faq_answer(query: str) -> dict:
    """
    Searches the FAQ collection for the most similar question.
    Returns the answer if similarity is above threshold.
    Returns None if no good match found.
    """
    collection = get_chroma_faq_collection()

    results = collection.query(
        query_texts=[query],
        n_results=1,
        include=['documents', 'distances', 'metadatas']
    )

    if not results['ids'][0]:
        return None

    distance = results['distances'][0][0]
    similarity = round(1 - distance, 4)
    matched_question = results['documents'][0][0]
    metadata = results['metadatas'][0][0]

    if similarity < SIMILARITY_THRESHOLD:
        return None

    return {
        'faq_id': metadata['faq_id'],
        'matched_question': matched_question,
        'answer': metadata['answer'],
        'category': metadata['category'],
        'similarity': similarity
    }

# ─────────────────────────────────────────────
# COMPLETE DETERMINISTIC WORKFLOW
# Orchestrates classification → retrieval → response
# ─────────────────────────────────────────────

def run_deterministic_workflow(query: str) -> dict:
    """
    The complete Layer 3 pipeline.
    
    Returns a result dict with:
    - routed_to: 'deterministic' or 'rag_pipeline'
    - response: answer text if deterministic
    - metadata for logging
    """
    start_time = datetime.now()

    # Step 1: Classify the query
    classification, classifier_tokens = classify_query(query)

    if classification == 'COMPLEX':
        # Hand off to RAG pipeline — Layer 4 will handle this
        latency_ms = int(
            (datetime.now() - start_time).total_seconds() * 1000
        )
        return {
            'routed_to': 'rag_pipeline',
            'classification': 'COMPLEX',
            'response': None,
            'classifier_tokens': classifier_tokens,
            'latency_ms': latency_ms
        }

    # Step 2: Retrieve best matching FAQ
    faq_result = retrieve_faq_answer(query)

    latency_ms = int(
        (datetime.now() - start_time).total_seconds() * 1000
    )

    if faq_result is None:
        # SIMPLE but no FAQ match — escalate
        return {
            'routed_to': 'escalate',
            'classification': 'SIMPLE',
            'response': (
                "I don't have specific information on that. "
                "Please contact our support team directly "
                "or visit quickbite.com/help."
            ),
            'classifier_tokens': classifier_tokens,
            'faq_match': None,
            'latency_ms': latency_ms
        }

    # Success — return FAQ answer directly, no LLM call
    return {
        'routed_to': 'deterministic',
        'classification': 'SIMPLE',
        'response': faq_result['answer'],
        'faq_id': faq_result['faq_id'],
        'matched_question': faq_result['matched_question'],
        'similarity': faq_result['similarity'],
        'classifier_tokens': classifier_tokens,
        'latency_ms': latency_ms
    }

# ─────────────────────────────────────────────
# TEST: Run deterministic workflow
# ─────────────────────────────────────────────

def test_deterministic_workflow():

    # Index FAQs first
    index_faqs()

    test_queries = [
        # Should route to deterministic
        "What time do you open?",
        "How long will my delivery take?",
        "Can I pay with UPI?",
        "Do you have veg options?",
        "How do I use a promo code?",

        # Should route to RAG pipeline
        "My food arrived cold and I want a refund",
        "There are items missing from my order",
        "I was charged twice please help",

        # Edge case — simple but no FAQ match
        "Do you have gluten free buns?",
    ]

    print("\n" + "="*60)
    print("DETERMINISTIC WORKFLOW TEST")
    print("="*60)

    for query in test_queries:
        result = run_deterministic_workflow(query)
        print(f"\nQUERY: {query}")
        print(f"  Classification:  {result['classification']}")
        print(f"  Routed to:       {result['routed_to']}")
        print(f"  Classifier cost: {result['classifier_tokens']} tokens")
        print(f"  Latency:         {result['latency_ms']}ms")

        if result['routed_to'] == 'deterministic':
            print(f"  FAQ matched:     {result.get('matched_question')}")
            print(f"  Similarity:      {result.get('similarity')}")
            print(f"  Response:        {result['response'][:80]}...")
        elif result['routed_to'] == 'rag_pipeline':
            print(f"  → Handed off to RAG pipeline")
        else:
            print(f"  Response:        {result['response']}")

if __name__ == "__main__":
    test_deterministic_workflow()