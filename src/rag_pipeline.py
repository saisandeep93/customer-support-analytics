import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import anthropic

# Load API key from .env file
load_dotenv()

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'support.db')
CHROMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')

# ─────────────────────────────────────────────
# INITIALISE CLIENTS
# ─────────────────────────────────────────────

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn

def get_chroma_collection():
    """
    ChromaDB is our vector store.
    A 'collection' is like a table — it stores 
    text chunks alongside their vectors.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # We use a sentence transformer model to create embeddings.
    # 'all-MiniLM-L6-v2' is small, fast, and works well for 
    # semantic similarity tasks — perfect for our use case.
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_or_create_collection(
        name="knowledge_base",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity
    )
    return collection

def get_anthropic_client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────────
# STEP 0: INDEXING
# Convert knowledge base documents into vectors
# and store them in ChromaDB.
# This runs ONCE — not at every query.
# ─────────────────────────────────────────────

def index_knowledge_base():
    """
    Reads all documents from SQLite, chunks them,
    embeds each chunk, and stores in ChromaDB.
    """
    print("Indexing knowledge base...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    docs = cursor.execute('''
        SELECT doc_id, title, content, category 
        FROM knowledge_base_documents
    ''').fetchall()
    
    conn.close()
    
    collection = get_chroma_collection()
    
    # Clear existing vectors so we don't duplicate on re-run
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])
        print(f"  Cleared {len(existing['ids'])} existing chunks")
    
    chunk_ids = []
    chunk_texts = []
    chunk_metadata = []
    
    for doc in docs:
        doc_id = doc['doc_id']
        title = doc['title']
        content = doc['content']
        category = doc['category']
        
        # CHUNKING LOGIC
        # Split content into chunks of ~500 characters with 
        # 100 character overlap so boundary context is preserved
        chunk_size = 500
        overlap = 100
        chunks = []
        
        start = 0
        chunk_index = 0
        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]
            
            # Prepend title to every chunk so the vector 
            # captures the document topic even for short chunks
            full_chunk = f"{title}\n\n{chunk_text}"
            chunks.append((chunk_index, full_chunk))
            
            # Move forward by chunk_size minus overlap
            start += (chunk_size - overlap)
            chunk_index += 1
        
        for idx, chunk_text in chunks:
            chunk_id = f"{doc_id}_chunk_{idx}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk_text)
            chunk_metadata.append({
                "doc_id": doc_id,
                "title": title,
                "category": category,
                "chunk_index": idx
            })
    
    # Add all chunks to ChromaDB
    # ChromaDB automatically converts texts to vectors
    # using the embedding function we defined
    collection.add(
        ids=chunk_ids,
        documents=chunk_texts,
        metadatas=chunk_metadata
    )
    
    print(f"  Indexed {len(chunk_ids)} chunks from {len(docs)} documents")
    print(f"  Vector store saved to: {os.path.abspath(CHROMA_PATH)}")
    return len(chunk_ids)

# ─────────────────────────────────────────────
# STEP 1 + 2: RETRIEVAL
# Convert query to vector, search ChromaDB,
# return most relevant chunks
# ─────────────────────────────────────────────

def retrieve_relevant_chunks(query: str, n_results: int = 3) -> list:
    """
    Takes a customer query, finds the most semantically
    similar chunks in the knowledge base.
    Returns list of (chunk_text, similarity_score, metadata)
    """
    collection = get_chroma_collection()
    
    results = collection.query(
        query_texts=[query],  # ChromaDB embeds this automatically
        n_results=n_results,
        include=['documents', 'distances', 'metadatas']
    )
    
    chunks = []
    for i in range(len(results['ids'][0])):
        chunk_text = results['documents'][0][i]
        # ChromaDB returns distance (lower = more similar)
        # Convert to similarity score (higher = more similar)
        distance = results['distances'][0][i]
        similarity = round(1 - distance, 4)
        metadata = results['metadatas'][0][i]
        
        chunks.append({
            'text': chunk_text,
            'similarity': similarity,
            'doc_id': metadata['doc_id'],
            'title': metadata['title'],
            'category': metadata['category']
        })
    
    return chunks

# ─────────────────────────────────────────────
# STEP 3: PROMPT ASSEMBLY (AUGMENTATION)
# Combine retrieved chunks + customer context
# into a structured prompt for the LLM
# ─────────────────────────────────────────────

def assemble_prompt(
    customer_query: str,
    retrieved_chunks: list,
    customer_profile: dict
) -> str:
    """
    This is the augmentation step.
    We take the retrieved policy documents and the 
    customer's profile and build a structured prompt
    that gives the LLM everything it needs to respond.
    """
    
    # Format retrieved chunks into readable context
    context_sections = []
    for i, chunk in enumerate(retrieved_chunks):
        context_sections.append(
            f"POLICY DOCUMENT {i+1} — {chunk['title']} "
            f"(relevance: {chunk['similarity']}):\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_sections)
    
    # Format customer profile
    segment = customer_profile.get('segment_label', 'standard')
    name = customer_profile.get('name', 'Customer')
    total_orders = customer_profile.get('total_orders_lifetime', 0)
    complaint_count = customer_profile.get('complaint_count_last_10', 0)
    avg_value = customer_profile.get('avg_order_value', 0)
    
    prompt = f"""You are a customer support agent for QuickBite, a QSR food delivery brand.
Your job is to resolve customer complaints based on company policy.

CUSTOMER PROFILE:
- Name: {name}
- Segment: {segment.upper()}
- Lifetime orders: {total_orders}
- Complaints in last 10 orders: {complaint_count}
- Average order value: Rs {avg_value:.0f}

RELEVANT POLICY DOCUMENTS:
{context_text}

CUSTOMER MESSAGE:
"{customer_query}"

INSTRUCTIONS:
1. Read the policy documents carefully
2. Apply the resolution that matches the customer's segment
3. Be empathetic and professional
4. State clearly what action you are taking (refund/replacement/voucher)
5. Do not make up policies not present in the documents above
6. Keep your response concise — 3 to 4 sentences maximum

YOUR RESPONSE:"""
    
    return prompt

# ─────────────────────────────────────────────
# STEP 4: LLM GENERATION
# Send assembled prompt to Claude,
# get grounded response
# ─────────────────────────────────────────────

def generate_response(prompt: str) -> tuple:
    """
    Sends the assembled prompt to Claude.
    Returns (response_text, input_tokens, output_tokens)
    """
    client = get_anthropic_client()
    
    start_time = datetime.now()
    
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    end_time = datetime.now()
    latency_ms = int((end_time - start_time).total_seconds() * 1000)
    
    response_text = message.content[0].text
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    
    return response_text, input_tokens, output_tokens, latency_ms

# ─────────────────────────────────────────────
# COMPLETE RAG PIPELINE
# Orchestrates all steps end to end
# ─────────────────────────────────────────────

def run_rag_pipeline(
    customer_query: str,
    customer_profile: dict,
    n_chunks: int = 3
) -> dict:
    """
    The complete RAG pipeline.
    Input: customer query + customer profile
    Output: grounded response + full metadata for logging
    """
    
    # Step 1 + 2: Retrieve relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(
        customer_query, 
        n_results=n_chunks
    )
    
    # Step 3: Assemble prompt
    prompt = assemble_prompt(
        customer_query,
        retrieved_chunks,
        customer_profile
    )
    
    # Step 4: Generate response
    response_text, input_tokens, output_tokens, latency_ms = generate_response(prompt)
    
    # Return everything — response + metadata for Layer 6 logging
    return {
        'response': response_text,
        'retrieved_chunks': retrieved_chunks,
        'retrieved_doc_ids': [c['doc_id'] for c in retrieved_chunks],
        'retrieval_scores': [c['similarity'] for c in retrieved_chunks],
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
        'latency_ms': latency_ms,
        'prompt': prompt
    }

# ─────────────────────────────────────────────
# TEST: Run the full pipeline end to end
# ─────────────────────────────────────────────

def test_rag_pipeline():
    
    # Step 0: Index the knowledge base first
    index_knowledge_base()
    
    # Test customer profiles — one per segment
    test_cases = [
        {
            'query': "My food arrived completely cold. The burger was freezing.",
            'profile': {
                'name': 'Priya Sharma',
                'segment_label': 'high_value',
                'total_orders_lifetime': 45,
                'complaint_count_last_10': 0,
                'avg_order_value': 520
            }
        },
        {
            'query': "I ordered chicken nuggets but they are missing from my order.",
            'profile': {
                'name': 'Rahul Verma',
                'segment_label': 'standard',
                'total_orders_lifetime': 8,
                'complaint_count_last_10': 1,
                'avg_order_value': 210
            }
        },
        {
            'query': "I was charged twice for my order. Please refund.",
            'profile': {
                'name': 'Amit Kumar',
                'segment_label': 'at_risk',
                'total_orders_lifetime': 12,
                'complaint_count_last_10': 4,
                'avg_order_value': 180
            }
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}: {test['profile']['name']} "
              f"({test['profile']['segment_label'].upper()})")
        print(f"{'='*60}")
        print(f"QUERY: {test['query']}")
        print(f"\nRETRIEVED DOCUMENTS:")
        
        result = run_rag_pipeline(
            test['query'],
            test['profile']
        )
        
        for chunk in result['retrieved_chunks']:
            print(f"  - {chunk['title']} "
                  f"(similarity: {chunk['similarity']})")
        
        print(f"\nRESPONSE:")
        print(f"{result['response']}")
        
        print(f"\nMETRICS:")
        print(f"  Input tokens:  {result['input_tokens']}")
        print(f"  Output tokens: {result['output_tokens']}")
        print(f"  Total tokens:  {result['total_tokens']}")
        print(f"  Latency:       {result['latency_ms']}ms")

if __name__ == "__main__":
    test_rag_pipeline()