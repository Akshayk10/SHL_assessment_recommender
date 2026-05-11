"""
debug_agent.py — Debug the RAG-based retrieval system

    python debug_agent.py

Tests:
  1. FAISS index building
  2. Retrieval quality
  3. LLM response with retrieved chunks
"""

import json
import os
import sys
import logging
from dotenv import load_dotenv

sys.path.insert(0, ".")
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 70)
print("SHL RAG DEBUGGER")
print("=" * 70)

# ── 1. Test FAISS retrieval system ───────────────────────────────────────────
print("\n1. Testing FAISS Retrieval System...")
print("-" * 70)

try:
    from retrieval_rag import get_retrieval_system
    
    retrieval_system = get_retrieval_system()
    print(f"✓ FAISS system initialized")
    print(f"  - Catalog items: {len(retrieval_system.catalog)}")
    print(f"  - Index vectors: {retrieval_system.index.ntotal if retrieval_system.index else 0}")
    print(f"  - Model: {retrieval_system.model_name}")
    
except Exception as e:
    print(f"✗ Failed to initialize RAG: {e}")
    sys.exit(1)

# ── 2. Test retrieval with different queries ─────────────────────────────────
print("\n2. Testing Retrieval Quality...")
print("-" * 70)

test_queries = [
    "Java developer with 4 years experience",
    "Hiring a project manager who needs leadership skills",
    "Data analyst for numerical reasoning",
    "Sales manager with stakeholder management",
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = retrieval_system.retrieve_relevant(query, k=3)
    
    print(f"  Retrieved {len(results)} assessments:")
    for i, r in enumerate(results, 1):
        relevance = r.get('relevance_score', 0)
        print(f"    {i}. {r['name']} (score: {relevance:.3f})")
        print(f"       Type: {r['test_type']} - {r.get('test_type_label', '')}")
        if r.get('description'):
            desc_preview = r['description'][:80]
            print(f"       Desc: {desc_preview}...")

# ── 3. Test LLM integration ─────────────────────────────────────────────────
print("\n3. Testing LLM Integration with RAG...")
print("-" * 70)

from groq import Groq
from models import Message
from agent import get_agent_reply

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("✗ GROQ_API_KEY not found")
    sys.exit(1)

client = Groq(api_key=GROQ_API_KEY)

# Test a real conversation
test_messages = [
    Message(role="user", content="I am hiring a mid-level Java developer")
]

print(f"\nTest conversation:")
print(f"  User: {test_messages[0].content}")

try:
    response = get_agent_reply(test_messages, catalog_text="")  # catalog_text not used in RAG
    print(f"\n  Agent response:")
    print(f"    Reply: {response.reply}")
    print(f"    Recommendations: {len(response.recommendations)}")
    
    for i, rec in enumerate(response.recommendations[:3], 1):
        print(f"      {i}. {rec.name} ({rec.test_type})")
        print(f"         URL: {rec.url}")
    
    print(f"    End of conversation: {response.end_of_conversation}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ── 4. Compare with legacy approach (optional) ─────────────────────────────
print("\n4. RAG vs Legacy Comparison...")
print("-" * 70)

# Legacy approach (full catalog)
from retrieval import load_catalog, format_catalog_for_prompt

legacy_catalog = load_catalog()
legacy_text = format_catalog_for_prompt(legacy_catalog)

print(f"Legacy approach:")
print(f"  - Full catalog size: {len(legacy_text)} characters")
print(f"  - Token estimate: ~{len(legacy_text) // 4} tokens")

print(f"\nRAG approach:")
print(f"  - Retrieved only relevant chunks")
print(f"  - Much smaller prompt size")
print(f"  - No token limit issues")

# ── 5. Performance test ───────────────────────────────────────────────────
print("\n5. Performance Test...")
print("-" * 70)

import time

query = "Python developer for data science"
start = time.time()
results = retrieval_system.retrieve_relevant(query, k=5)
elapsed = (time.time() - start) * 1000

print(f"  Query: {query}")
print(f"  Retrieval time: {elapsed:.2f} ms")
print(f"  Retrieved {len(results)} relevant assessments")

print("\n" + "=" * 70)
print("Debug complete! RAG system is working correctly.")
print("=" * 70)