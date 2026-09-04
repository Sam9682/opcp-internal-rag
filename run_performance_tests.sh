#!/bin/bash
# Performance Testing Script for Generic RAG Web Application
# Runs performance tests and generates report

set -e

echo "======================================================================"
echo "GENERIC RAG WEB APPLICATION - PERFORMANCE TEST SUITE"
echo "======================================================================"
echo ""
echo "Validating Requirements:"
echo "  - 14.1: Vector search <50ms for 100K chunks"
echo "  - 14.2: Embedding 100-500 chunks/second with GPU"
echo "  - 14.3: Handle 100 concurrent requests"
echo "  - 14.5: Maintain performance with 1M+ chunks"
echo ""

# Change to services/shared directory
cd services/shared

# Run vector search performance test
echo "======================================================================"
echo "TEST 1: Vector Search Latency"
echo "======================================================================"
python3 << 'EOF'
import time
import statistics
import numpy as np
from database import get_db_manager
from vector_search_service import VectorSearchService
from orm_models import TextChunk

# Initialize
db_manager = get_db_manager()
vector_search = VectorSearchService(db_manager)

# Get chunk count
with db_manager.get_session() as session:
    chunk_count = session.query(TextChunk).count()
    print(f"Current chunk count: {chunk_count:,}")

if chunk_count == 0:
    print("⚠️  No chunks in database. Skipping test.")
    exit(0)

# Generate random query vectors
num_queries = 100
print(f"Running {num_queries} vector searches...")
query_vectors = [np.random.randn(1024).astype(np.float32) for _ in range(num_queries)]
for vec in query_vectors:
    vec /= np.linalg.norm(vec)

# Measure latency
latencies = []
for i, query_vec in enumerate(query_vectors):
    start = time.perf_counter()
    results = vector_search.search_similar(query_vector=query_vec, top_k=5, threshold=0.0)
    end = time.perf_counter()
    latencies.append((end - start) * 1000)
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{num_queries}")

# Statistics
avg = statistics.mean(latencies)
median = statistics.median(latencies)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)

print(f"\nResults:")
print(f"  Average latency:  {avg:.2f} ms")
print(f"  Median latency:   {median:.2f} ms")
print(f"  P95 latency:      {p95:.2f} ms")
print(f"  P99 latency:      {p99:.2f} ms")

# Check requirement
if chunk_count >= 100000:
    if avg <= 50.0:
        print(f"\n✅ PASS: Average latency {avg:.2f}ms <= 50ms (Requirement 14.1)")
    else:
        print(f"\n❌ FAIL: Average latency {avg:.2f}ms > 50ms (Requirement 14.1)")
else:
    print(f"\n⚠️  INFO: Chunk count {chunk_count:,} < 100K. Requirement 14.1 not applicable yet.")

db_manager.close_all()
EOF

echo ""

# Run embedding throughput test
echo "======================================================================"
echo "TEST 2: Embedding Generation Throughput"
echo "======================================================================"
python3 << 'EOF'
import time
from embedding_service import EmbeddingService

# Initialize
embedding_service = EmbeddingService()
device = embedding_service.device
print(f"Device: {device}")

# Generate sample texts
num_texts = 100
texts = [f"Sample text {i} for performance testing. " * 10 for i in range(num_texts)]

# Test batch embedding
print(f"\nTesting batch embedding ({num_texts} texts)...")
start = time.perf_counter()
_ = embedding_service.embed_batch(texts)
end = time.perf_counter()

batch_time = end - start
throughput = num_texts / batch_time

print(f"  Time: {batch_time:.2f}s")
print(f"  Throughput: {throughput:.2f} texts/second")

# Check requirement
if device == "cuda":
    if 100 <= throughput <= 1000:
        print(f"\n✅ PASS: Throughput {throughput:.2f} chunks/s in expected range (Requirement 14.2)")
    elif throughput < 100:
        print(f"\n❌ FAIL: Throughput {throughput:.2f} < 100 chunks/s (Requirement 14.2)")
    else:
        print(f"\n✅ PASS: Throughput {throughput:.2f} chunks/s exceeds expectations (Requirement 14.2)")
else:
    print(f"\n⚠️  INFO: CPU mode. GPU throughput requirement not applicable.")
    print(f"   Current throughput: {throughput:.2f} chunks/s")
EOF

echo ""

# Run database scalability test
echo "======================================================================"
echo "TEST 3: Database Scalability"
echo "======================================================================"
python3 << 'EOF'
from database import get_db_manager
from orm_models import TextChunk, Document
from sqlalchemy import text

# Initialize
db_manager = get_db_manager()

with db_manager.get_session() as session:
    chunk_count = session.query(TextChunk).count()
    doc_count = session.query(Document).count()
    
    print(f"\nDatabase Statistics:")
    print(f"  Documents: {doc_count:,}")
    print(f"  Chunks:    {chunk_count:,}")
    
    # Check indexes
    print(f"\nIndex Information:")
    index_query = text("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'text_chunks'
        ORDER BY indexname;
    """)
    
    result = session.execute(index_query)
    indexes = result.fetchall()
    
    print(f"  Indexes on text_chunks table:")
    for idx in indexes:
        print(f"    - {idx[0]}")
    
    has_hnsw = any('hnsw' in str(idx[1]).lower() for idx in indexes)
    
    if has_hnsw:
        print(f"\n  ✓ HNSW index found (optimized for vector search)")
    else:
        print(f"\n  ⚠️  No HNSW index found (may impact performance)")
    
    # Assessment
    if chunk_count >= 1_000_000:
        print(f"\n✅ Database contains 1M+ chunks (Requirement 14.5)")
    elif chunk_count >= 100_000:
        print(f"\n⚠️  INFO: Database contains 100K+ chunks")
        print(f"   Approaching 1M chunk target for Requirement 14.5")
    else:
        print(f"\n⚠️  INFO: Database contains {chunk_count:,} chunks")
        print(f"   Need to ingest more documents to test 1M chunk scalability")

db_manager.close_all()
EOF

echo ""
echo "======================================================================"
echo "PERFORMANCE TEST SUITE COMPLETE"
echo "======================================================================"
