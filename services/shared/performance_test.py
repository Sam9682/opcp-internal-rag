#!/usr/bin/env python3
"""
Performance Testing Script for Generic RAG Web Application

Tests:
1. Query latency and throughput
2. Vector search performance with 100K+ chunks
3. Embedding generation performance
4. Concurrent request handling

Requirements validated:
- 14.1: Vector search on 100K chunks within 50ms
- 14.2: Embedding generation 100-500 chunks/second with GPU
- 14.3: Handle 100 concurrent requests
- 14.5: Maintain performance with 1M+ chunks
"""

import sys
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import numpy as np

from database import init_db, get_db_manager
from embedding_service import EmbeddingService
from vector_search_service import VectorSearchService
from text_preprocessor import TextPreprocessor
from orm_models import TextChunk, Document


class PerformanceTest:
    """Performance testing suite for RAG application"""
    
    def __init__(self):
        self.results = {}
        self.db_manager = None
        self.embedding_service = None
        self.vector_search = None
        self.preprocessor = None
        
    def setup(self):
        """Initialize services for testing"""
        print("Setting up services...")
        
        # Initialize database
        init_db()
        self.db_manager = get_db_manager()
        
        # Initialize services
        self.embedding_service = EmbeddingService()
        self.vector_search = VectorSearchService(self.db_manager)
        self.preprocessor = TextPreprocessor()
        
        print("✓ Services initialized")
    
    def teardown(self):
        """Cleanup after tests"""
        if self.db_manager:
            self.db_manager.close_all()
    
    def test_vector_search_latency(self, num_queries: int = 100) -> Dict:
        """
        Test vector search latency
        Requirement 14.1: <50ms for 100K chunks
        """
        print(f"\n{'='*70}")
        print("TEST 1: Vector Search Latency")
        print(f"{'='*70}")
        
        # Get current chunk count
        with self.db_manager.get_session() as session:
            chunk_count = session.query(TextChunk).count()
            print(f"Current chunk count: {chunk_count:,}")
        
        if chunk_count == 0:
            print("⚠️  No chunks in database. Skipping vector search test.")
            return {"status": "skipped", "reason": "no_chunks"}
        
        # Generate random query vectors
        print(f"Generating {num_queries} random query vectors...")
        query_vectors = [
            np.random.randn(1024).astype(np.float32) 
            for _ in range(num_queries)
        ]
        
        # Normalize vectors
        for vec in query_vectors:
            vec /= np.linalg.norm(vec)
        
        # Measure search latency
        print(f"Running {num_queries} vector searches...")
        latencies = []
        
        for i, query_vec in enumerate(query_vectors):
            start = time.perf_counter()
            results = self.vector_search.search_similar(
                query_vector=query_vec,
                top_k=5,
                threshold=0.0
            )
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_queries} queries")
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Print results
        print(f"\n{'Results:':<20}")
        print(f"  {'Chunk count:':<20} {chunk_count:,}")
        print(f"  {'Queries:':<20} {num_queries}")
        print(f"  {'Average latency:':<20} {avg_latency:.2f} ms")
        print(f"  {'Median latency:':<20} {median_latency:.2f} ms")
        print(f"  {'P95 latency:':<20} {p95_latency:.2f} ms")
        print(f"  {'P99 latency:':<20} {p99_latency:.2f} ms")
        print(f"  {'Min latency:':<20} {min_latency:.2f} ms")
        print(f"  {'Max latency:':<20} {max_latency:.2f} ms")
        
        # Check requirement
        target_latency = 50.0  # ms for 100K chunks
        if chunk_count >= 100000:
            if avg_latency <= target_latency:
                print(f"\n✅ PASS: Average latency {avg_latency:.2f}ms <= {target_latency}ms (Requirement 14.1)")
                status = "pass"
            else:
                print(f"\n❌ FAIL: Average latency {avg_latency:.2f}ms > {target_latency}ms (Requirement 14.1)")
                status = "fail"
        else:
            print(f"\n⚠️  INFO: Chunk count {chunk_count:,} < 100K. Requirement 14.1 not applicable yet.")
            status = "info"
        
        return {
            "status": status,
            "chunk_count": chunk_count,
            "num_queries": num_queries,
            "avg_latency_ms": avg_latency,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "target_latency_ms": target_latency
        }
    
    def test_embedding_throughput(self, num_texts: int = 100) -> Dict:
        """
        Test embedding generation throughput
        Requirement 14.2: 100-500 chunks/second with GPU
        """
        print(f"\n{'='*70}")
        print("TEST 2: Embedding Generation Throughput")
        print(f"{'='*70}")
        
        # Generate sample texts
        print(f"Generating {num_texts} sample texts...")
        texts = [
            f"This is sample text number {i} for embedding performance testing. " * 10
            for i in range(num_texts)
        ]
        
        # Test individual embedding
        print(f"\nTesting individual embedding ({num_texts} texts)...")
        start = time.perf_counter()
        for text in texts:
            _ = self.embedding_service.embed_text(text)
        end = time.perf_counter()
        
        individual_time = end - start
        individual_throughput = num_texts / individual_time
        
        print(f"  Time: {individual_time:.2f}s")
        print(f"  Throughput: {individual_throughput:.2f} texts/second")
        
        # Test batch embedding
        print(f"\nTesting batch embedding ({num_texts} texts)...")
        start = time.perf_counter()
        _ = self.embedding_service.embed_batch(texts)
        end = time.perf_counter()
        
        batch_time = end - start
        batch_throughput = num_texts / batch_time
        speedup = batch_throughput / individual_throughput
        
        print(f"  Time: {batch_time:.2f}s")
        print(f"  Throughput: {batch_throughput:.2f} texts/second")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Check device
        device = self.embedding_service.device
        print(f"\n  Device: {device}")
        
        # Check requirement
        min_throughput = 100  # chunks/second
        max_throughput = 500  # chunks/second (with GPU)
        
        if device == "cuda":
            if min_throughput <= batch_throughput <= max_throughput * 2:
                print(f"\n✅ PASS: Throughput {batch_throughput:.2f} chunks/s in expected range (Requirement 14.2)")
                status = "pass"
            elif batch_throughput < min_throughput:
                print(f"\n❌ FAIL: Throughput {batch_throughput:.2f} < {min_throughput} chunks/s (Requirement 14.2)")
                status = "fail"
            else:
                print(f"\n✅ PASS: Throughput {batch_throughput:.2f} chunks/s exceeds expectations (Requirement 14.2)")
                status = "pass"
        else:
            print(f"\n⚠️  INFO: CPU mode. GPU throughput requirement not applicable.")
            print(f"   Current throughput: {batch_throughput:.2f} chunks/s")
            status = "info"
        
        return {
            "status": status,
            "device": device,
            "num_texts": num_texts,
            "individual_time_s": individual_time,
            "individual_throughput": individual_throughput,
            "batch_time_s": batch_time,
            "batch_throughput": batch_throughput,
            "speedup": speedup,
            "min_throughput": min_throughput,
            "max_throughput": max_throughput
        }
    
    def test_concurrent_queries(self, num_concurrent: int = 50) -> Dict:
        """
        Test concurrent query handling
        Requirement 14.3: Handle 100 concurrent requests
        """
        print(f"\n{'='*70}")
        print("TEST 3: Concurrent Query Handling")
        print(f"{'='*70}")
        
        # Check if we have chunks
        with self.db_manager.get_session() as session:
            chunk_count = session.query(TextChunk).count()
        
        if chunk_count == 0:
            print("⚠️  No chunks in database. Skipping concurrent query test.")
            return {"status": "skipped", "reason": "no_chunks"}
        
        print(f"Testing with {num_concurrent} concurrent queries...")
        
        # Generate query vectors
        query_vectors = [
            np.random.randn(1024).astype(np.float32) / np.sqrt(1024)
            for _ in range(num_concurrent)
        ]
        
        # Function to execute single query
        def execute_query(query_vec):
            start = time.perf_counter()
            try:
                results = self.vector_search.search_similar(
                    query_vector=query_vec,
                    top_k=5,
                    threshold=0.0
                )
                end = time.perf_counter()
                return {
                    "success": True,
                    "latency": (end - start) * 1000,
                    "results": len(results)
                }
            except Exception as e:
                end = time.perf_counter()
                return {
                    "success": False,
                    "latency": (end - start) * 1000,
                    "error": str(e)
                }
        
        # Execute concurrent queries
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(execute_query, vec)
                for vec in query_vectors
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        latencies = [r["latency"] for r in successful]
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            p95_latency = np.percentile(latencies, 95)
            throughput = len(successful) / total_time
        else:
            avg_latency = median_latency = p95_latency = throughput = 0
        
        # Print results
        print(f"\n{'Results:':<25}")
        print(f"  {'Total time:':<25} {total_time:.2f}s")
        print(f"  {'Successful queries:':<25} {len(successful)}/{num_concurrent}")
        print(f"  {'Failed queries:':<25} {len(failed)}")
        print(f"  {'Throughput:':<25} {throughput:.2f} queries/second")
        
        if latencies:
            print(f"  {'Average latency:':<25} {avg_latency:.2f} ms")
            print(f"  {'Median latency:':<25} {median_latency:.2f} ms")
            print(f"  {'P95 latency:':<25} {p95_latency:.2f} ms")
        
        # Check requirement
        target_concurrent = 100
        success_rate = len(successful) / num_concurrent
        
        if num_concurrent >= target_concurrent:
            if success_rate >= 0.95:
                print(f"\n✅ PASS: {success_rate*100:.1f}% success rate with {num_concurrent} concurrent requests (Requirement 14.3)")
                status = "pass"
            else:
                print(f"\n❌ FAIL: {success_rate*100:.1f}% success rate < 95% (Requirement 14.3)")
                status = "fail"
        else:
            print(f"\n⚠️  INFO: Testing with {num_concurrent} < {target_concurrent} concurrent requests")
            print(f"   Success rate: {success_rate*100:.1f}%")
            status = "info"
        
        return {
            "status": status,
            "num_concurrent": num_concurrent,
            "total_time_s": total_time,
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": success_rate,
            "throughput": throughput,
            "avg_latency_ms": avg_latency if latencies else None,
            "median_latency_ms": median_latency if latencies else None,
            "p95_latency_ms": p95_latency if latencies else None,
            "target_concurrent": target_concurrent
        }
    
    def test_database_scalability(self) -> Dict:
        """
        Test database scalability and index performance
        Requirement 14.5: Maintain performance with 1M+ chunks
        """
        print(f"\n{'='*70}")
        print("TEST 4: Database Scalability")
        print(f"{'='*70}")
        
        with self.db_manager.get_session() as session:
            # Get statistics
            chunk_count = session.query(TextChunk).count()
            doc_count = session.query(Document).count()
            
            print(f"\n{'Database Statistics:':<25}")
            print(f"  {'Documents:':<25} {doc_count:,}")
            print(f"  {'Chunks:':<25} {chunk_count:,}")
            
            # Check index usage
            print(f"\n{'Index Information:':<25}")
            
            # Query for index information
            from sqlalchemy import text
            index_query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE tablename = 'text_chunks'
                ORDER BY indexname;
            """)
            
            result = session.execute(index_query)
            indexes = result.fetchall()
            
            print(f"  Indexes on text_chunks table:")
            for idx in indexes:
                print(f"    - {idx[2]}")
            
            # Check for HNSW index
            has_hnsw = any('hnsw' in str(idx[3]).lower() for idx in indexes)
            
            if has_hnsw:
                print(f"\n  ✓ HNSW index found (optimized for vector search)")
            else:
                print(f"\n  ⚠️  No HNSW index found (may impact performance)")
        
        # Performance assessment
        if chunk_count >= 1_000_000:
            print(f"\n✅ Database contains 1M+ chunks (Requirement 14.5)")
            print(f"   Run vector search test to verify performance maintained")
            status = "pass"
        elif chunk_count >= 100_000:
            print(f"\n⚠️  INFO: Database contains 100K+ chunks")
            print(f"   Approaching 1M chunk target for Requirement 14.5")
            status = "info"
        else:
            print(f"\n⚠️  INFO: Database contains {chunk_count:,} chunks")
            print(f"   Need to ingest more documents to test 1M chunk scalability")
            status = "info"
        
        return {
            "status": status,
            "chunk_count": chunk_count,
            "doc_count": doc_count,
            "has_hnsw_index": has_hnsw,
            "target_chunks": 1_000_000
        }
    
    def run_all_tests(self):
        """Run all performance tests"""
        print("="*70)
        print("GENERIC RAG WEB APPLICATION - PERFORMANCE TEST SUITE")
        print("="*70)
        print("\nValidating Requirements:")
        print("  - 14.1: Vector search <50ms for 100K chunks")
        print("  - 14.2: Embedding 100-500 chunks/second with GPU")
        print("  - 14.3: Handle 100 concurrent requests")
        print("  - 14.5: Maintain performance with 1M+ chunks")
        
        try:
            self.setup()
            
            # Run tests
            self.results["vector_search"] = self.test_vector_search_latency()
            self.results["embedding_throughput"] = self.test_embedding_throughput()
            self.results["concurrent_queries"] = self.test_concurrent_queries()
            self.results["database_scalability"] = self.test_database_scalability()
            
            # Print summary
            self.print_summary()
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            self.teardown()
        
        return 0
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")
        
        tests = [
            ("Vector Search Latency", "vector_search"),
            ("Embedding Throughput", "embedding_throughput"),
            ("Concurrent Queries", "concurrent_queries"),
            ("Database Scalability", "database_scalability")
        ]
        
        for name, key in tests:
            if key in self.results:
                result = self.results[key]
                status = result.get("status", "unknown")
                
                if status == "pass":
                    icon = "✅"
                elif status == "fail":
                    icon = "❌"
                elif status == "skipped":
                    icon = "⏭️ "
                else:
                    icon = "⚠️ "
                
                print(f"{icon} {name:<30} {status.upper()}")
        
        print(f"{'='*70}")
        
        # Overall assessment
        statuses = [r.get("status") for r in self.results.values()]
        
        if "fail" in statuses:
            print("\n❌ OVERALL: Some performance requirements not met")
            print("   Review failed tests and optimize as needed")
        elif "skipped" in statuses or "info" in statuses:
            print("\n⚠️  OVERALL: Performance tests partially complete")
            print("   Ingest more documents to enable full testing")
        else:
            print("\n✅ OVERALL: All performance requirements met")
            print("   System ready for production deployment")


def main():
    """Main entry point"""
    test = PerformanceTest()
    return test.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
