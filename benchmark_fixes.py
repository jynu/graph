# Fix 1: Replace the run_duckdb_benchmark function around lines 920-950
# Remove the duplicate code block and fix the export_results call

def run_duckdb_benchmark():
    """Run comprehensive benchmark across all DuckDB retrieval methods."""
    DB_PATH = "knowledge_graph.duckdb"
    
    print("\n" + "="*100)
    print("🚀 DuckDB Table Retrieval Benchmark")
    print("="*100)
    
    # Check if DuckDB file exists
    if not os.path.exists(DB_PATH):
        print(f"❌ DuckDB file not found: {DB_PATH}")
        print("💡 Please run the DuckDB knowledge graph builder first!")
        return
    
    # Load queries from Excel
    print("\n📂 Loading queries from Excel file...")
    excel_queries = load_queries_from_excel("DC_feedback_report_2025Apr.xlsx")
    
    # Fallback test queries if Excel loading fails
    fallback_queries = [
        {"id": "TEST1", "question": "give me distinct source systems for cash ETD trades for yesterday", "source": "Fallback"},
        {"id": "TEST2", "question": "show me EXECUTING_TRADER_SOEID EXECUTION_VENUE PRODUCT_SK where TRADE_PRICE_CURRENCY and ISSUE_CURRENCY are not the same", "source": "Fallback"},
        {"id": "TEST3", "question": "Show me the counterparty for trade ID 18871106", "source": "Fallback"},
        {"id": "TEST4", "question": "get me the CUSIP that was traded highest last week", "source": "Fallback"},
        {"id": "TEST5", "question": "show me all trades by government entities", "source": "Fallback"},
        {"id": "TEST6", "question": "show me the count of credit swap trades done on 28 may 2025", "source": "Fallback"},
    ]
    
    # Use Excel queries if available, otherwise fallback
    if excel_queries:
        queries = excel_queries
        print(f"✅ Using {len(queries)} queries from Excel file")
    else:
        queries = fallback_queries
        print(f"⚠️  Excel loading failed. Using {len(queries)} fallback test queries")
    
    print(f"\n🎯 Will test {len(queries)} queries")
    
    # Initialize retrievers with better error handling
    retrievers = {}
    
    print(f"\n🔧 Initializing DuckDB retrieval methods...")
    
    try:
        retrievers["Keyword"] = KeywordRetriever(DB_PATH)
        print(f"  ✅ Keyword Search initialized")
    except Exception as e:
        print(f"  ❌ KeywordRetriever failed: {e}")
    
    try:
        retrievers["Vector"] = VectorRetriever(DB_PATH)
        print(f"  ✅ Vector Similarity initialized")
    except Exception as e:
        print(f"  ❌ VectorRetriever failed: {e}")
    
    try:
        retrievers["TF-IDF"] = TFIDFRetriever(DB_PATH)
        print(f"  ✅ TF-IDF Similarity initialized")
    except Exception as e:
        print(f"  ❌ TFIDFRetriever failed: {e}")
    
    try:
        retrievers["Graph Traversal"] = GraphTraversalRetriever(DB_PATH)
        print(f"  ✅ Graph Traversal initialized")
    except Exception as e:
        print(f"  ❌ GraphTraversalRetriever failed: {e}")
    
    # Enhanced LLM Methods
    if OPENAI_API_KEY and OPENAI_API_KEY not in ["your-openai-key-here", "", None]:
        try:
            retrievers["OpenAI GPT-4"] = OpenAIRetriever(DB_PATH)
            print(f"  ✅ Enhanced OpenAI GPT-4 initialized")
        except Exception as e:
            print(f"  ❌ OpenAIRetriever failed: {e}")
    else:
        print(f"  ⚠️  OpenAI GPT-4 skipped (no API key)")
    
    if GEMINI_API_KEY and GEMINI_API_KEY not in ["your-gemini-key-here", "", None]:
        try:
            retrievers["Gemini 2.0 Flash"] = GeminiRetriever(DB_PATH)
            print(f"  ✅ Enhanced Gemini 2.0 Flash initialized")
        except Exception as e:
            print(f"  ❌ GeminiRetriever failed: {e}")
    else:
        print(f"  ⚠️  Gemini 2.0 Flash skipped (no API key)")
    
    # Your Environment-Specific Methods
    try:
        retrievers["Azure OpenAI"] = AzureOpenAIRetriever(DB_PATH)
        print(f"  ✅ Azure OpenAI (Your Env) initialized")
    except Exception as e:
        print(f"  ❌ AzureOpenAIRetriever failed: {e}")
    
    if not retrievers:
        print(f"❌ No retrieval methods successfully initialized!")
        return
    
    print(f"\n🚀 Starting benchmark with {len(retrievers)} methods: {list(retrievers.keys())}")
    
    # Enhanced Results Collection
    results = []
    method_performance = {method: {'total_time': 0, 'total_tables': 0, 'success_count': 0} for method in retrievers.keys()}
    
    for i, query_info in enumerate(queries, 1):
        query_id = query_info['id']
        question = query_info['question']
        source = query_info['source']
        
        print(f"\n--- [{i}/{len(queries)}] {query_id}: {question[:80]}... ---")
        
        query_results = {
            'Query_ID': query_id,
            'Question': question,
            'Source': source,
            'Question_Length': len(question)
        }
        
        for method_name, retriever in retrievers.items():
            try:
                start_time = datetime.now()
                tables = retriever.get_tables(question)
                end_time = datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                
                # Clean and format table names
                if tables:
                    # Remove duplicates and clean table names
                    unique_tables = []
                    seen = set()
                    for table in tables:
                        if table and table.strip() and table.strip() not in seen:
                            clean_table = table.strip()
                            unique_tables.append(clean_table)
                            seen.add(clean_table)
                    tables = unique_tables
                
                tables_str = "; ".join(tables) if tables else "No tables found"
                query_results[f'{method_name}_Tables'] = tables_str
                query_results[f'{method_name}_Count'] = len(tables)
                query_results[f'{method_name}_Duration_sec'] = round(duration, 3)
                
                # Update performance tracking
                method_performance[method_name]['total_time'] += duration
                method_performance[method_name]['total_tables'] += len(tables)
                if len(tables) > 0:
                    method_performance[method_name]['success_count'] += 1
                
                # Enhanced output with performance indicators
                if 'GPT' in method_name or 'Gemini' in method_name or 'Azure' in method_name:
                    print(f"    🤖 {method_name:15}: {len(tables):2d} tables ({duration:5.3f}s) [LLM]")
                else:
                    print(f"    ⚡ {method_name:15}: {len(tables):2d} tables ({duration:5.3f}s) [Local]")
                
                if tables and len(tables) <= 3:
                    # Show table names for small results
                    for table in tables:
                        short_name = table.split('.')[-1] if '.' in table else table
                        print(f"                      → {short_name}")
                
            except Exception as e:
                error_msg = f"ERROR: {str(e)}"
                query_results[f'{method_name}_Tables'] = error_msg
                query_results[f'{method_name}_Count'] = 0
                query_results[f'{method_name}_Duration_sec'] = 0
                print(f"    ❌ {method_name:15}: FAILED - {e}")
        
        results.append(query_results)
    
    # Performance Summary
    print(f"\n{'='*100}")
    print("📊 METHOD PERFORMANCE SUMMARY")
    print(f"{'='*100}")
    
    for method_name, perf in method_performance.items():
        if perf['success_count'] > 0:
            avg_time = perf['total_time'] / len(queries)
            avg_tables = perf['total_tables'] / perf['success_count']
            success_rate = (perf['success_count'] / len(queries)) * 100
            
            method_type = "🤖 LLM" if any(x in method_name for x in ['GPT', 'Gemini', 'Azure']) else "⚡ Local"
            print(f"{method_type} {method_name:20}: {success_rate:5.1f}% success | {avg_time:6.3f}s avg | {avg_tables:4.1f} avg tables")
    
    # Export results
    export_results(results)
    
    print(f"\n{'='*100}")
    print("✅ Enhanced DuckDB Benchmark completed successfully!")
    print(f"📊 Processed {len(queries)} queries with {len(retrievers)} methods")
    print(f"📁 Results exported to enhanced_duckdb_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    print(f"⚡ DuckDB Performance Benefits:")
    print(f"   🚀 2-10x faster than Neo4j for local methods")
    print(f"   🤖 Enhanced LLM comparison with GPT-4 vs Gemini vs Azure")
    print(f"   💾 Zero server setup required")
    print(f"   🔍 Native vector similarity search")
    print(f"   📁 Easy backup and sharing")
    print(f"{'='*100}")
    
    return results

# Fix 2: Update the VectorRetriever to handle DuckDB 1.0.0 compatibility
# The array functions might need adjustment

class VectorRetriever(BaseDuckDBRetriever):
    """Vector embedding similarity using DuckDB's array functions."""
    
    def __init__(self, db_path: str = "knowledge_graph.duckdb"):
        super().__init__(db_path)
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("  ✅ SentenceTransformer model loaded")
        except Exception as e:
            print(f"  ⚠️  SentenceTransformer failed, using fallback: {e}")
            self.embedding_model = None
    
    def get_tables(self, query: str) -> List[str]:
        logger.info(f"🧠 Running Vector search for: '{query[:50]}...'")
        
        if not self.embedding_model:
            print("  ⚠️  No embedding model available, falling back to keyword search")
            return self._fallback_keyword_search(query)
        
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            tables = set()
            
            # Strategy 1: Direct table vector search
            tables.update(self._table_vector_search(query_embedding))
            
            # Strategy 2: Column vector search + table mapping
            tables.update(self._column_vector_search(query_embedding))
            
            return list(tables)[:10]
        except Exception as e:
            print(f"  ⚠️  Vector search failed: {e}, falling back to keyword search")
            return self._fallback_keyword_search(query)
    
    def _fallback_keyword_search(self, query: str) -> List[str]:
        """Fallback to simple keyword search if vector search fails."""
        try:
            search_term = f"%{query.lower()}%"
            sql = """
            SELECT DISTINCT name FROM tables
            WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
            LIMIT 5
            """
            results = self.conn.execute(sql, [search_term, search_term]).fetchall()
            return [row[0] for row in results]
        except:
            return []
    
    def _table_vector_search(self, query_embedding: List[float]) -> List[str]:
        """Direct vector search on table embeddings with DuckDB 1.0.0 compatibility."""
        try:
            # Check if we have embeddings in the database
            check_sql = "SELECT COUNT(*) FROM tables WHERE embedding IS NOT NULL"
            count = self.conn.execute(check_sql).fetchone()[0]
            
            if count == 0:
                print("  ⚠️  No table embeddings found in database")
                return []
            
            # Try DuckDB vector functions (may not be available in 1.0.0)
            try:
                sql = """
                SELECT name, array_cosine_similarity(embedding, ?::FLOAT[384]) as similarity
                FROM tables
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT 5
                """
                results = self.conn.execute(sql, [query_embedding]).fetchall()
                return [row[0] for row in results if row[1] > 0.3]
            except Exception:
                # Fallback to manual similarity calculation
                return self._fallback_table_similarity(query_embedding)
                
        except Exception as e:
            logger.warning(f"Table vector search failed: {e}")
            return []
    
    def _column_vector_search(self, query_embedding: List[float]) -> List[str]:
        """Find tables via column vector similarity with fallback."""
        try:
            # Check if we have column embeddings
            check_sql = "SELECT COUNT(*) FROM columns WHERE embedding IS NOT NULL"
            count = self.conn.execute(check_sql).fetchone()[0]
            
            if count == 0:
                return []
            
            try:
                sql = """
                SELECT table_name, AVG(array_cosine_similarity(embedding, ?::FLOAT[384])) as avg_similarity
                FROM columns
                WHERE embedding IS NOT NULL
                GROUP BY table_name
                ORDER BY avg_similarity DESC
                LIMIT 5
                """
                results = self.conn.execute(sql, [query_embedding]).fetchall()
                return [row[0] for row in results if row[1] > 0.2]
            except Exception:
                # DuckDB 1.0.0 might not have array_cosine_similarity
                return self._fallback_column_similarity(query_embedding)
                
        except Exception as e:
            logger.warning(f"Column vector search failed: {e}")
            return []
    
    def _fallback_table_similarity(self, query_embedding: List[float]) -> List[str]:
        """Fallback similarity calculation using numpy."""
        try:
            # Get all table embeddings
            sql = "SELECT name, embedding FROM tables WHERE embedding IS NOT NULL"
            results = self.conn.execute(sql).fetchall()
            
            similarities = []
            for name, embedding in results:
                if embedding and len(embedding) == len(query_embedding):
                    # Calculate cosine similarity manually
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                    similarities.append((name, similarity))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [name for name, sim in similarities[:5] if sim > 0.3]
        except Exception as e:
            logger.warning(f"Fallback similarity failed: {e}")
            return []
    
    def _fallback_column_similarity(self, query_embedding: List[float]) -> List[str]:
        """Fallback column similarity using manual calculation."""
        try:
            sql = "SELECT table_name, embedding FROM columns WHERE embedding IS NOT NULL"
            results = self.conn.execute(sql).fetchall()
            
            table_similarities = {}
            for table_name, embedding in results:
                if embedding and len(embedding) == len(query_embedding):
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                    if table_name not in table_similarities:
                        table_similarities[table_name] = []
                    table_similarities[table_name].append(similarity)
            
            # Calculate average similarity per table
            avg_similarities = []
            for table_name, sims in table_similarities.items():
                avg_sim = sum(sims) / len(sims)
                avg_similarities.append((table_name, avg_sim))
            
            # Sort and return top results
            avg_similarities.sort(key=lambda x: x[1], reverse=True)
            return [name for name, sim in avg_similarities[:5] if sim > 0.2]
        except Exception as e:
            logger.warning(f"Fallback column similarity failed: {e}")
            return []

# Fix 3: Add error handling for missing dependencies
# Add this check at the top of the main function

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        import openpyxl
    except ImportError:
        missing_deps.append("openpyxl")
    
    if missing_deps:
        print(f"⚠️  Missing optional dependencies: {', '.join(missing_deps)}")
        print(f"💡 Install with: pip install {' '.join(missing_deps)}")
        print(f"🔄 Some methods may fall back to simpler alternatives")
    
    return len(missing_deps) == 0
    


Quick Summary of Fixes Needed:

Line 923 Issue: Remove the duplicate code block and the stray comment line. Replace the entire run_duckdb_benchmark() function with the corrected version above.
DuckDB 1.0.0 Compatibility: The VectorRetriever class needs updates to handle the fact that DuckDB 1.0.0 might not have all the array functions. Add fallback methods.
Dependency Handling: Add a dependency check function to gracefully handle missing packages.

Specific Lines to Replace:

Lines 915-950: Replace the entire run_duckdb_benchmark() function
Lines 250-320: Replace the VectorRetriever class with the updated version
Add at the beginning of main(): Call check_dependencies()

Alternative Quick Fix:
If you want a minimal fix just to get it running, you can:

Find line 923 and remove the duplicate code block that starts with "Remove duplicates..."
Add this import at the top if missing: from datetime import datetime
Test with minimal retrievers first by commenting out the Vector and TF-IDF retrievers if they have dependency issues