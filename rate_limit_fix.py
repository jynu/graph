Update the _encode_query_with_openai method in text_to_sql_backend.py:
pythonimport time
import random

def _encode_query_with_openai(self, query: str, max_retries: int = 3, base_delay: float = 1.0) -> np.ndarray:
    """Use OpenAI embeddings for query encoding with retry logic."""
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Getting OpenAI embedding (attempt {attempt + 1}/{max_retries}) for query: '{query[:50]}...'")
            
            # Use your existing embedding function
            from app.rag.embedding import embedding
            query_embedding = embedding.embed_text(query)
            
            logger.info(f"✅ Successfully got OpenAI embedding on attempt {attempt + 1}")
            return np.array(query_embedding)
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limit or server error
            is_retryable = any(code in error_msg for code in ['502', '503', '504', 'rate limit', 'timeout', 'bad gateway'])
            
            if attempt < max_retries - 1 and is_retryable:
                # Calculate delay with exponential backoff + jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"⚠️ OpenAI embedding failed (attempt {attempt + 1}): {e}")
                logger.info(f"🔄 Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                # Last attempt or non-retryable error
                if is_retryable:
                    logger.error(f"❌ OpenAI embedding failed after {max_retries} attempts: {e}")
                else:
                    logger.error(f"❌ OpenAI embedding failed with non-retryable error: {e}")
                
                # Fallback to local encoding
                logger.info("🔄 Falling back to local query encoding")
                return self._encode_query(query)
    
    # This should never be reached, but just in case
    logger.warning("🔄 Falling back to local query encoding (unexpected path)")
    return self._encode_query(query)
Also add retry logic to the table embedding retrieval method:
pythondef _get_table_embeddings_from_db_with_retry(self, max_retries: int = 2) -> Dict[str, np.ndarray]:
    """Retrieve table embeddings from DuckDB with retry logic."""
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Getting table embeddings from DB (attempt {attempt + 1}/{max_retries})")
            
            embeddings_sql = "SELECT name, embedding FROM tables WHERE embedding IS NOT NULL"
            results = self.conn.execute(embeddings_sql).fetchall()
            
            table_embeddings = {}
            for table_name, embedding_blob in results:
                if embedding_blob:
                    try:
                        table_embeddings[table_name] = np.array(embedding_blob)
                    except Exception as e:
                        logger.warning(f"Failed to parse embedding for table {table_name}: {e}")
            
            logger.info(f"✅ Successfully retrieved {len(table_embeddings)} table embeddings")
            return table_embeddings
            
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 1.0 * (attempt + 1)  # Simple linear backoff for DB operations
                logger.warning(f"⚠️ Database query failed (attempt {attempt + 1}): {e}")
                logger.info(f"🔄 Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"❌ Failed to retrieve table embeddings after {max_retries} attempts: {e}")
                return {}
    
    return {}
Update the _embedding_attention_retrieval_with_scores method:
pythondef _embedding_attention_retrieval_with_scores(self, query: str, 
                                             query_embedding: np.ndarray = None) -> Tuple[List[str], List[float], Dict[str, Dict[str, float]]]:
    """Enhanced embedding-based retrieval with attention mechanism and component scores."""
    try:
        # Use provided embedding or get it with retry logic
        if query_embedding is None:
            query_embedding = self._encode_query_with_openai(query, max_retries=3)
        
        # Get all table embeddings from database with retry
        table_embeddings = self._get_table_embeddings_from_db_with_retry(max_retries=2)
        
        if not table_embeddings:
            logger.warning("⚠️ No table embeddings available, using fallback method")
            # Fallback to basic retrieval
            fallback_tables = self._semantic_matching(query)
            fallback_scores = [0.5] * len(fallback_tables)
            fallback_components = {
                table: {
                    'embedding': 0.0,
                    'structure': 0.0,
                    'attention': 0.5,  # Give some attention score for semantic matching
                    'path_diversity': 0.0,
                    'combined_score': 0.5
                } for table in fallback_tables
            }
            return fallback_tables, fallback_scores, fallback_components
        
        attention_scores = {}
        component_scores = {}
        
        for table_name, table_embedding in table_embeddings.items():
            try:
                # Compute cosine similarity with query
                similarity = self._cosine_similarity(query_embedding, table_embedding)
                
                # Apply attention mechanism with component breakdown
                component_breakdown = self._compute_attention_with_structure(table_name, similarity, query)
                
                attention_scores[table_name] = component_breakdown['combined_score']
                component_scores[table_name] = component_breakdown
                
            except Exception as e:
                logger.warning(f"Failed to compute scores for table {table_name}: {e}")
                # Add default scores for failed tables
                attention_scores[table_name] = 0.0
                component_scores[table_name] = {
                    'embedding': 0.0,
                    'structure': 0.0,
                    'attention': 0.0,
                    'path_diversity': 0.0,
                    'combined_score': 0.0
                }
        
        # Sort by attention scores (high to low)
        sorted_items = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return tables, scores, and component scores separately
        tables = [table for table, score in sorted_items]
        scores = [score for table, score in sorted_items]
        
        logger.info(f"📊 Processed {len(tables)} tables with embedding-based scoring")
        return tables, scores, component_scores
        
    except Exception as e:
        logger.error(f"❌ Embedding attention retrieval failed: {e}")
        # Complete fallback
        fallback_tables = self._semantic_matching(query)[:10]
        fallback_scores = [0.3] * len(fallback_tables)
        fallback_components = {
            table: {
                'embedding': 0.0,
                'structure': 0.0,
                'attention': 0.3,
                'path_diversity': 0.0,
                'combined_score': 0.3
            } for table in fallback_tables
        }
        return fallback_tables, fallback_scores, fallback_components
Add configuration options to the class initialization:
pythondef __init__(self, db_path: str = DB_PATH):
    try:
        self.conn = duckdb.connect(db_path)
        # Test connection
        table_count = self.conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
        logger.info(f"✅ Connected to DuckDB: {table_count} tables available")
        
        # ... existing initialization code ...
        
        # Add retry configuration
        self.embedding_retry_config = {
            'max_retries': 3,
            'base_delay': 1.0,
            'max_delay': 10.0
        }
        
        # Rate limiting tracking
        self.last_embedding_request = 0
        self.min_request_interval = 0.1  # Minimum 100ms between requests
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to DuckDB: {e}")
        raise Exception(f"Database connection failed: {str(e)}")
Add rate limiting to prevent hitting limits:
pythondef _rate_limit_check(self):
    """Ensure we don't hit rate limits."""
    current_time = time.time()
    time_since_last = current_time - self.last_embedding_request
    
    if time_since_last < self.min_request_interval:
        sleep_time = self.min_request_interval - time_since_last
        logger.info(f"⏱️ Rate limiting: sleeping {sleep_time:.3f}s")
        time.sleep(sleep_time)
    
    self.last_embedding_request = time.time()
Then update _encode_query_with_openai to include rate limiting:
pythondef _encode_query_with_openai(self, query: str, max_retries: int = 3, base_delay: float = 1.0) -> np.ndarray:
    """Use OpenAI embeddings for query encoding with retry logic and rate limiting."""
    
    # Check rate limiting
    self._rate_limit_check()
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Getting OpenAI embedding (attempt {attempt + 1}/{max_retries}) for query: '{query[:50]}...'")
            
            # Use your existing embedding function
            from app.rag.embedding import embedding
            query_embedding = embedding.embed_text(query)
            
            logger.info(f"✅ Successfully got OpenAI embedding on attempt {attempt + 1}")
            return np.array(query_embedding)
            
        except Exception as e:
            # ... existing retry logic ...
Key improvements:

Exponential backoff: Delays increase with each retry (1s, 2s, 4s + jitter)