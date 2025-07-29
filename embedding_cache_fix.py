class AdvancedGraphTraversalRetriever:
    """Advanced graph traversal using GNN + RL + Multi-level reasoning for table retrieval."""
    
    def __init__(self, db_path: str = DB_PATH):
        # ... existing initialization code ...
        
        # Add query embedding cache
        self.query_embedding_cache = {}
        
    def get_tables_with_details(self, query: str, max_tables: int = 10, 
                          similarity_threshold: float = None) -> Tuple[List[str], Dict]:
        """Enhanced table retrieval with embedding-based attention and similarity scores."""
        logger.info(f"🧠 Enhanced embedding-based retrieval for: '{query[:50]}...'")
        
        # Cache the query embedding once at the beginning
        query_embedding = self._get_cached_query_embedding(query)
        
        # Use provided threshold or default
        if similarity_threshold is not None:
            effective_threshold = max(similarity_threshold, self.min_similarity_threshold)
        else:
            effective_threshold = self.similarity_threshold
        
        try:
            # 1. Embedding-based attention retrieval with scores (pass cached embedding)
            embedding_results, embedding_scores = self._embedding_attention_retrieval_with_scores(
                query, query_embedding
            )
            
            # 2. Multi-hop embedding reasoning (pass cached embedding)
            multihop_results = self._multi_hop_embedding_reasoning(
                query, embedding_results[:5], query_embedding
            )
            
            # 3. Column-level matching (pass cached embedding)
            column_matches = self._embedding_column_matching(
                query, multihop_results, query_embedding
            )
            
            # ... rest of the method unchanged ...
            
        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            # Fallback to existing methods
            fallback_tables = self._fallback_basic_traversal(query)
            table_details = self._get_table_details(fallback_tables[:max_tables])
            return fallback_tables[:max_tables], table_details
    
    def _get_cached_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding from cache or create new one."""
        # Use query as cache key (you might want to hash it for very long queries)
        cache_key = query.strip().lower()
        
        if cache_key not in self.query_embedding_cache:
            logger.info(f"🔄 Creating new embedding for query: '{query[:50]}...'")
            self.query_embedding_cache[cache_key] = self._encode_query_with_openai(query)
        else:
            logger.info(f"✅ Using cached embedding for query: '{query[:50]}...'")
            
        return self.query_embedding_cache[cache_key]
    
    def _embedding_attention_retrieval_with_scores(self, query: str, 
                                                 query_embedding: np.ndarray = None) -> Tuple[List[str], List[float]]:
        """Enhanced embedding-based retrieval with attention mechanism and scores."""
        # Use provided embedding or get it (but preferably always provide it)
        if query_embedding is None:
            query_embedding = self._encode_query_with_openai(query)
        
        # Get all table embeddings from database
        table_embeddings = self._get_table_embeddings_from_db()
        
        attention_scores = {}
        
        for table_name, table_embedding in table_embeddings.items():
            # Compute cosine similarity with query
            similarity = self._cosine_similarity(query_embedding, table_embedding)
            
            # Apply attention mechanism considering graph structure
            attention_score = self._compute_attention_with_structure(
                table_name, similarity, query
            )
            
            attention_scores[table_name] = attention_score
        
        # Sort by attention scores (high to low)
        sorted_items = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return tables and scores separately
        tables = [table for table, score in sorted_items]
        scores = [score for table, score in sorted_items]
        
        return tables, scores
    
    def _multi_hop_embedding_reasoning(self, query: str, seed_tables: List[str], 
                                     query_embedding: np.ndarray = None) -> List[str]:
        """Multi-hop reasoning using embeddings and attention."""
        # Use provided embedding or get it
        if query_embedding is None:
            query_embedding = self._encode_query_with_openai(query)
            
        discovered_tables = set(seed_tables)
        
        for hop in range(self.max_hops):
            hop_discoveries = set()
            
            for current_table in list(discovered_tables):
                # Get neighbor tables through relationships
                neighbors = self._get_table_neighbors(current_table)
                
                for neighbor in neighbors:
                    # Compute path score using embeddings
                    path_score = self._compute_path_embedding_score(
                        current_table, neighbor, query_embedding
                    )
                    
                    # Apply attention mechanism for multi-hop scoring
                    hop_attention = self._compute_hop_attention(
                        current_table, neighbor, hop, query_embedding
                    )
                    
                    final_score = path_score * hop_attention
                    
                    if final_score > 0.4:  # Dynamic threshold
                        hop_discoveries.add(neighbor)
            
            # Add diverse paths to avoid local optima
            if hop_discoveries:
                discovered_tables.update(hop_discoveries)
            else:
                break  # No more promising paths
        
        return list(discovered_tables)
    
    def _embedding_column_matching(self, query: str, tables: List[str], 
                                 query_embedding: np.ndarray = None) -> Dict[str, List[str]]:
        """Match query to relevant columns using embeddings."""
        # Use provided embedding or get it
        if query_embedding is None:
            query_embedding = self._encode_query_with_openai(query)
            
        table_column_matches = {}
        
        for table_name in tables:
            column_scores = []
            
            try:
                # Get column embeddings from database
                columns_sql = """
                SELECT name, embedding, description 
                FROM columns 
                WHERE table_name = ? AND embedding IS NOT NULL
                """
                columns = self.conn.execute(columns_sql, [table_name]).fetchall()
                
                for col_name, col_embedding, col_desc in columns:
                    if col_embedding:
                        col_embedding_array = np.array(col_embedding)
                        similarity = self._cosine_similarity(query_embedding, col_embedding_array)
                        
                        # Boost score for columns with relevant descriptions
                        if col_desc and any(term in col_desc.lower() for term in query.lower().split()):
                            similarity *= 1.2
                        
                        column_scores.append((col_name, similarity))
                
                # Sort and filter relevant columns
                column_scores.sort(key=lambda x: x[1], reverse=True)
                relevant_columns = [col for col, score in column_scores if score > 0.5]
                
                if relevant_columns:
                    table_column_matches[table_name] = relevant_columns[:5]  # Top 5 columns
                    
            except Exception as e:
                logger.warning(f"Failed to match columns for {table_name}: {e}")
        
        return table_column_matches
    
    def clear_query_cache(self):
        """Clear the query embedding cache (useful for memory management)."""
        self.query_embedding_cache.clear()
        logger.info("🗑️ Query embedding cache cleared")
    
    def get_cache_stats(self):
        """Get cache statistics for monitoring."""
        return {
            'cached_queries': len(self.query_embedding_cache),
            'cache_keys': list(self.query_embedding_cache.keys())
        }