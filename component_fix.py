Here's the fix for the _embedding_attention_retrieval_with_scores method in text_to_sql_backend.py:
Fix the method signature and variable:
pythondef _embedding_attention_retrieval_with_scores(self, query: str, 
                                             query_embedding: np.ndarray = None) -> Tuple[List[str], List[float], Dict[str, Dict[str, float]]]:
    """Enhanced embedding-based retrieval with attention mechanism and component scores."""
    # Use provided embedding or get it (but preferably always provide it)
    if query_embedding is None:
        query_embedding = self._encode_query_with_openai(query)
    
    # Get all table embeddings from database - THIS LINE WAS MISSING
    table_embeddings = self._get_table_embeddings_from_db()
    
    attention_scores = {}
    component_scores = {}  # New: store individual component scores
    
    for table_name, table_embedding in table_embeddings.items():  # NOW table_embeddings is defined
        # Compute cosine similarity with query
        similarity = self._cosine_similarity(query_embedding, table_embedding)
        
        # Apply attention mechanism with component breakdown
        component_breakdown = self._compute_attention_with_structure(table_name, similarity, query)
        
        attention_scores[table_name] = component_breakdown['combined_score']
        component_scores[table_name] = component_breakdown
    
    # Sort by attention scores (high to low)
    sorted_items = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return tables, scores, and component scores separately
    tables = [table for table, score in sorted_items]
    scores = [score for table, score in sorted_items]
    
    return tables, scores, component_scores
Also update the method call in get_tables_with_details:
pythondef get_tables_with_details(self, query: str, max_tables: int = 10, 
                      similarity_threshold: float = None) -> Tuple[List[str], Dict]:
    """Enhanced table retrieval with embedding-based attention and similarity scores."""
    # ... existing code ...
    
    try:
        # 1. Embedding-based attention retrieval with component scores
        embedding_results, embedding_scores, component_scores = self._embedding_attention_retrieval_with_scores(
            query, query_embedding
        )
        
        # ... rest of the existing code ...
        
        # 5. Enrich details with component scores
        table_details = self._enrich_details_with_similarity_scores(
            table_details, query, final_results, final_scores, component_scores
        )
        
        return filtered_results, table_details
The key fix is adding this line that was missing:
pythontable_embeddings = self._get_table_embeddings_from_db()