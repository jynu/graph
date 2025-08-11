Fix 1: Update _compute_attention_with_structure method
pythondef _compute_attention_with_structure(self, table_name: str, similarity: float, query: str) -> Dict[str, float]:
    """Compute attention score considering both embedding similarity and graph structure."""
    # Base score from embedding similarity
    embedding_score = similarity * self.embedding_weight
    
    # Add structural information
    try:
        # Get table centrality (hub/authority scores)
        centrality_score = self._get_table_centrality(table_name)
        structure_score = centrality_score * self.structure_weight
        
        # Add query-specific relevance
        query_relevance = self._compute_query_table_relevance(table_name, query)
        attention_score = query_relevance * self.attention_weight
        
    except Exception as e:
        logger.warning(f"Failed to compute structural attention for {table_name}: {e}")
        structure_score = 0.0
        attention_score = 0.0
    
    # Combined attention score
    combined_score = embedding_score + structure_score + attention_score
    
    # Return component breakdown
    return {
        'embedding': similarity,  # Raw embedding similarity (0-1)
        'structure': centrality_score if 'centrality_score' in locals() else 0.0,  # Raw centrality score
        'attention': query_relevance if 'query_relevance' in locals() else 0.0,  # Raw query relevance
        'path_diversity': 0.0,  # Placeholder - will be updated by path diversity calculation
        'combined_score': combined_score
    }
Fix 2: Add error handling in _embedding_attention_retrieval_with_scores
pythondef _embedding_attention_retrieval_with_scores(self, query: str, 
                                             query_embedding: np.ndarray = None) -> Tuple[List[str], List[float], Dict[str, Dict[str, float]]]:
    """Enhanced embedding-based retrieval with attention mechanism and component scores."""
    # Use provided embedding or get it
    if query_embedding is None:
        query_embedding = self._encode_query_with_openai(query)
    
    # Get all table embeddings from database
    table_embeddings = self._get_table_embeddings_from_db()
    
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
    
    return tables, scores, component_scores
Fix 3: Add safety check in _enrich_details_with_similarity_scores
pythondef _enrich_details_with_similarity_scores(self, table_details: Dict, query: str, 
                                     all_tables: List[str], all_scores: List[float],
                                     component_scores: Dict[str, Dict[str, float]] = None) -> Dict:
    """Enrich table details with similarity scores and ranking information."""
    try:
        # Create score lookup
        score_lookup = dict(zip(all_tables, all_scores))
        
        for table_name in table_details:
            if 'error' not in table_details[table_name]:
                # Add similarity score
                similarity_score = score_lookup.get(table_name, 0.0)
                table_details[table_name]['similarity_score'] = round(similarity_score, 3)
                
                # Add ranking
                if table_name in all_tables:
                    rank = all_tables.index(table_name) + 1
                    table_details[table_name]['rank'] = rank
                else:
                    table_details[table_name]['rank'] = 999
                
                # Add confidence level based on score
                if similarity_score >= 0.8:
                    confidence = "High"
                elif similarity_score >= 0.5:
                    confidence = "Medium"
                elif similarity_score >= 0.2:
                    confidence = "Low"
                else:
                    confidence = "Very Low"
                
                table_details[table_name]['confidence'] = confidence
                
                # Add component scores breakdown with safety checks
                if component_scores and table_name in component_scores:
                    components = component_scores[table_name]
                    
                    # Ensure all required keys exist
                    safe_components = {
                        'embedding_score': round(components.get('embedding', 0.0), 3),
                        'structure_score': round(components.get('structure', 0.0), 3),
                        'attention_score': round(components.get('attention', 0.0), 3),
                        'path_diversity_score': round(components.get('path_diversity', 0.0), 3)
                    }
                    
                    table_details[table_name]['component_breakdown'] = safe_components
                    table_details[table_name]['component_scores'] = components
        
        return table_details
        
    except Exception as e:
        logger.warning(f"Failed to enrich details with similarity scores: {e}")
        return table_details
Fix 4: Update the method call in get_tables_with_details
pythondef get_tables_with_details(self, query: str, max_tables: int = 10, 
                      similarity_threshold: float = None) -> Tuple[List[str], Dict]:
    """Enhanced table retrieval with embedding-based attention and similarity scores."""
    logger.info(f"🧠 Enhanced embedding-based retrieval for: '{query[:50]}...'")
    
    query_embedding = self._get_cached_query_embedding(query)
    
    # Use provided threshold or default
    if similarity_threshold is not None:
        effective_threshold = max(similarity_threshold, self.min_similarity_threshold)
    else:
        effective_threshold = self.similarity_threshold
    
    try:
        # 1. Embedding-based attention retrieval with scores (pass cached embedding)
        embedding_results, embedding_scores, component_scores = self._embedding_attention_retrieval_with_scores(
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
        
        # 4. Ensemble ranking with embedding awareness
        final_results, final_scores = self._embedding_aware_ensemble_with_scores(
            embedding_results, embedding_scores, multihop_results, column_matches, query
        )
        
        # 5. Filter by threshold and limit results
        filtered_results = [
            table for table, score in zip(final_results, final_scores) 
            if score >= effective_threshold
        ][:max_tables]
        
        # If no results above threshold, take top results anyway
        if not filtered_results and final_results:
            filtered_results = final_results[:min(3, max_tables)]
            logger.info(f"🔄 No results above threshold {effective_threshold:.2f}, returning top {len(filtered_results)} results")
        
        # Get detailed information for top tables
        table_details = self._get_table_details(filtered_results)
        
        # Enrich details with similarity scores and component breakdown
        table_details = self._enrich_details_with_similarity_scores(
            table_details, query, final_results, final_scores, component_scores
        )
        
        logger.info(f"📊 Found {len(filtered_results)} tables using Enhanced Embedding Retrieval")
        return filtered_results, table_details
    
    except Exception as e:
        logger.error(f"Enhanced retrieval failed: {e}")
        # Fallback to existing methods
        fallback_tables = self._fallback_basic_traversal(query)
        table_details = self._get_table_details(fallback_tables[:max_tables])
        return fallback_tables[:max_tables], table_details
The key fixes are: