Backend Changes (text_to_sql_backend.py)
1. Update _compute_attention_with_structure method
Currently returns only the final combined score. Modify to return individual component scores:
pythondef _compute_attention_with_structure(self, table_name: str, similarity: float, query: str) -> Dict[str, float]:
    """Compute attention score with individual component breakdowns."""
    # Base score from embedding similarity
    embedding_score = similarity * self.embedding_weight
    
    # Structural information  
    centrality_score = self._get_table_centrality(table_name) * self.structure_weight
    
    # Query-specific relevance
    query_relevance = self._compute_query_table_relevance(table_name, query) * self.attention_weight
    
    # Combined attention score
    attention_score = embedding_score + centrality_score + query_relevance
    
    return {
        'embedding': similarity,
        'structure': centrality_score / self.structure_weight if self.structure_weight > 0 else 0,
        'attention': query_relevance / self.attention_weight if self.attention_weight > 0 else 0,
        'path_diversity': 0.0,  # Will be updated by path diversity calculation
        'combined_score': attention_score
    }
2. Update _embedding_attention_retrieval_with_scores method
Modify to collect and return component scores:
pythondef _embedding_attention_retrieval_with_scores(self, query: str, query_embedding: np.ndarray = None) -> Tuple[List[str], List[float], Dict[str, Dict[str, float]]]:
    """Enhanced embedding-based retrieval with component score breakdown."""
    
    attention_scores = {}
    component_scores = {}  # New: store individual component scores
    
    for table_name, table_embedding in table_embeddings.items():
        # Compute cosine similarity with query
        similarity = self._cosine_similarity(query_embedding, table_embedding)
        
        # Apply attention mechanism with component breakdown
        component_breakdown = self._compute_attention_with_structure(table_name, similarity, query)
        
        attention_scores[table_name] = component_breakdown['combined_score']
        component_scores[table_name] = component_breakdown
    
    # Sort and return
    sorted_items = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
    tables = [table for table, score in sorted_items]
    scores = [score for table, score in sorted_items]
    
    return tables, scores, component_scores
3. Update _enrich_details_with_similarity_scores method
Add component scores to table details:
pythondef _enrich_details_with_similarity_scores(self, table_details: Dict, query: str, 
                                     all_tables: List[str], all_scores: List[float],
                                     component_scores: Dict[str, Dict[str, float]] = None) -> Dict:
    """Enrich table details with similarity scores and component breakdown."""
    
    for table_name in table_details:
        if 'error' not in table_details[table_name]:
            # Existing code for similarity_score, rank, confidence...
            
            # Add component scores breakdown
            if component_scores and table_name in component_scores:
                table_details[table_name]['component_scores'] = component_scores[table_name]
                
                # Add readable component breakdown
                components = component_scores[table_name]
                table_details[table_name]['component_breakdown'] = {
                    'embedding_score': round(components.get('embedding', 0), 3),
                    'structure_score': round(components.get('structure', 0), 3),
                    'attention_score': round(components.get('attention', 0), 3),
                    'path_diversity_score': round(components.get('path_diversity', 0), 3)
                }
    
    return table_details
4. Update get_tables_with_details method
Propagate component scores through the pipeline:
pythondef get_tables_with_details(self, query: str, max_tables: int = 10, similarity_threshold: float = None):
    """Enhanced table retrieval with component score breakdown."""
    
    # 1. Get embedding results with component scores
    embedding_results, embedding_scores, component_scores = self._embedding_attention_retrieval_with_scores(
        query, query_embedding
    )
    
    # ... existing multi-hop and column matching code ...
    
    # 5. Enrich details with component scores
    table_details = self._enrich_details_with_similarity_scores(
        table_details, query, final_results, final_scores, component_scores
    )
    
    return filtered_results, table_details
Frontend Changes (text_to_sql_frontend.py)
1. Update create_table_summary_with_scores function
Add component score display:
pythondef create_table_summary_with_scores(tables: List[str], table_details: Dict) -> str:
    """Create summary with component score breakdown."""
    
    for table_info_item in tables_in_group:
        # ... existing code ...
        
        # Add component breakdown if available
        component_text = ""
        if 'component_breakdown' in table_details.get(table_name, {}):
            components = table_details[table_name]['component_breakdown']
            component_text = f" | E:{components['embedding_score']:.2f} S:{components['structure_score']:.2f} A:{components['attention_score']:.2f} P:{components['path_diversity_score']:.2f}"
        
        summary_text += f"**#{rank}** `{display_name}` ({table_type}) *Score: {score:.3f}*{component_text}{desc_text}\n"
2. Update format_table_details_for_display function
Add detailed component score section:
pythondef format_table_details_for_display(table_details: Dict) -> str:
    """Format table details with component score breakdown."""
    
    for table_name, details in table_details.items():
        # ... existing code ...
        
        # Add component score breakdown
        if 'component_breakdown' in details:
            components = details['component_breakdown']
            formatted_text += f"\n**Similarity Component Breakdown:**\n"
            formatted_text += f"- **Embedding Score:** {components['embedding_score']:.3f}\n"
            formatted_text += f"- **Structure Score:** {components['structure_score']:.3f}\n"
            formatted_text += f"- **Attention Score:** {components['attention_score']:.3f}\n"
            formatted_text += f"- **Path Diversity Score:** {components['path_diversity_score']:.3f}\n"
Optional: Add Debug Logging
Add to backend initialization:
pythondef __init__(self, db_path: str = DB_PATH):
    # ... existing code ...
    self.debug_component_scores = True  # Add flag for component score logging
Add logging in scoring methods:
pythonif self.debug_component_scores:
    logger.info(f"🔍 Component scores for {table_name}: "
               f"Embedding={components['embedding']:.3f}, "
               f"Structure={components['structure']:.3f}, "
               f"Attention={components['attention']:.3f}, "
               f"PathDiv={components['path_diversity']:.3f}, "
               f"Combined={components['combined_score']:.3f}")
Summary of Files to Update: