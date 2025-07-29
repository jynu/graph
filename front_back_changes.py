1. Update AdvancedGraphTraversalRetriever.__init__()
pythondef __init__(self, db_path: str = DB_PATH):
    try:
        self.conn = duckdb.connect(db_path)
        # Test connection
        table_count = self.conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
        logger.info(f"✅ Connected to DuckDB: {table_count} tables available")
        
        self.graph_structure = self._build_enhanced_graph()
        self.table_embeddings = self._compute_table_embeddings()
        self.centrality_cache = {}
        
        # Add embedding-based similarity computation
        self.embedding_cache = {}
        self.attention_weights = {}
        
        # Lower threshold - we'll filter by rank instead
        self.similarity_threshold = 0.1  # Much lower threshold
        self.min_similarity_threshold = 0.05  # Absolute minimum
        
        # Initialize multi-hop reasoning parameters
        self.max_hops = 3
        self.path_diversity_weight = 0.2
        self.embedding_weight = 0.5
        self.structure_weight = 0.3
        self.attention_weight = 0.2
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to DuckDB: {e}")
        raise Exception(f"Database connection failed: {str(e)}")
2. Update get_tables_with_details() to include similarity scores
pythondef get_tables_with_details(self, query: str, max_tables: int = 10, 
                          similarity_threshold: float = None) -> Tuple[List[str], Dict]:
    """Enhanced table retrieval with embedding-based attention and similarity scores."""
    logger.info(f"🧠 Enhanced embedding-based retrieval for: '{query[:50]}...'")
    
    # Use provided threshold or default
    if similarity_threshold is not None:
        effective_threshold = max(similarity_threshold, self.min_similarity_threshold)
    else:
        effective_threshold = self.similarity_threshold
    
    try:
        # 1. Embedding-based attention retrieval with scores
        embedding_results, embedding_scores = self._embedding_attention_retrieval_with_scores(query)
        
        # 2. Multi-hop embedding reasoning
        multihop_results = self._multi_hop_embedding_reasoning(query, embedding_results[:5])
        
        # 3. Column-level matching
        column_matches = self._embedding_column_matching(query, multihop_results)
        
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
            filtered_results = final_results[:min(3, max_tables)]  # At least top 3
            logger.info(f"🔄 No results above threshold {effective_threshold:.2f}, returning top {len(filtered_results)} results")
        
        # Get detailed information for top tables
        table_details = self._get_table_details(filtered_results)
        
        # Enrich details with similarity scores
        table_details = self._enrich_details_with_similarity_scores(
            table_details, query, final_results, final_scores
        )
        
        logger.info(f"📊 Found {len(filtered_results)} tables using Enhanced Embedding Retrieval")
        return filtered_results, table_details
        
    except Exception as e:
        logger.error(f"Enhanced retrieval failed: {e}")
        # Fallback to existing methods
        fallback_tables = self._fallback_basic_traversal(query)
        table_details = self._get_table_details(fallback_tables[:max_tables])
        return fallback_tables[:max_tables], table_details
3. Add new methods with similarity scores
pythondef _embedding_attention_retrieval_with_scores(self, query: str) -> Tuple[List[str], List[float]]:
    """Enhanced embedding-based retrieval with attention mechanism and scores."""
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

def _embedding_aware_ensemble_with_scores(self, embedding_results: List[str], 
                                        embedding_scores: List[float],
                                        multihop_results: List[str], 
                                        column_matches: Dict[str, List[str]], 
                                        query: str) -> Tuple[List[str], List[float]]:
    """Ensemble ranking with embedding awareness and similarity scores."""
    table_scores = {}
    
    # Score from embedding similarity (use actual scores)
    for i, (table, score) in enumerate(zip(embedding_results, embedding_scores)):
        position_weight = 1.0 / (i + 1)  # Position decay
        table_scores[table] = table_scores.get(table, 0) + (score * 0.7 + position_weight * 0.1) * 0.4
    
    # Score from multi-hop reasoning
    for i, table in enumerate(multihop_results):
        position_weight = 1.0 / (i + 1)
        table_scores[table] = table_scores.get(table, 0) + position_weight * 0.3
    
    # Score from column matching
    for table, columns in column_matches.items():
        column_boost = len(columns) * 0.1  # Boost for having relevant columns
        table_scores[table] = table_scores.get(table, 0) + column_boost * 0.3
    
    # Sort by combined score
    sorted_items = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return tables and scores separately
    tables = [table for table, score in sorted_items]
    scores = [score for table, score in sorted_items]
    
    return tables, scores

def _enrich_details_with_similarity_scores(self, table_details: Dict, query: str, 
                                         all_tables: List[str], all_scores: List[float]) -> Dict:
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
        
        return table_details
        
    except Exception as e:
        logger.warning(f"Failed to enrich details with similarity scores: {e}")
        return table_details
4. Update the backend interface method
pythonasync def find_relevant_tables(self, query: str, max_tables: int = 10, 
                             similarity_threshold: float = None) -> Tuple[List[str], Dict, str]:
    """Find relevant tables using Advanced Graph Traversal with similarity threshold."""
    try:
        start_time = time.time()
        
        tables, table_details = self.graph_retriever.get_tables_with_details(
            query, max_tables, similarity_threshold
        )
        
        processing_time = time.time() - start_time
        
        # Enhanced status message with similarity info
        if tables and table_details:
            # Get similarity range
            scores = [details.get('similarity_score', 0) for details in table_details.values() 
                     if 'similarity_score' in details]
            if scores:
                max_score = max(scores)
                min_score = min(scores)
                threshold_used = similarity_threshold if similarity_threshold else 0.1
                status_message = (f"✅ Found {len(tables)} relevant tables in {processing_time:.3f}s. "
                               f"Similarity range: {min_score:.3f} - {max_score:.3f} "
                               f"(threshold: {threshold_used:.2f})")
            else:
                status_message = f"✅ Found {len(tables)} relevant tables in {processing_time:.3f}s"
        else:
            status_message = f"⚠️ No tables found above similarity threshold in {processing_time:.3f}s"
        
        return tables, table_details, status_message
        
    except Exception as e:
        error_message = f"❌ Table discovery failed: {str(e)}"
        logger.error(error_message)
        return [], {}, error_message
Frontend Changes
1. Add similarity threshold control in the interface
Update the create_text_to_sql_interface() function:
python# Step 1: Query Input and Table Discovery
with gr.Group():
    gr.Markdown("### 🔍 Step 1: Enter Query and Discover Tables")
    
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="💭 Natural Language Query",
                placeholder="Example: Fetch us a list of all quotes for equities in APAC region",
                lines=3,
                value=""
            )
            
            with gr.Row():
                max_tables_slider = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Max Tables to Find"
                )
                similarity_threshold_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.8,
                    value=0.2,
                    step=0.05,
                    label="Similarity Threshold",
                    info="Lower = more results, Higher = more precise"
                )
            
            find_tables_btn = gr.Button("🔍 Find Tables", variant="primary")
2. Update the step1_find_tables function
pythondef step1_find_tables(query: str, max_tables: int = 10, similarity_threshold: float = 0.2):
    """Step 1: Find relevant tables using Advanced Graph Traversal with similarity control."""
    if not query.strip():
        return "Please enter a query.", "", "", gr.CheckboxGroup(choices=[], value=[]), pd.DataFrame(), "Enter a query to find tables"
    
    if not app_state['backend_ready']:
        return "❌ Backend not initialized", "", "", gr.CheckboxGroup(choices=[], value=[]), pd.DataFrame(), "System not ready"
    
    try:
        # Update app state
        app_state['current_query'] = query
        
        # Find tables using backend with similarity threshold
        backend = get_backend_instance()
        tables, table_details, status_message = run_async_function(
            backend.find_relevant_tables, query, max_tables, similarity_threshold
        )
        
        # Update app state
        app_state['found_tables'] = tables
        app_state['selected_tables'] = tables.copy()  # Select all by default
        app_state['table_details'] = table_details
        
        # Create table summary with similarity scores
        table_summary_text = create_table_summary_with_scores(tables, table_details)
        
        # Format detailed table information
        table_details_text = format_table_details_for_display(table_details)
        
        # Create performance summary with similarity info
        similarity_info = ""
        if table_details:
            scores = [details.get('similarity_score', 0) for details in table_details.values() 
                     if 'similarity_score' in details]
            if scores:
                avg_score = sum(scores) / len(scores)
                similarity_info = f" (Avg similarity: {avg_score:.3f})"
        
        performance_data = [{
            'Step': 'Table Discovery',
            'Method': 'Enhanced Embedding + Attention',
            'Tables Found': len(tables),
            'Status': f'✅ Success{similarity_info}' if tables else '⚠️ No tables found',
            'Threshold': f'{similarity_threshold:.2f}',
            'Timestamp': datetime.now().strftime('%H:%M:%S')
        }]
        
        performance_df = pd.DataFrame(performance_data)
        
        # Create checkbox group
        table_checkbox = gr.CheckboxGroup(
            choices=tables,
            value=tables,  # Select all by default
            label="✅ Select Tables for SQL Generation",
            info=f"Found {len(tables)} tables ranked by similarity. All selected by default."
        )
        
        guidance_message = (f"Found {len(tables)} tables with similarity ≥ {similarity_threshold:.2f}. "
                          f"Adjust threshold if needed, select tables, and click 'Generate SQL'.")
        
        return (
            status_message,
            table_summary_text,
            table_details_text,
            table_checkbox,
            performance_df,
            guidance_message
        )
        
    except Exception as e:
        error_msg = f"❌ Error in table discovery: {str(e)}"
        logger.error(error_msg)
        return error_msg, "", "", gr.CheckboxGroup(choices=[], value=[]), pd.DataFrame(), "Table discovery failed"
3. Update the table summary function with similarity scores
pythondef create_table_summary_with_scores(tables: List[str], table_details: Dict) -> str:
    """Create a compact summary of found tables with similarity scores."""
    if not tables:
        return "No tables found."
    
    summary_text = f"## 📊 Found {len(tables)} Relevant Tables (Ranked by Similarity)\n\n"
    
    # Create list of tables with their scores for sorting
    table_info = []
    for table_name in tables:
        if table_name in table_details and 'error' not in table_details[table_name]:
            details = table_details[table_name]
            score = details.get('similarity_score', 0.0)
            confidence = details.get('confidence', 'Unknown')
            rank = details.get('rank', 999)
            table_type = details.get('table_type', 'unknown')
            description = details.get('description', '')
            
            table_info.append({
                'name': table_name,
                'score': score,
                'confidence': confidence,
                'rank': rank,
                'type': table_type,
                'description': description
            })
    
    # Sort by rank (already sorted, but ensure consistency)
    table_info.sort(key=lambda x: x['rank'])
    
    # Group by confidence level for better presentation
    confidence_groups = {
        'High': [t for t in table_info if t['confidence'] == 'High'],
        'Medium': [t for t in table_info if t['confidence'] == 'Medium'],
        'Low': [t for t in table_info if t['confidence'] == 'Low'],
        'Very Low': [t for t in table_info if t['confidence'] == 'Very Low']
    }
    
    for confidence_level, tables_in_group in confidence_groups.items():
        if not tables_in_group:
            continue
            
        # Confidence level emoji
        confidence_emoji = {
            'High': '🟢',
            'Medium': '🟡', 
            'Low': '🟠',
            'Very Low': '🔴'
        }
        
        summary_text += f"### {confidence_emoji[confidence_level]} {confidence_level} Confidence ({len(tables_in_group)} tables):\n"
        
        for table_info_item in tables_in_group:
            table_name = table_info_item['name']
            score = table_info_item['score']
            rank = table_info_item['rank']
            table_type = table_info_item['type']
            description = table_info_item['description']
            
            # Truncate description
            desc_text = ""
            if description and len(description) > 0:
                desc_text = f" - {description[:60]}..." if len(description) > 60 else f" - {description}"
            
            summary_text += f"**#{rank}** **{table_name}** ({table_type}) - *Score: {score:.3f}*{desc_text}\n"
        
        summary_text += "\n"
    
    summary_text += """
💡 **Tips:**
- **Higher scores** = more relevant to your query
- **Adjust similarity threshold** above to see more/fewer results  
- **All found tables** are ranked by relevance
- Click 'Detailed Table Information' below for column details
"""
    
    return summary_text
4. Update the event handler
python# Step 1: Find tables
find_tables_btn.click(
    fn=step1_find_tables,
    inputs=[query_input, max_tables_slider, similarity_threshold_slider],
    outputs=[
        step1_status,
        table_summary,
        table_details_display,
        table_selection,
        performance_table,
        step_guidance
    ]
)
Summary of Changes
Backend:

Lowered default threshold to 0.1 (from 0.7)
Added similarity scores to all retrieval methods
Enhanced table details with similarity scores, rankings, and confidence levels
Added threshold parameter to the main interface method

Frontend:

Added similarity threshold slider (0.05 - 0.8, default 0.2)
Enhanced table display with rankings, scores, and confidence levels
Better guidance messages with similarity information
Grouped tables by confidence for easier interpretation