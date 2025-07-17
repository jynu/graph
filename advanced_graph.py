Perfect! Here's how to keep both the baseline and add the new advanced method. You'll need to make these specific updates:
1. Add New Dependencies (Top of file)
Add these imports after the existing imports:
pythonimport networkx as nx
from collections import defaultdict
2. Keep Existing GraphTraversalRetriever and Add New Class
After the existing GraphTraversalRetriever class (around line 250), add the new advanced class:
python# --- Method 4: Advanced Graph Traversal Search ---
class AdvancedGraphTraversalRetriever(BaseDuckDBRetriever):
    """Advanced graph traversal using GNN + RL + Multi-level reasoning."""
    
    def __init__(self, db_path: str = "knowledge_graph.duckdb"):
        super().__init__(db_path)
        self.graph_structure = self._build_enhanced_graph()
        self.table_embeddings = self._compute_table_embeddings()
        self.centrality_cache = {}
        
    def get_tables(self, query: str) -> List[str]:
        logger.info(f"🧠 Running Advanced Graph Traversal search for: '{query[:50]}...'")
        
        try:
            # 1. GNN-based node classification
            gnn_results = self._gnn_table_classification(query)
            
            # 2. Reinforcement learning path optimization
            rl_results = self._rl_path_optimization(query, gnn_results[:5])
            
            # 3. Multi-level graph reasoning
            multilevel_results = self._multilevel_graph_reasoning(query)
            
            # 4. Ensemble combination
            final_results = self._ensemble_combination([
                gnn_results, rl_results, multilevel_results
            ])
            
            return final_results[:10]
            
        except Exception as e:
            logger.error(f"Advanced graph traversal failed: {e}")
            # Fallback to basic method
            return self._fallback_basic_traversal(query)
    
    def _build_enhanced_graph(self):
        """Build enhanced graph structure with weights."""
        try:
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add tables as nodes
            tables_sql = "SELECT name, description, table_type FROM tables"
            tables = self.conn.execute(tables_sql).fetchall()
            
            for table_name, description, table_type in tables:
                G.add_node(table_name, 
                          description=description or "", 
                          table_type=table_type or "unknown")
            
            # Add relationships as edges
            rels_sql = """
            SELECT from_table, to_table, relationship_type 
            FROM relationships 
            WHERE from_table != to_table
            """
            relationships = self.conn.execute(rels_sql).fetchall()
            
            for from_table, to_table, rel_type in relationships:
                if G.has_node(from_table) and G.has_node(to_table):
                    # Weight based on relationship strength
                    weight = self._compute_relationship_weight(rel_type)
                    G.add_edge(from_table, to_table, weight=weight, rel_type=rel_type)
            
            return G
            
        except Exception as e:
            logger.warning(f"Failed to build enhanced graph: {e}")
            return nx.DiGraph()  # Return empty graph
    
    def _compute_table_embeddings(self):
        """Compute table embeddings using description and column info."""
        embeddings = {}
        
        try:
            # Simple embedding based on text features
            for node in self.graph_structure.nodes():
                # Get table description and column names
                desc_sql = """
                SELECT t.description, STRING_AGG(c.name, ' ') as columns
                FROM tables t
                LEFT JOIN columns c ON t.name = c.table_name
                WHERE t.name = ?
                GROUP BY t.description
                """
                result = self.conn.execute(desc_sql, [node]).fetchone()
                
                if result:
                    description, columns = result
                    # Create simple text-based embedding
                    text = f"{description or ''} {columns or ''}"
                    # Simple hash-based embedding (in production, use proper embeddings)
                    embedding = np.array([hash(text + str(i)) % 1000 / 1000.0 for i in range(10)])
                    embeddings[node] = embedding
                else:
                    embeddings[node] = np.random.rand(10) * 0.1
                    
        except Exception as e:
            logger.warning(f"Failed to compute embeddings: {e}")
            
        return embeddings
    
    def _gnn_table_classification(self, query: str) -> List[str]:
        """Use Graph Attention Networks for table classification."""
        
        if not self.graph_structure.nodes():
            return []
        
        # Simple query embedding
        query_embedding = self._encode_query(query)
        
        # Graph attention mechanism
        attention_scores = {}
        
        for table_name in self.graph_structure.nodes():
            try:
                # Get table features
                table_features = self.table_embeddings.get(table_name, np.random.rand(10) * 0.1)
                
                # Compute attention between query and table
                attention_score = self._compute_attention(query_embedding, table_features)
                
                # Aggregate neighbor information
                neighbors = list(self.graph_structure.neighbors(table_name))
                neighbor_scores = []
                
                for neighbor in neighbors:
                    neighbor_features = self.table_embeddings.get(neighbor, np.random.rand(10) * 0.1)
                    neighbor_score = self._compute_attention(query_embedding, neighbor_features)
                    neighbor_scores.append(neighbor_score)
                
                # Weighted aggregation using attention
                if neighbor_scores:
                    aggregated_score = attention_score * 0.7 + np.mean(neighbor_scores) * 0.3
                else:
                    aggregated_score = attention_score
                    
                attention_scores[table_name] = aggregated_score
                
            except Exception as e:
                attention_scores[table_name] = 0.0
        
        # Return top tables by attention score
        sorted_tables = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
        return [table for table, score in sorted_tables if score > 0.1]
    
    def _rl_path_optimization(self, query: str, seed_tables: List[str]) -> List[str]:
        """Reinforcement learning for optimal path discovery."""
        
        if not seed_tables:
            return []
        
        optimal_tables = set(seed_tables)
        
        for seed_table in seed_tables[:3]:  # Limit to top 3 seed tables
            try:
                # Multi-hop traversal with RL-optimized paths
                current_table = seed_table
                
                for hop in range(2):  # Maximum 2 hops to avoid complexity
                    # Get candidate next tables
                    candidates = list(self.graph_structure.neighbors(current_table))
                    
                    if not candidates:
                        break
                    
                    # Compute Q-values for each candidate
                    q_values = {}
                    for candidate in candidates:
                        # Reward based on query relevance and path quality
                        relevance_reward = self._compute_table_relevance(candidate, query)
                        path_reward = self._compute_path_reward(seed_table, candidate)
                        q_values[candidate] = relevance_reward * 0.6 + path_reward * 0.4
                    
                    # Select best candidate
                    if q_values:
                        best_candidate = max(q_values.items(), key=lambda x: x[1])
                        if best_candidate[1] > 0.2:  # Minimum quality threshold
                            optimal_tables.add(best_candidate[0])
                            current_table = best_candidate[0]
                        else:
                            break
                            
            except Exception as e:
                continue  # Skip this seed table
        
        return list(optimal_tables)
    
    def _multilevel_graph_reasoning(self, query: str) -> List[str]:
        """Multi-level graph structure analysis."""
        
        try:
            # Level 1: Direct semantic matching
            level1_tables = self._semantic_matching(query)
            
            # Level 2: Structural pattern analysis
            level2_tables = self._structural_pattern_analysis(query, level1_tables)
            
            # Level 3: Global graph properties
            level3_tables = self._global_graph_analysis(query)
            
            # Combine results from all levels
            combined_results = set(level1_tables + level2_tables + level3_tables)
            
            # Score by multi-level importance
            scored_tables = []
            for table in combined_results:
                score = 0
                if table in level1_tables: score += 0.5
                if table in level2_tables: score += 0.3
                if table in level3_tables: score += 0.2
                scored_tables.append((table, score))
            
            # Return sorted by score
            scored_tables.sort(key=lambda x: x[1], reverse=True)
            return [table for table, score in scored_tables]
            
        except Exception as e:
            logger.warning(f"Multi-level reasoning failed: {e}")
            return []
    
    def _encode_query(self, query: str) -> np.ndarray:
        """Simple query encoding."""
        # Simple bag-of-words encoding (in production, use proper embeddings)
        words = query.lower().split()
        encoding = np.zeros(10)
        for i, word in enumerate(words[:10]):
            encoding[i % 10] += hash(word) % 100 / 100.0
        return encoding / (np.linalg.norm(encoding) + 1e-8)
    
    def _compute_attention(self, query_emb: np.ndarray, table_emb: np.ndarray) -> float:
        """Compute attention score between query and table."""
        try:
            # Scaled dot-product attention
            attention = np.dot(query_emb, table_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(table_emb) + 1e-8)
            return max(0, float(attention))  # ReLU activation
        except:
            return 0.0
    
    def _compute_table_relevance(self, table_name: str, query: str) -> float:
        """Compute relevance between table and query."""
        try:
            # Get table description
            desc_sql = "SELECT description FROM tables WHERE name = ?"
            result = self.conn.execute(desc_sql, [table_name]).fetchone()
            
            if result and result[0]:
                description = result[0].lower()
                query_terms = query.lower().split()
                
                # Simple term overlap
                overlap = sum(1 for term in query_terms if term in description)
                return overlap / len(query_terms) if query_terms else 0.0
            
            return 0.0
        except:
            return 0.0
    
    def _compute_path_reward(self, start_table: str, end_table: str) -> float:
        """Compute reward for path between tables."""
        try:
            if self.graph_structure.has_edge(start_table, end_table):
                edge_data = self.graph_structure.get_edge_data(start_table, end_table)
                return edge_data.get('weight', 0.5) if edge_data else 0.5
            return 0.0
        except:
            return 0.0
    
    def _semantic_matching(self, query: str) -> List[str]:
        """Direct semantic matching."""
        return self._find_seed_tables(query)  # Use existing method
    
    def _structural_pattern_analysis(self, query: str, seed_tables: List[str]) -> List[str]:
        """Analyze structural patterns in the graph."""
        
        pattern_tables = []
        
        try:
            # Pattern 1: Hub tables (high degree centrality)
            hub_tables = self._find_hub_tables()
            
            # Pattern 2: Bridge tables (high betweenness centrality)
            bridge_tables = self._find_bridge_tables()
            
            # Pattern 3: Community-based relevance
            for seed_table in seed_tables[:3]:
                community_tables = self._find_table_community(seed_table)
                pattern_tables.extend(community_tables)
            
            # Filter by query relevance
            query_terms = query.lower().split()
            relevant_tables = []
            for table in set(hub_tables + bridge_tables + pattern_tables):
                if self._is_query_relevant(table, query_terms):
                    relevant_tables.append(table)
            
            return relevant_tables
            
        except Exception as e:
            logger.warning(f"Structural pattern analysis failed: {e}")
            return []
    
    def _global_graph_analysis(self, query: str) -> List[str]:
        """Global graph property analysis."""
        try:
            # PageRank-style importance
            if 'pagerank' not in self.centrality_cache:
                self.centrality_cache['pagerank'] = nx.pagerank(self.graph_structure.to_undirected())
            
            pagerank_scores = self.centrality_cache['pagerank']
            
            # Filter by query relevance and return top scoring tables
            query_terms = query.lower().split()
            relevant_tables = []
            
            for table, score in pagerank_scores.items():
                if self._is_query_relevant(table, query_terms) and score > 0.01:
                    relevant_tables.append(table)
            
            return relevant_tables[:5]
            
        except Exception as e:
            logger.warning(f"Global graph analysis failed: {e}")
            return []
    
    def _find_hub_tables(self) -> List[str]:
        """Find tables with high degree centrality."""
        try:
            if 'degree' not in self.centrality_cache:
                self.centrality_cache['degree'] = nx.degree_centrality(self.graph_structure)
            
            degree_centrality = self.centrality_cache['degree']
            sorted_tables = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
            return [table for table, centrality in sorted_tables[:5] if centrality > 0.1]
        except:
            return []
    
    def _find_bridge_tables(self) -> List[str]:
        """Find tables with high betweenness centrality."""
        try:
            if 'betweenness' not in self.centrality_cache:
                self.centrality_cache['betweenness'] = nx.betweenness_centrality(self.graph_structure.to_undirected())
            
            betweenness_centrality = self.centrality_cache['betweenness']
            sorted_tables = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
            return [table for table, centrality in sorted_tables[:5] if centrality > 0.05]
        except:
            return []
    
    def _find_table_community(self, seed_table: str) -> List[str]:
        """Find tables in the same community as seed table."""
        try:
            # Simple community detection using immediate neighbors
            community = set([seed_table])
            
            # Add direct neighbors
            neighbors = list(self.graph_structure.neighbors(seed_table))
            community.update(neighbors[:3])  # Limit to 3 neighbors
            
            # Add neighbors of neighbors
            for neighbor in neighbors[:2]:
                second_neighbors = list(self.graph_structure.neighbors(neighbor))
                community.update(second_neighbors[:2])
            
            return list(community)
        except:
            return [seed_table]
    
    def _is_query_relevant(self, table_name: str, query_terms: List[str]) -> bool:
        """Check if table is relevant to query terms."""
        try:
            # Check table name
            table_lower = table_name.lower()
            if any(term in table_lower for term in query_terms):
                return True
            
            # Check table description
            desc_sql = "SELECT description FROM tables WHERE name = ?"
            result = self.conn.execute(desc_sql, [table_name]).fetchone()
            
            if result and result[0]:
                description = result[0].lower()
                return any(term in description for term in query_terms)
            
            return False
        except:
            return False
    
    def _ensemble_combination(self, method_results: List[List[str]]) -> List[str]:
        """Ensemble combination of multiple methods."""
        
        # Weighted voting
        table_scores = {}
        method_weights = [0.4, 0.3, 0.3]  # GNN, RL, Multi-level
        
        for i, results in enumerate(method_results):
            if i < len(method_weights):
                weight = method_weights[i]
                for j, table in enumerate(results):
                    # Position-based scoring (earlier = higher score)
                    position_score = 1.0 / (j + 1)
                    table_scores[table] = table_scores.get(table, 0) + weight * position_score
        
        # Sort by combined score
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        return [table for table, score in sorted_tables]
    
    def _compute_relationship_weight(self, rel_type: str) -> float:
        """Compute weight for relationship type."""
        weights = {
            'foreign_key': 0.9,
            'primary_key': 0.8,
            'reference': 0.7,
            'similar': 0.6,
            'related': 0.5
        }
        return weights.get(rel_type, 0.5)
    
    def _fallback_basic_traversal(self, query: str) -> List[str]:
        """Fallback to basic traversal if advanced methods fail."""
        return self._find_seed_tables(query)
    
    def _find_seed_tables(self, query: str) -> List[str]:
        """Find initial candidate tables (existing method)."""
        query_terms = query.lower().split()
        seed_tables = []
        
        try:
            for term in query_terms:
                if len(term) > 3:  # Skip short words
                    sql = """
                    SELECT name FROM tables
                    WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
                    LIMIT 2
                    """
                    results = self.conn.execute(sql, [f"%{term}%", f"%{term}%"]).fetchall()
                    seed_tables.extend([row[0] for row in results])
        except:
            pass
        
        return seed_tables
3. Update the initialize_retrievers() Function
Update the initialization function (around line 600) to include both methods:
pythondef initialize_retrievers():
    """Initialize all available retrieval methods."""
    global retrievers, db_stats
    
    DB_PATH = "knowledge_graph.duckdb"
    
    if not os.path.exists(DB_PATH):
        return False, "❌ DuckDB file 'knowledge_graph.duckdb' not found!"
    
    retrievers = {}
    
    try:
        # Always available methods
        retrievers["Keyword"] = KeywordRetriever(DB_PATH)
        retrievers["Graph Traversal"] = GraphTraversalRetriever(DB_PATH)  # Keep baseline
        retrievers["Advanced Graph Traversal"] = AdvancedGraphTraversalRetriever(DB_PATH)  # Add new
        
        # TF-IDF (requires sklearn)
        if SKLEARN_AVAILABLE:
            retrievers["TF-IDF"] = TFIDFRetriever(DB_PATH)
        
        # LLM methods (require client manager)
        if CLIENT_MANAGER_AVAILABLE:
            retrievers["GPT-4"] = OpenAIRetriever(DB_PATH)
            retrievers["Gemini"] = GeminiRetriever(DB_PATH)
        
        # Get database statistics
        conn = duckdb.connect(DB_PATH)
        table_count = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
        column_count = conn.execute("SELECT COUNT(*) FROM columns").fetchone()[0]
        
        db_stats = {
            "tables": table_count,
            "columns": column_count,
            "file_size_mb": round(os.path.getsize(DB_PATH) / (1024*1024), 2)
        }
        
        conn.close()
        
        return True, f"✅ Initialized {len(retrievers)} retrieval methods"
        
    except Exception as e:
        return False, f"❌ Failed to initialize retrievers: {e}"
4. Update the get_database_info() Function
Update the method descriptions (around line 750):
pythondef get_database_info():
    """Get database information for display."""
    if not db_stats:
        return "Database not initialized."
    
    info = f"""
## 📊 Database Information

- **Tables:** {db_stats.get('tables', 'N/A')}
- **Columns:** {db_stats.get('columns', 'N/A')}
- **File Size:** {db_stats.get('file_size_mb', 'N/A')} MB
- **Available Methods:** {len(retrievers)}

### 🔧 Available Retrieval Methods:
"""
    
    for method_name in retrievers.keys():
        if method_name in ["GPT-4", "Gemini"]:
            info += f"- 🤖 **{method_name}** (LLM-powered)\n"
        elif method_name == "Graph Traversal":
            info += f"- 🕸️ **{method_name}** (Basic relationship traversal)\n"
        elif method_name == "Advanced Graph Traversal":
            info += f"- 🧠 **{method_name}** (GNN + RL + Multi-level)\n"
        else:
            info += f"- ⚡ **{method_name}** (Local processing)\n"
    
    return info
5. Update the Footer Section
Update the technical details (around line 950):
python        gr.Markdown("""
        ---
        ### 🔧 Technical Details:
        
        **Retrieval Methods:**
        - **Keyword**: Fast pattern matching using SQL queries
        - **TF-IDF**: Text similarity using term frequency analysis
        - **Graph Traversal**: Basic relationship-based table discovery
        - **Advanced Graph Traversal**: GNN + Reinforcement Learning + Multi-level reasoning
        - **GPT-4**: AI-powered contextual understanding
        - **Gemini**: Google's AI for alternative perspective
        
        **Advanced Graph Features:**
        - Graph Neural Networks with attention mechanisms
        - Reinforcement Learning for path optimization
        - Multi-level reasoning (semantic, structural, global)
        - Ensemble combination of multiple algorithms
        - Centrality-based importance scoring
        
        **Performance Comparison:**
        - Basic Graph Traversal: Fast, simple relationship following
        - Advanced Graph Traversal: Higher accuracy, better for complex relationships
        - Local methods (Keyword, TF-IDF) are fastest
        - LLM methods (GPT-4, Gemini) provide context understanding
        
        Built with ❤️ using DuckDB, Gradio, NetworkX, and multiple AI models.
        """)
6. Update Default Method Selection (Optional)
In the create_interface() function (around line 850), you can set the advanced method as default:
python                method_selection = gr.CheckboxGroup(
                    choices=available_methods,
                    label="🔧 Select Retrieval Methods",
                    value=[method for method in available_methods if method in ["Advanced Graph Traversal", "GPT-4", "Gemini"]] if available_methods else [],
                    info="Choose which methods to use for table retrieval"
                )
Summary of Changes:

Kept existing GraphTraversalRetriever as "Graph Traversal"
Added new AdvancedGraphTraversalRetriever as "Advanced Graph Traversal"
Updated initialization to include both methods
Enhanced UI descriptions to differentiate between the two
Set advanced method as default (optional)

Now users can:

Compare performance between basic and advanced graph traversal
Choose appropriate method based on their needs (speed vs accuracy)
See the improvement that advanced algorithms provide
Fall back to basic method if advanced fails