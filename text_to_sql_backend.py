#!/usr/bin/env python3
"""
Text-to-SQL Backend Implementation

Backend classes and functions for the text-to-SQL web UI.
Includes Advanced Graph Traversal, SQL Generation, and SQL Evaluation.
"""

import asyncio
import datetime
import json
import logging
import os
import time
import uuid
import re
from typing import Dict, List, Optional, Tuple, Any
import duckdb
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import client manager if available
try:
    from app.utils.client_manager import client_manager
    CLIENT_MANAGER_AVAILABLE = True
    logger.info("✅ Client manager imported successfully")
except ImportError:
    CLIENT_MANAGER_AVAILABLE = False
    logger.warning("⚠️ Client manager not available - GPT features will be limited")

# Database configuration
DB_PATH = "knowledge_graph.duckdb"

# === Advanced Graph Traversal Implementation ===

class AdvancedGraphTraversalRetriever:
    """Advanced graph traversal using GNN + RL + Multi-level reasoning for table retrieval."""
    
    def __init__(self, db_path: str = DB_PATH):
        try:
            self.conn = duckdb.connect(db_path)
            # Test connection
            table_count = self.conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
            logger.info(f"✅ Connected to DuckDB: {table_count} tables available")
            
            self.graph_structure = self._build_enhanced_graph()
            self.table_embeddings = self._compute_table_embeddings()
            self.centrality_cache = {}
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to DuckDB: {e}")
            raise Exception(f"Database connection failed: {str(e)}")
    
    def get_tables_with_details(self, query: str, max_tables: int = 10) -> Tuple[List[str], Dict]:
        """Get tables with detailed information for UI display."""
        logger.info(f"🧠 Advanced Graph Traversal search for: '{query[:50]}...'")
        
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
            
            # Get detailed information for top tables
            selected_tables = final_results[:max_tables]
            table_details = self._get_table_details(selected_tables)
            
            logger.info(f"📊 Found {len(selected_tables)} tables using Advanced Graph Traversal")
            return selected_tables, table_details
            
        except Exception as e:
            logger.error(f"Advanced graph traversal failed: {e}")
            # Fallback to basic search
            fallback_tables = self._fallback_basic_traversal(query)
            table_details = self._get_table_details(fallback_tables[:max_tables])
            return fallback_tables[:max_tables], table_details
    
    def _build_enhanced_graph(self):
        """Build enhanced graph structure with weights."""
        try:
            G = nx.DiGraph()
            
            # Add tables as nodes
            tables_sql = "SELECT name, description, table_type FROM tables"
            tables = self.conn.execute(tables_sql).fetchall()
            
            for table_name, description, table_type in tables:
                G.add_node(table_name, 
                          description=description or "", 
                          table_type=table_type or "unknown")
            
            # Add relationships as edges (if relationships table exists)
            try:
                rels_sql = """
                SELECT from_table, to_table, relationship_type 
                FROM relationships 
                WHERE from_table != to_table
                """
                relationships = self.conn.execute(rels_sql).fetchall()
                
                for from_table, to_table, rel_type in relationships:
                    if G.has_node(from_table) and G.has_node(to_table):
                        weight = self._compute_relationship_weight(rel_type)
                        G.add_edge(from_table, to_table, weight=weight, rel_type=rel_type)
                
                logger.info(f"📊 Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges")
            except:
                logger.warning("No relationships table found, using table-only graph")
            
            return G
            
        except Exception as e:
            logger.warning(f"Failed to build enhanced graph: {e}")
            return nx.DiGraph()
    
    def _compute_table_embeddings(self):
        """Compute table embeddings using description and column info."""
        embeddings = {}
        
        try:
            for node in self.graph_structure.nodes():
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
        
        query_embedding = self._encode_query(query)
        attention_scores = {}
        
        for table_name in self.graph_structure.nodes():
            try:
                table_features = self.table_embeddings.get(table_name, np.random.rand(10) * 0.1)
                attention_score = self._compute_attention(query_embedding, table_features)
                
                # Get neighbor information
                neighbors = list(self.graph_structure.neighbors(table_name))
                neighbor_scores = []
                
                for neighbor in neighbors:
                    neighbor_features = self.table_embeddings.get(neighbor, np.random.rand(10) * 0.1)
                    neighbor_score = self._compute_attention(query_embedding, neighbor_features)
                    neighbor_scores.append(neighbor_score)
                
                # Aggregate attention scores
                if neighbor_scores:
                    aggregated_score = attention_score * 0.7 + np.mean(neighbor_scores) * 0.3
                else:
                    aggregated_score = attention_score
                    
                attention_scores[table_name] = aggregated_score
                
            except Exception as e:
                attention_scores[table_name] = 0.0
        
        # Return sorted tables by attention score
        sorted_tables = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
        return [table for table, score in sorted_tables if score > 0.1]
    
    def _rl_path_optimization(self, query: str, seed_tables: List[str]) -> List[str]:
        """Reinforcement learning for optimal path discovery."""
        if not seed_tables:
            return []
        
        optimal_tables = set(seed_tables)
        
        for seed_table in seed_tables[:3]:  # Limit to top 3 seed tables
            try:
                current_table = seed_table
                
                # Multi-hop traversal with RL-optimized paths
                for hop in range(2):  # Maximum 2 hops
                    candidates = list(self.graph_structure.neighbors(current_table))
                    
                    if not candidates:
                        break
                    
                    # Compute Q-values for each candidate
                    q_values = {}
                    for candidate in candidates:
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
    
    def _get_table_details(self, table_names: List[str]) -> Dict:
        """Get detailed information about tables and their columns."""
        details = {}
        
        for table_name in table_names:
            try:
                # Get table information
                table_sql = """
                SELECT name, description, table_type 
                FROM tables 
                WHERE name = ?
                """
                table_info = self.conn.execute(table_sql, [table_name]).fetchone()
                
                # Get column information
                columns_sql = """
                SELECT name, data_type, description
                FROM columns 
                WHERE table_name = ?
                ORDER BY ordinal_position
                """
                columns = self.conn.execute(columns_sql, [table_name]).fetchall()
                
                if table_info:
                    details[table_name] = {
                        'name': table_info[0],
                        'description': table_info[1] or '',
                        'table_type': table_info[2] or '',
                        'columns': [
                            {
                                'name': col[0],
                                'data_type': col[1],
                                'description': col[2] or ''
                            }
                            for col in columns
                        ]
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to get details for table {table_name}: {e}")
                details[table_name] = {'error': str(e)}
        
        return details
    
    # Helper methods (simplified implementations)
    def _encode_query(self, query: str) -> np.ndarray:
        """Simple query encoding."""
        words = query.lower().split()
        encoding = np.zeros(10)
        for i, word in enumerate(words[:10]):
            encoding[i % 10] += hash(word) % 100 / 100.0
        return encoding / (np.linalg.norm(encoding) + 1e-8)
    
    def _compute_attention(self, query_emb: np.ndarray, table_emb: np.ndarray) -> float:
        """Compute attention score between query and table."""
        try:
            attention = np.dot(query_emb, table_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(table_emb) + 1e-8)
            return max(0, float(attention))
        except:
            return 0.0
    
    def _compute_table_relevance(self, table_name: str, query: str) -> float:
        """Compute relevance between table and query."""
        try:
            desc_sql = "SELECT description FROM tables WHERE name = ?"
            result = self.conn.execute(desc_sql, [table_name]).fetchone()
            
            if result and result[0]:
                description = result[0].lower()
                query_terms = query.lower().split()
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
        query_terms = query.lower().split()
        seed_tables = []
        
        try:
            for term in query_terms:
                if len(term) > 3:  # Skip short words
                    sql = """
                    SELECT name FROM tables
                    WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
                    LIMIT 3
                    """
                    results = self.conn.execute(sql, [f"%{term}%", f"%{term}%"]).fetchall()
                    seed_tables.extend([row[0] for row in results])
        except:
            pass
        
        return seed_tables
    
    def _structural_pattern_analysis(self, query: str, seed_tables: List[str]) -> List[str]:
        """Analyze structural patterns in the graph."""
        pattern_tables = []
        
        try:
            # Find hub tables (high degree centrality)
            hub_tables = self._find_hub_tables()
            
            # Find bridge tables (high betweenness centrality)  
            bridge_tables = self._find_bridge_tables()
            
            # Community-based relevance
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
            community = set([seed_table])
            
            # Add direct neighbors
            neighbors = list(self.graph_structure.neighbors(seed_table))
            community.update(neighbors[:3])
            
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
        return self._semantic_matching(query)

# === SQL Generator Implementation ===

class SQLGenerator:
    """Enhanced SQL generator using latest best practices for text-to-SQL."""
    
    def __init__(self):
        self.conn = duckdb.connect(DB_PATH)
    
    async def generate_sql(self, query: str, tables: List[str], table_details: Dict) -> Tuple[str, str]:
        """Generate SQL using GPT with enhanced prompting strategy."""
        
        # Create enhanced schema context
        schema_context = self._create_schema_context(tables, table_details)
        
        # Generate SQL using latest prompting techniques
        sql_prompt = self._create_enhanced_sql_prompt(query, schema_context)
        
        try:
            if not CLIENT_MANAGER_AVAILABLE:
                # Fallback SQL generation without GPT
                return self._generate_fallback_sql(query, tables), "Fallback SQL generation used (no GPT available)"
            
            # Use your internal client manager to call GPT
            response = await client_manager.ask_gpt(sql_prompt)
            
            # Extract and validate SQL
            sql_code = self._extract_sql_from_response(response)
            
            # Validate SQL syntax
            validation_result = self._validate_sql(sql_code)
            
            if validation_result['is_valid']:
                return sql_code, response
            else:
                # Attempt to fix SQL if validation fails
                fixed_sql = await self._fix_sql(sql_code, validation_result['error'], query)
                return fixed_sql, f"Original response: {response}\n\nFixed SQL due to: {validation_result['error']}"
                
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            # Return fallback SQL
            fallback_sql = self._generate_fallback_sql(query, tables)
            return fallback_sql, f"Error generating SQL, using fallback: {str(e)}"
    
    def _create_schema_context(self, tables: List[str], table_details: Dict) -> str:
        """Create comprehensive schema context for SQL generation."""
        schema_parts = []
        
        for table_name in tables:
            if table_name in table_details:
                details = table_details[table_name]
                
                # Table header
                table_type = details.get('table_type', 'table')
                description = details.get('description', '')
                
                schema_parts.append(f"\n-- {table_name} ({table_type})")
                if description:
                    schema_parts.append(f"-- Description: {description}")
                
                # Column definitions
                columns = details.get('columns', [])
                if columns:
                    schema_parts.append(f"CREATE TABLE {table_name} (")
                    col_definitions = []
                    
                    for col in columns:
                        col_def = f"  {col['name']} {col['data_type']}"
                        
                        # Add description as comment
                        if col.get('description'):
                            col_def += f" -- {col['description']}"
                        
                        col_definitions.append(col_def)
                    
                    schema_parts.append(",\n".join(col_definitions))
                    schema_parts.append(");")
                
                schema_parts.append("")  # Empty line between tables
        
        return "\n".join(schema_parts)
    
    def _create_enhanced_sql_prompt(self, query: str, schema_context: str) -> str:
        """Create enhanced SQL prompt using latest best practices."""
        
        prompt = f"""You are a highly skilled SQL expert specializing in data analysis and query generation. Your task is to write precise, efficient SQL queries based on natural language requests.

**TASK**: Convert the user's natural language question into a syntactically correct and logically sound SQL query.

**DATABASE SCHEMA:**
{schema_context}

**USER QUESTION:**
{query}

**SQL GENERATION GUIDELINES:**

1. **Accuracy First**: 
   - Understand the business intent behind the question
   - Use appropriate tables and columns based on the question context
   - Ensure logical relationships between tables are respected

2. **Best Practices**:
   - Use explicit JOIN syntax (INNER JOIN, LEFT JOIN, etc.)
   - Include table aliases for readability
   - Use appropriate WHERE clauses for filtering
   - Apply GROUP BY when aggregation is needed
   - Use ORDER BY for sorting when implied
   - Include LIMIT when appropriate

3. **Query Structure**:
   - SELECT: Choose relevant columns, use aggregations when needed
   - FROM: Start with the main table containing primary data
   - JOIN: Connect related tables using foreign key relationships
   - WHERE: Filter data based on conditions in the question
   - GROUP BY: Group data when using aggregate functions
   - HAVING: Filter grouped results
   - ORDER BY: Sort results logically
   - LIMIT: Restrict results when appropriate

4. **Common Patterns**:
   - For "show me" queries: SELECT relevant columns
   - For "count" queries: Use COUNT() with appropriate GROUP BY
   - For "highest/lowest" queries: Use ORDER BY with LIMIT
   - For "average/sum" queries: Use AVG()/SUM() functions
   - For date-based queries: Use date functions and filters

5. **Error Prevention**:
   - Verify column names exist in the specified tables
   - Ensure JOIN conditions are valid
   - Use proper data type comparisons
   - Handle NULL values appropriately

**RESPONSE FORMAT:**
Return ONLY the SQL query without any explanations, comments, or markdown formatting. The query should be ready to execute.

**SQL QUERY:**"""

        return prompt
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL code from GPT response."""
        # Remove markdown code blocks
        response = re.sub(r'```sql\n?', '', response, flags=re.IGNORECASE)
        response = re.sub(r'```\n?', '', response)
        
        # Remove common prefixes
        response = re.sub(r'^(SQL QUERY:|Query:|Answer:)\s*', '', response, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up the response
        response = response.strip()
        
        # If multiple queries, take the first one
        queries = response.split(';')
        sql_code = queries[0].strip()
        
        return sql_code
    
    def _validate_sql(self, sql: str) -> Dict:
        """Validate SQL syntax using DuckDB."""
        try:
            # Try to prepare the statement (syntax check without execution)
            self.conn.execute(f"EXPLAIN {sql}")
            return {'is_valid': True, 'error': None}
        except Exception as e:
            return {'is_valid': False, 'error': str(e)}
    
    async def _fix_sql(self, sql: str, error: str, original_query: str) -> str:
        """Attempt to fix SQL based on validation error."""
        if not CLIENT_MANAGER_AVAILABLE:
            return sql  # Return original if no GPT available
        
        fix_prompt = f"""The following SQL query has a syntax error. Please fix it and return only the corrected SQL.

**Original Question:** {original_query}

**SQL with Error:**
{sql}

**Error Message:**
{error}

**Instructions:**
- Fix the syntax error while preserving the original intent
- Return only the corrected SQL query
- Do not include explanations or markdown

**Corrected SQL:**"""

        try:
            response = await client_manager.ask_gpt(fix_prompt)
            return self._extract_sql_from_response(response)
        except Exception as e:
            logger.error(f"SQL fixing failed: {e}")
            return sql  # Return original if fixing fails
    
    def _generate_fallback_sql(self, query: str, tables: List[str]) -> str:
        """Generate a basic fallback SQL when GPT is not available."""
        # Simple fallback - basic SELECT statement
        if tables:
            main_table = tables[0]
            return f"SELECT * FROM {main_table} LIMIT 10;"
        else:
            return "-- No tables found for query"

# === SQL Evaluation Implementation ===

class SQLEvaluator:
    """SQL quality evaluation using multiple metrics and GPT-4 assessment."""
    
    def __init__(self):
        self.conn = duckdb.connect(DB_PATH)
    
    async def evaluate_sql_quality(self, generated_sql: str, ground_truth_sql: str, 
                                 original_query: str) -> Dict[str, Any]:
        """
        Comprehensive SQL quality evaluation using multiple metrics.
        
        Based on latest research in text-to-SQL evaluation:
        - Execution Accuracy (EX): Whether results match exactly
        - Logical Form Accuracy (LF): Whether SQL structure is equivalent
        - Partial Component Matching (PCM): Component-wise evaluation
        - GPT-4 Semantic Assessment: AI-powered quality scoring
        """
        
        evaluation_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'generated_sql': generated_sql,
            'ground_truth_sql': ground_truth_sql,
            'original_query': original_query
        }
        
        try:
            # 1. Execution Accuracy (EX)
            ex_result = await self._execution_accuracy(generated_sql, ground_truth_sql)
            evaluation_results['execution_accuracy'] = ex_result
            
            # 2. Logical Form Accuracy (LF) 
            lf_result = await self._logical_form_accuracy(generated_sql, ground_truth_sql)
            evaluation_results['logical_form_accuracy'] = lf_result
            
            # 3. Partial Component Matching (PCM)
            pcm_result = await self._partial_component_matching(generated_sql, ground_truth_sql)
            evaluation_results['partial_component_matching'] = pcm_result
            
            # 4. GPT-4 Semantic Assessment
            if CLIENT_MANAGER_AVAILABLE:
                gpt_result = await self._gpt4_semantic_assessment(
                    generated_sql, ground_truth_sql, original_query
                )
                evaluation_results['gpt4_assessment'] = gpt_result
            else:
                evaluation_results['gpt4_assessment'] = {
                    'available': False,
                    'reason': 'Client manager not available'
                }
            
            # 5. Calculate overall quality score
            overall_score = self._calculate_overall_score(evaluation_results)
            evaluation_results['overall_score'] = overall_score
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"SQL evaluation failed: {e}")
            evaluation_results['error'] = str(e)
            evaluation_results['overall_score'] = 0.0
            return evaluation_results
    
    async def _execution_accuracy(self, generated_sql: str, ground_truth_sql: str) -> Dict[str, Any]:
        """Execute both SQLs and compare results."""
        try:
            # Execute generated SQL
            try:
                gen_results = self.conn.execute(generated_sql).fetchall()
                gen_execution_success = True
                gen_error = None
            except Exception as e:
                gen_results = []
                gen_execution_success = False
                gen_error = str(e)
            
            # Execute ground truth SQL
            try:
                gt_results = self.conn.execute(ground_truth_sql).fetchall()
                gt_execution_success = True
                gt_error = None
            except Exception as e:
                gt_results = []
                gt_execution_success = False
                gt_error = str(e)
            
            # Compare results
            if gen_execution_success and gt_execution_success:
                # Convert to sets for comparison (order-independent)
                gen_set = set(gen_results) if gen_results else set()
                gt_set = set(gt_results) if gt_results else set()
                
                exact_match = gen_set == gt_set
                
                # Calculate similarity metrics
                if gt_set:
                    intersection = len(gen_set.intersection(gt_set))
                    union = len(gen_set.union(gt_set))
                    precision = intersection / len(gen_set) if gen_set else 0.0
                    recall = intersection / len(gt_set) if gt_set else 0.0
                    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    jaccard = intersection / union if union > 0 else 0.0
                else:
                    precision = recall = f1_score = jaccard = 0.0
                
                return {
                    'exact_match': exact_match,
                    'generated_execution_success': gen_execution_success,
                    'ground_truth_execution_success': gt_execution_success,
                    'generated_row_count': len(gen_results),
                    'ground_truth_row_count': len(gt_results),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'jaccard_similarity': jaccard,
                    'score': f1_score  # Use F1 as primary score
                }
            else:
                return {
                    'exact_match': False,
                    'generated_execution_success': gen_execution_success,
                    'ground_truth_execution_success': gt_execution_success,
                    'generated_error': gen_error,
                    'ground_truth_error': gt_error,
                    'score': 0.0
                }
                
        except Exception as e:
            return {
                'exact_match': False,
                'error': str(e),
                'score': 0.0
            }
    
    async def _logical_form_accuracy(self, generated_sql: str, ground_truth_sql: str) -> Dict[str, Any]:
        """Compare SQL logical structure and components."""
        try:
            gen_components = self._parse_sql_components(generated_sql)
            gt_components = self._parse_sql_components(ground_truth_sql)
            
            # Component-wise comparison
            component_scores = {}
            
            for component in ['select', 'from', 'where', 'join', 'group_by', 'having', 'order_by']:
                gen_comp = gen_components.get(component, set())
                gt_comp = gt_components.get(component, set())
                
                if gt_comp:
                    intersection = len(gen_comp.intersection(gt_comp))
                    union = len(gen_comp.union(gt_comp))
                    component_scores[component] = intersection / len(gt_comp) if gt_comp else 0.0
                else:
                    component_scores[component] = 1.0 if not gen_comp else 0.0
            
            # Overall logical form score
            lf_score = sum(component_scores.values()) / len(component_scores)
            
            return {
                'component_scores': component_scores,
                'logical_form_score': lf_score,
                'generated_components': {k: list(v) for k, v in gen_components.items()},
                'ground_truth_components': {k: list(v) for k, v in gt_components.items()},
                'score': lf_score
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'score': 0.0
            }
    
    async def _partial_component_matching(self, generated_sql: str, ground_truth_sql: str) -> Dict[str, Any]:
        """Evaluate partial matching of SQL components."""
        try:
            gen_tokens = self._tokenize_sql(generated_sql)
            gt_tokens = self._tokenize_sql(ground_truth_sql)
            
            # Keywords matching
            sql_keywords = {'SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP', 'BY', 'ORDER', 'HAVING', 'INNER', 'LEFT', 'RIGHT', 'OUTER'}
            gen_keywords = set(token.upper() for token in gen_tokens if token.upper() in sql_keywords)
            gt_keywords = set(token.upper() for token in gt_tokens if token.upper() in sql_keywords)
            
            keyword_precision = len(gen_keywords.intersection(gt_keywords)) / len(gen_keywords) if gen_keywords else 0.0
            keyword_recall = len(gen_keywords.intersection(gt_keywords)) / len(gt_keywords) if gt_keywords else 0.0
            
            # Table names matching
            gen_tables = self._extract_table_names(generated_sql)
            gt_tables = self._extract_table_names(ground_truth_sql)
            
            table_precision = len(gen_tables.intersection(gt_tables)) / len(gen_tables) if gen_tables else 0.0
            table_recall = len(gen_tables.intersection(gt_tables)) / len(gt_tables) if gt_tables else 0.0
            
            # Column names matching (simplified)
            gen_columns = self._extract_column_references(generated_sql)
            gt_columns = self._extract_column_references(ground_truth_sql)
            
            column_precision = len(gen_columns.intersection(gt_columns)) / len(gen_columns) if gen_columns else 0.0
            column_recall = len(gen_columns.intersection(gt_columns)) / len(gt_columns) if gt_columns else 0.0
            
            # Calculate F1 scores
            keyword_f1 = 2 * keyword_precision * keyword_recall / (keyword_precision + keyword_recall) if (keyword_precision + keyword_recall) > 0 else 0.0
            table_f1 = 2 * table_precision * table_recall / (table_precision + table_recall) if (table_precision + table_recall) > 0 else 0.0
            column_f1 = 2 * column_precision * column_recall / (column_precision + column_recall) if (column_precision + column_recall) > 0 else 0.0
            
            # Overall PCM score
            pcm_score = (keyword_f1 + table_f1 + column_f1) / 3
            
            return {
                'keyword_metrics': {
                    'precision': keyword_precision,
                    'recall': keyword_recall,
                    'f1': keyword_f1
                },
                'table_metrics': {
                    'precision': table_precision,
                    'recall': table_recall,
                    'f1': table_f1
                },
                'column_metrics': {
                    'precision': column_precision,
                    'recall': column_recall,
                    'f1': column_f1
                },
                'pcm_score': pcm_score,
                'score': pcm_score
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'score': 0.0
            }
    
    async def _gpt4_semantic_assessment(self, generated_sql: str, ground_truth_sql: str, 
                                      original_query: str) -> Dict[str, Any]:
        """Use GPT-4 to assess semantic equivalence and quality."""
        
        assessment_prompt = f"""You are an expert SQL analyst. Your task is to evaluate the quality and correctness of a generated SQL query against a ground truth SQL query.

**Original Natural Language Query:**
{original_query}

**Generated SQL:**
{generated_sql}

**Ground Truth SQL:**
{ground_truth_sql}

**Evaluation Criteria:**

1. **Semantic Equivalence (0-100)**: Do both queries answer the same question and would produce equivalent results?

2. **Syntactic Correctness (0-100)**: Is the generated SQL syntactically correct and executable?

3. **Logical Structure (0-100)**: Does the generated SQL follow proper logical structure (appropriate JOINs, WHERE clauses, etc.)?

4. **Efficiency (0-100)**: Is the generated SQL reasonably efficient compared to the ground truth?

5. **Completeness (0-100)**: Does the generated SQL address all aspects of the original query?

**Response Format (JSON):**
{{
  "semantic_equivalence": <score 0-100>,
  "syntactic_correctness": <score 0-100>,
  "logical_structure": <score 0-100>,
  "efficiency": <score 0-100>,
  "completeness": <score 0-100>,
  "overall_quality": <average score 0-100>,
  "strengths": ["list of strengths"],
  "weaknesses": ["list of weaknesses"],
  "explanation": "detailed explanation of the assessment"
}}

**Assessment:**"""

        try:
            response = await client_manager.ask_gpt(assessment_prompt)
            
            # Try to parse JSON response
            try:
                # Clean the response to extract JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    assessment = json.loads(json_str)
                else:
                    # Fallback parsing
                    assessment = self._parse_gpt_response_fallback(response)
                
                # Normalize scores to 0-1 range
                normalized_assessment = {}
                for key, value in assessment.items():
                    if key in ['semantic_equivalence', 'syntactic_correctness', 'logical_structure', 'efficiency', 'completeness', 'overall_quality']:
                        normalized_assessment[key] = value / 100.0 if isinstance(value, (int, float)) else 0.0
                    else:
                        normalized_assessment[key] = value
                
                normalized_assessment['score'] = normalized_assessment.get('overall_quality', 0.0)
                normalized_assessment['raw_response'] = response
                
                return normalized_assessment
                
            except json.JSONDecodeError:
                # Fallback to simple parsing
                return self._parse_gpt_response_fallback(response)
                
        except Exception as e:
            logger.error(f"GPT-4 assessment failed: {e}")
            return {
                'error': str(e),
                'score': 0.0
            }
    
    def _parse_gpt_response_fallback(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for GPT response."""
        try:
            # Extract scores using regex
            scores = {}
            score_patterns = [
                (r'semantic_equivalence[:\s]+(\d+)', 'semantic_equivalence'),
                (r'syntactic_correctness[:\s]+(\d+)', 'syntactic_correctness'),
                (r'logical_structure[:\s]+(\d+)', 'logical_structure'),
                (r'efficiency[:\s]+(\d+)', 'efficiency'),
                (r'completeness[:\s]+(\d+)', 'completeness'),
                (r'overall_quality[:\s]+(\d+)', 'overall_quality')
            ]
            
            for pattern, key in score_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    scores[key] = float(match.group(1)) / 100.0
                else:
                    scores[key] = 0.0
            
            scores['score'] = scores.get('overall_quality', 0.0)
            scores['explanation'] = response
            scores['fallback_parsing'] = True
            
            return scores
            
        except Exception as e:
            return {
                'error': f"Fallback parsing failed: {e}",
                'score': 0.0
            }
    
    def _calculate_overall_score(self, evaluation_results: Dict[str, Any]) -> float:
        """Calculate weighted overall quality score."""
        try:
            weights = {
                'execution_accuracy': 0.4,
                'logical_form_accuracy': 0.25,
                'partial_component_matching': 0.2,
                'gpt4_assessment': 0.15
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in evaluation_results and 'score' in evaluation_results[metric]:
                    weighted_score += evaluation_results[metric]['score'] * weight
                    total_weight += weight
            
            return weighted_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Overall score calculation failed: {e}")
            return 0.0
    
    def _parse_sql_components(self, sql: str) -> Dict[str, set]:
        """Parse SQL into logical components."""
        try:
            sql = sql.upper().strip()
            components = {}
            
            # Extract SELECT clause
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.DOTALL)
            if select_match:
                select_items = [item.strip() for item in select_match.group(1).split(',')]
                components['select'] = set(select_items)
            else:
                components['select'] = set()
            
            # Extract FROM clause
            from_match = re.search(r'FROM\s+([^\s]+)', sql)
            if from_match:
                components['from'] = {from_match.group(1)}
            else:
                components['from'] = set()
            
            # Extract WHERE clause
            where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|$)', sql, re.DOTALL)
            if where_match:
                components['where'] = {where_match.group(1).strip()}
            else:
                components['where'] = set()
            
            # Extract JOIN clauses
            join_matches = re.findall(r'((?:INNER|LEFT|RIGHT|OUTER|FULL)?\s*JOIN\s+[^\s]+(?:\s+ON\s+[^WHERE^GROUP^ORDER^HAVING]+)?)', sql)
            components['join'] = set(join_matches)
            
            # Extract GROUP BY clause
            group_by_match = re.search(r'GROUP\s+BY\s+(.*?)(?:\s+HAVING|\s+ORDER\s+BY|$)', sql, re.DOTALL)
            if group_by_match:
                group_items = [item.strip() for item in group_by_match.group(1).split(',')]
                components['group_by'] = set(group_items)
            else:
                components['group_by'] = set()
            
            # Extract HAVING clause
            having_match = re.search(r'HAVING\s+(.*?)(?:\s+ORDER\s+BY|$)', sql, re.DOTALL)
            if having_match:
                components['having'] = {having_match.group(1).strip()}
            else:
                components['having'] = set()
            
            # Extract ORDER BY clause
            order_by_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)', sql, re.DOTALL)
            if order_by_match:
                order_items = [item.strip() for item in order_by_match.group(1).split(',')]
                components['order_by'] = set(order_items)
            else:
                components['order_by'] = set()
            
            return components
            
        except Exception as e:
            logger.warning(f"SQL component parsing failed: {e}")
            return {}
    
    def _tokenize_sql(self, sql: str) -> List[str]:
        """Simple SQL tokenization."""
        # Remove comments and extra whitespace
        sql = re.sub(r'--.*?\n', ' ', sql)
        sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
        sql = re.sub(r'\s+', ' ', sql)
        
        # Split by common delimiters but preserve them
        tokens = re.findall(r'\w+|[^\w\s]', sql)
        return [token for token in tokens if token.strip()]
    
    def _extract_table_names(self, sql: str) -> set:
        """Extract table names from SQL."""
        try:
            sql = sql.upper()
            # Find FROM and JOIN clauses
            table_patterns = [
                r'FROM\s+([^\s,\)]+)',
                r'JOIN\s+([^\s,\)]+)',
                r'UPDATE\s+([^\s,\)]+)',
                r'INTO\s+([^\s,\)]+)'
            ]
            
            tables = set()
            for pattern in table_patterns:
                matches = re.findall(pattern, sql)
                for match in matches:
                    # Remove alias if present
                    table_name = match.split()[0]
                    tables.add(table_name)
            
            return tables
            
        except Exception as e:
            logger.warning(f"Table name extraction failed: {e}")
            return set()
    
    def _extract_column_references(self, sql: str) -> set:
        """Extract column references from SQL (simplified)."""
        try:
            # This is a simplified extraction - in practice, you'd want more sophisticated parsing
            sql = sql.upper()
            
            # Remove string literals and comments
            sql = re.sub(r"'[^']*'", '', sql)
            sql = re.sub(r'"[^"]*"', '', sql)
            sql = re.sub(r'--.*?\n', ' ', sql)
            
            columns = set()
            
            # Extract from SELECT clause
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.DOTALL)
            if select_match:
                select_items = select_match.group(1).split(',')
                for item in select_items:
                    item = item.strip()
                    # Extract column name (remove functions, aliases, etc.)
                    col_match = re.search(r'([A-Z_][A-Z0-9_]*)', item)
                    if col_match and col_match.group(1) not in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'DISTINCT']:
                        columns.add(col_match.group(1))
            
            # Extract from WHERE clause
            where_matches = re.findall(r'([A-Z_][A-Z0-9_]*)\s*[=<>!]', sql)
            for match in where_matches:
                if match not in ['AND', 'OR', 'NOT', 'IN', 'EXISTS']:
                    columns.add(match)
            
            return columns
            
        except Exception as e:
            logger.warning(f"Column reference extraction failed: {e}")
            return set()

# === Main Backend Service Class ===

class TextToSQLBackend:
    """Main backend service that coordinates all components."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.graph_retriever = None
        self.sql_generator = None
        self.sql_evaluator = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all backend components."""
        try:
            self.graph_retriever = AdvancedGraphTraversalRetriever(self.db_path)
            self.sql_generator = SQLGenerator()
            self.sql_evaluator = SQLEvaluator()
            logger.info("✅ All backend components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize backend components: {e}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database statistics and information."""
        try:
            conn = duckdb.connect(self.db_path)
            
            # Get table statistics
            table_count = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
            column_count = conn.execute("SELECT COUNT(*) FROM columns").fetchone()[0]
            
            # Get table types
            table_types = conn.execute("""
                SELECT table_type, COUNT(*) as count 
                FROM tables 
                GROUP BY table_type
            """).fetchall()
            
            # Get relationships if available
            try:
                relationship_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            except:
                relationship_count = 0
            
            conn.close()
            
            return {
                'database_path': self.db_path,
                'table_count': table_count,
                'column_count': column_count,
                'relationship_count': relationship_count,
                'table_types': {table_type: count for table_type, count in table_types},
                'file_size_mb': round(os.path.getsize(self.db_path) / (1024*1024), 2) if os.path.exists(self.db_path) else 0,
                'client_manager_available': CLIENT_MANAGER_AVAILABLE
            }
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {'error': str(e)}
    
    async def find_relevant_tables(self, query: str, max_tables: int = 10) -> Tuple[List[str], Dict, str]:
        """Find relevant tables using Advanced Graph Traversal."""
        try:
            start_time = time.time()
            
            tables, table_details = self.graph_retriever.get_tables_with_details(query, max_tables)
            
            processing_time = time.time() - start_time
            
            status_message = f"✅ Found {len(tables)} relevant tables in {processing_time:.3f}s using Advanced Graph Traversal"
            
            return tables, table_details, status_message
            
        except Exception as e:
            error_message = f"❌ Table discovery failed: {str(e)}"
            logger.error(error_message)
            return [], {}, error_message
    
    async def generate_sql_from_tables(self, query: str, selected_tables: List[str], 
                                     table_details: Dict) -> Tuple[str, str, str]:
        """Generate SQL from selected tables."""
        try:
            start_time = time.time()
            
            sql_code, reasoning = await self.sql_generator.generate_sql(
                query, selected_tables, table_details
            )
            
            processing_time = time.time() - start_time
            
            # Validate the generated SQL
            validation = self.sql_generator._validate_sql(sql_code)
            if validation['is_valid']:
                status_message = f"✅ SQL generated successfully in {processing_time:.3f}s"
            else:
                status_message = f"⚠️ SQL generated in {processing_time:.3f}s but validation failed: {validation['error']}"
            
            return sql_code, reasoning, status_message
            
        except Exception as e:
            error_message = f"❌ SQL generation failed: {str(e)}"
            logger.error(error_message)
            return "", f"Error: {str(e)}", error_message
    
    async def evaluate_sql_quality(self, generated_sql: str, ground_truth_sql: str, 
                                 original_query: str) -> Tuple[Dict[str, Any], str]:
        """Evaluate SQL quality against ground truth."""
        try:
            start_time = time.time()
            
            evaluation_results = await self.sql_evaluator.evaluate_sql_quality(
                generated_sql, ground_truth_sql, original_query
            )
            
            processing_time = time.time() - start_time
            
            overall_score = evaluation_results.get('overall_score', 0.0)
            status_message = f"✅ SQL evaluation completed in {processing_time:.3f}s - Overall score: {overall_score:.2f}"
            
            return evaluation_results, status_message
            
        except Exception as e:
            error_message = f"❌ SQL evaluation failed: {str(e)}"
            logger.error(error_message)
            return {'error': str(e), 'overall_score': 0.0}, error_message

# === Utility Functions ===

def check_system_requirements() -> Tuple[bool, str]:
    """Check if all system requirements are met."""
    try:
        # Check if database file exists
        if not os.path.exists(DB_PATH):
            return False, f"❌ Database file '{DB_PATH}' not found"
        
        # Test database connection
        conn = duckdb.connect(DB_PATH)
        table_count = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
        conn.close()
        
        if table_count == 0:
            return False, "❌ Database has no tables"
        
        # Check client manager availability
        client_status = "✅ Available" if CLIENT_MANAGER_AVAILABLE else "⚠️ Not available (limited functionality)"
        
        return True, f"✅ System ready - {table_count} tables in database, Client manager: {client_status}"
        
    except Exception as e:
        return False, f"❌ System check failed: {str(e)}"

def format_table_details_for_display(table_details: Dict) -> str:
    """Format table details for UI display."""
    if not table_details:
        return "No table details available."
    
    formatted_text = ""
    
    for table_name, details in table_details.items():
        if 'error' in details:
            formatted_text += f"\n## ❌ {table_name}\n**Error:** {details['error']}\n"
            continue
        
        formatted_text += f"\n## 📋 {table_name}\n"
        
        # Table info
        table_type = details.get('table_type', 'unknown')
        description = details.get('description', '')
        
        formatted_text += f"**Type:** {table_type}\n"
        if description:
            formatted_text += f"**Description:** {description}\n"
        
        # Columns
        columns = details.get('columns', [])
        if columns:
            formatted_text += f"**Columns ({len(columns)}):**\n"
            for col in columns:
                col_name = col.get('name', '')
                col_type = col.get('data_type', '')
                col_desc = col.get('description', '')
                
                formatted_text += f"- **{col_name}** ({col_type})"
                if col_desc:
                    formatted_text += f": {col_desc}"
                formatted_text += "\n"
        
        formatted_text += "\n---\n"
    
    return formatted_text

def format_evaluation_results_for_display(evaluation_results: Dict[str, Any]) -> str:
    """Format evaluation results for UI display."""
    if not evaluation_results or 'error' in evaluation_results:
        return f"❌ Evaluation failed: {evaluation_results.get('error', 'Unknown error')}"
    
    formatted_text = f"# 📊 SQL Quality Evaluation Results\n\n"
    
    # Overall score
    overall_score = evaluation_results.get('overall_score', 0.0)
    score_emoji = "🟢" if overall_score >= 0.8 else "🟡" if overall_score >= 0.5 else "🔴"
    formatted_text += f"## {score_emoji} Overall Quality Score: {overall_score:.2f} ({overall_score*100:.1f}%)\n\n"
    
    # Individual metrics
    formatted_text += "## 📈 Detailed Metrics:\n\n"
    
    # Execution Accuracy
    if 'execution_accuracy' in evaluation_results:
        ex_data = evaluation_results['execution_accuracy']
        ex_score = ex_data.get('score', 0.0)
        exact_match = ex_data.get('exact_match', False)
        
        formatted_text += f"### 🎯 Execution Accuracy: {ex_score:.2f}\n"
        formatted_text += f"- **Exact Match:** {'✅ Yes' if exact_match else '❌ No'}\n"
        
        if 'f1_score' in ex_data:
            formatted_text += f"- **F1 Score:** {ex_data['f1_score']:.3f}\n"
            formatted_text += f"- **Precision:** {ex_data['precision']:.3f}\n"
            formatted_text += f"- **Recall:** {ex_data['recall']:.3f}\n"
        
        if 'generated_row_count' in ex_data and 'ground_truth_row_count' in ex_data:
            formatted_text += f"- **Generated Rows:** {ex_data['generated_row_count']}\n"
            formatted_text += f"- **Expected Rows:** {ex_data['ground_truth_row_count']}\n"
        
        formatted_text += "\n"
    
    # Logical Form Accuracy
    if 'logical_form_accuracy' in evaluation_results:
        lf_data = evaluation_results['logical_form_accuracy']
        lf_score = lf_data.get('score', 0.0)
        
        formatted_text += f"### 🏗️ Logical Form Accuracy: {lf_score:.2f}\n"
        
        if 'component_scores' in lf_data:
            formatted_text += "**Component Scores:**\n"
            for component, score in lf_data['component_scores'].items():
                formatted_text += f"- **{component.replace('_', ' ').title()}:** {score:.3f}\n"
        
        formatted_text += "\n"
    
    # Partial Component Matching
    if 'partial_component_matching' in evaluation_results:
        pcm_data = evaluation_results['partial_component_matching']
        pcm_score = pcm_data.get('score', 0.0)
        
        formatted_text += f"### 🧩 Partial Component Matching: {pcm_score:.2f}\n"
        
        for metric_type in ['keyword_metrics', 'table_metrics', 'column_metrics']:
            if metric_type in pcm_data:
                metrics = pcm_data[metric_type]
                metric_name = metric_type.replace('_metrics', '').title()
                formatted_text += f"**{metric_name} Matching:**\n"
                formatted_text += f"- Precision: {metrics.get('precision', 0):.3f}\n"
                formatted_text += f"- Recall: {metrics.get('recall', 0):.3f}\n"
                formatted_text += f"- F1: {metrics.get('f1', 0):.3f}\n"
        
        formatted_text += "\n"
    
    # GPT-4 Assessment
    if 'gpt4_assessment' in evaluation_results:
        gpt_data = evaluation_results['gpt4_assessment']
        
        if gpt_data.get('available', True):
            formatted_text += f"### 🤖 GPT-4 Semantic Assessment\n"
            
            # Individual scores
            score_items = [
                ('semantic_equivalence', 'Semantic Equivalence'),
                ('syntactic_correctness', 'Syntactic Correctness'),
                ('logical_structure', 'Logical Structure'),
                ('efficiency', 'Efficiency'),
                ('completeness', 'Completeness')
            ]
            
            for key, label in score_items:
                if key in gpt_data:
                    score = gpt_data[key]
                    formatted_text += f"- **{label}:** {score:.2f} ({score*100:.0f}%)\n"
            
            # Strengths and weaknesses
            if 'strengths' in gpt_data and gpt_data['strengths']:
                formatted_text += f"\n**Strengths:**\n"
                for strength in gpt_data['strengths']:
                    formatted_text += f"- {strength}\n"
            
            if 'weaknesses' in gpt_data and gpt_data['weaknesses']:
                formatted_text += f"\n**Areas for Improvement:**\n"
                for weakness in gpt_data['weaknesses']:
                    formatted_text += f"- {weakness}\n"
            
            if 'explanation' in gpt_data:
                formatted_text += f"\n**Detailed Analysis:**\n{gpt_data['explanation']}\n"
        else:
            formatted_text += f"### 🤖 GPT-4 Assessment: Not Available\n"
            formatted_text += f"Reason: {gpt_data.get('reason', 'Unknown')}\n"
        
        formatted_text += "\n"
    
    # Summary and recommendations
    formatted_text += "## 💡 Summary & Recommendations\n\n"
    
    if overall_score >= 0.8:
        formatted_text += "🎉 **Excellent!** The generated SQL demonstrates high quality and correctness.\n\n"
    elif overall_score >= 0.6:
        formatted_text += "👍 **Good!** The generated SQL is generally correct with some room for improvement.\n\n"
    elif overall_score >= 0.4:
        formatted_text += "⚠️ **Fair.** The generated SQL has significant issues that should be addressed.\n\n"
    else:
        formatted_text += "🚨 **Poor.** The generated SQL requires major revision or regeneration.\n\n"
    
    # Specific recommendations based on scores
    if 'execution_accuracy' in evaluation_results:
        ex_score = evaluation_results['execution_accuracy'].get('score', 0.0)
        if ex_score < 0.5:
            formatted_text += "🔧 **Recommendation:** Focus on correctness - the generated SQL produces different results than expected.\n\n"
    
    if 'logical_form_accuracy' in evaluation_results:
        lf_score = evaluation_results['logical_form_accuracy'].get('score', 0.0)
        if lf_score < 0.5:
            formatted_text += "🏗️ **Recommendation:** Review SQL structure - improve JOIN conditions, WHERE clauses, and overall logic.\n\n"
    
    formatted_text += f"**Evaluation completed at:** {evaluation_results.get('timestamp', 'Unknown time')}\n"
    
    return formatted_text

# Global backend instance
_backend_instance = None

def get_backend_instance() -> TextToSQLBackend:
    """Get or create the global backend instance."""
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = TextToSQLBackend()
    return _backend_instance

def initialize_backend() -> Tuple[bool, str]:
    """Initialize the backend and return status."""
    try:
        global _backend_instance
        _backend_instance = TextToSQLBackend()
        return True, "✅ Backend initialized successfully"
    except Exception as e:
        return False, f"❌ Backend initialization failed: {str(e)}"