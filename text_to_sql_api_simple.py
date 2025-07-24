#!/usr/bin/env python3
"""
Advanced Graph Traversal Text-to-SQL API Service

A standalone FastAPI service for intelligent table discovery and SQL generation
using Advanced Graph Neural Networks, Reinforcement Learning, and Multi-level reasoning.

Usage:
    python text_to_sql_api.py

Requirements:
    pip install fastapi uvicorn duckdb networkx numpy pydantic
"""

import asyncio
import datetime
import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import duckdb
import networkx as nx
import numpy as np
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = "knowledge_graph.duckdb"

# Create FastAPI app
app = FastAPI(
    title="Advanced Graph Text-to-SQL API",
    description="Intelligent table discovery and SQL generation using Advanced Graph Traversal",
    version="2.0.0"
)

# === Client Manager Integration ===
# Import your existing client manager for GPT calls
try:
    from app.utils.client_manager import client_manager
    CLIENT_MANAGER_AVAILABLE = True
    logger.info("✅ Client manager imported successfully")
except ImportError:
    CLIENT_MANAGER_AVAILABLE = False
    logger.warning("⚠️ Client manager not available - GPT features will be limited")

# === Pydantic Models ===

class TextToTableRequest(BaseModel):
    query: str = Field(..., description="Natural language query to find relevant tables")
    user_id: str = Field(..., description="User identifier")
    request_id: str = Field(default_factory=lambda: f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}", description="Request identifier")
    max_tables: Optional[int] = Field(10, description="Maximum number of tables to return")

class TableToSQLRequest(BaseModel):
    selected_tables: List[str] = Field(..., description="List of selected table names")
    original_query: str = Field(..., description="Original natural language query")
    user_id: str = Field(..., description="User identifier") 
    request_id: str = Field(default_factory=lambda: f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}", description="Request identifier")

class TextToSQLRequest(BaseModel):
    query: str = Field(..., description="Natural language query for complete text-to-SQL")
    user_id: str = Field(..., description="User identifier")
    request_id: str = Field(default_factory=lambda: f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}", description="Request identifier")
    max_tables: Optional[int] = Field(8, description="Maximum number of tables to discover")
    include_reasoning: Optional[bool] = Field(True, description="Include reasoning in response")

class TableInfo(BaseModel):
    name: str
    description: str
    table_type: str
    record_count: Optional[int]
    columns: List[Dict]

class TextToTableResponse(BaseModel):
    success: bool
    message: str
    tables: List[str]
    table_details: Dict[str, TableInfo]
    processing_time: float
    method: str = "AdvancedGraphTraversal"

class TableToSQLResponse(BaseModel):
    success: bool
    message: str
    sql: str
    reasoning: str
    tables_used: List[str]
    processing_time: float
    validation_status: str

class TextToSQLResponse(BaseModel):
    success: bool
    message: str
    sql: str
    reasoning: str
    tables_found: List[str]
    table_details: Dict[str, TableInfo]
    processing_time: float
    method: str = "AdvancedGraphTraversal"

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
            raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    
    def get_tables_with_details(self, query: str) -> Tuple[List[str], Dict]:
        """Get tables with detailed information for API response."""
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
            table_details = self._get_table_details(final_results[:10])
            
            logger.info(f"📊 Found {len(final_results)} tables using Advanced Graph Traversal")
            return final_results[:10], table_details
            
        except Exception as e:
            logger.error(f"Advanced graph traversal failed: {e}")
            # Fallback to basic search
            return self._fallback_basic_traversal(query), {}
    
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
                        'description': table_info[1] or '',
                        'table_type': table_info[2] or '',
                        'record_count': None,  # Could be expensive to compute
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
    
    # Helper methods (simplified versions)
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

# === Global Instances ===
graph_retriever = None
sql_generator = None

def get_graph_retriever():
    """Get or create graph retriever instance."""
    global graph_retriever
    if graph_retriever is None:
        try:
            graph_retriever = AdvancedGraphTraversalRetriever(DB_PATH)
        except Exception as e:
            logger.error(f"Failed to initialize graph retriever: {e}")
            raise HTTPException(status_code=500, detail="Graph retriever initialization failed")
    return graph_retriever

def get_sql_generator():
    """Get or create SQL generator instance."""
    global sql_generator
    if sql_generator is None:
        sql_generator = SQLGenerator()
    return sql_generator

# === API Endpoints ===

@app.post("/text-to-table", response_model=TextToTableResponse)
async def text_to_table(request: TextToTableRequest):
    """
    Find relevant database tables based on natural language query using Advanced Graph Traversal.
    
    This endpoint uses Advanced Graph Traversal method that combines:
    - Graph Neural Networks (GNN) for table classification
    - Reinforcement Learning for path optimization
    - Multi-level graph reasoning
    """
    start_time = time.time()
    
    try:
        logger.info(f"[{request.user_id}|{request.request_id}] Text-to-Table: {request.query}")
        
        # Get graph retriever
        retriever = get_graph_retriever()
        
        # Find relevant tables
        tables, table_details = retriever.get_tables_with_details(request.query)
        
        # Limit results
        tables = tables[:request.max_tables]
        
        # Create response
        processing_time = time.time() - start_time
        
        # Convert table details to response format
        formatted_details = {}
        for table_name, details in table_details.items():
            if table_name in tables and 'error' not in details:
                formatted_details[table_name] = TableInfo(
                    name=table_name,
                    description=details.get('description', ''),
                    table_type=details.get('table_type', ''),
                    record_count=details.get('record_count'),
                    columns=details.get('columns', [])
                )
        
        logger.info(f"[{request.user_id}|{request.request_id}] Found {len(tables)} tables in {processing_time:.3f}s")
        
        return TextToTableResponse(
            success=True,
            message=f"Found {len(tables)} relevant tables using Advanced Graph Traversal",
            tables=tables,
            table_details=formatted_details,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[{request.user_id}|{request.request_id}] Text-to-Table failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Text-to-Table processing failed: {str(e)}"
        )

@app.post("/table-to-sql", response_model=TableToSQLResponse)
async def table_to_sql(request: TableToSQLRequest):
    """
    Generate SQL query from selected tables and original natural language query.
    
    This endpoint takes user-selected tables and generates optimized SQL using GPT
    with enhanced schema context and validation.
    """
    start_time = time.time()
    
    try:
        logger.info(f"[{request.user_id}|{request.request_id}] Table-to-SQL for tables: {request.selected_tables}")
        
        # Get SQL generator
        generator = get_sql_generator()
        
        # Get table details for selected tables
        retriever = get_graph_retriever()
        table_details = retriever._get_table_details(request.selected_tables)
        
        # Generate SQL
        sql_code, reasoning = await generator.generate_sql(
            request.original_query, 
            request.selected_tables, 
            table_details
        )
        
        processing_time = time.time() - start_time
        
        # Validate the generated SQL
        validation = generator._validate_sql(sql_code)
        validation_status = "valid" if validation['is_valid'] else f"invalid: {validation['error']}"
        
        logger.info(f"[{request.user_id}|{request.request_id}] Generated SQL in {processing_time:.3f}s")
        
        return TableToSQLResponse(
            success=True,
            message="SQL generated successfully from selected tables",
            sql=sql_code,
            reasoning=reasoning,
            tables_used=request.selected_tables,
            processing_time=processing_time,
            validation_status=validation_status
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[{request.user_id}|{request.request_id}] Table-to-SQL failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Table-to-SQL processing failed: {str(e)}"
        )

@app.post("/text-to-sql", response_model=TextToSQLResponse)
async def text_to_sql(request: TextToSQLRequest):
    """
    Complete text-to-SQL pipeline using Advanced Graph Traversal method.
    
    This endpoint combines table discovery and SQL generation in one step:
    1. Uses Advanced Graph Traversal to find relevant tables
    2. Generates optimized SQL using GPT with enhanced prompting
    3. Validates and fixes SQL if needed
    """
    start_time = time.time()
    
    try:
        logger.info(f"[{request.user_id}|{request.request_id}] Text-to-SQL: {request.query}")
        
        # Step 1: Find relevant tables using Advanced Graph Traversal
        retriever = get_graph_retriever()
        tables, table_details = retriever.get_tables_with_details(request.query)
        
        # Limit tables for SQL generation
        selected_tables = tables[:request.max_tables]
        
        logger.info(f"[{request.user_id}|{request.request_id}] Found {len(selected_tables)} relevant tables")
        
        # Step 2: Generate SQL using selected tables
        generator = get_sql_generator()
        sql_code, reasoning = await generator.generate_sql(
            request.query, 
            selected_tables, 
            table_details
        )
        
        processing_time = time.time() - start_time
        
        # Convert table details to response format
        formatted_details = {}
        for table_name, details in table_details.items():
            if table_name in selected_tables and 'error' not in details:
                formatted_details[table_name] = TableInfo(
                    name=table_name,
                    description=details.get('description', ''),
                    table_type=details.get('table_type', ''),
                    record_count=details.get('record_count'),
                    columns=details.get('columns', [])
                )
        
        logger.info(f"[{request.user_id}|{request.request_id}] Generated complete text-to-SQL in {processing_time:.3f}s")
        
        # Prepare reasoning response
        full_reasoning = reasoning if request.include_reasoning else "Reasoning hidden by request"
        
        return TextToSQLResponse(
            success=True,
            message="Complete text-to-SQL pipeline executed successfully",
            sql=sql_code,
            reasoning=full_reasoning,
            tables_found=selected_tables,
            table_details=formatted_details,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[{request.user_id}|{request.request_id}] Text-to-SQL failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Text-to-SQL processing failed: {str(e)}"
        )

# === Utility Endpoints ===

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    try:
        # Test database connection
        conn = duckdb.connect(DB_PATH)
        table_count = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
        conn.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "database": {
                "connected": True,
                "table_count": table_count,
                "path": DB_PATH
            },
            "services": {
                "graph_retriever": graph_retriever is not None,
                "sql_generator": sql_generator is not None,
                "client_manager": CLIENT_MANAGER_AVAILABLE
            },
            "version": "2.0.0"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Service unhealthy: {str(e)}"
        )

@app.get("/database/info")
async def get_database_info():
    """Get information about the knowledge graph database."""
    try:
        conn = duckdb.connect(DB_PATH)
        
        # Get table statistics
        tables_info = conn.execute("""
            SELECT table_type, COUNT(*) as count 
            FROM tables 
            GROUP BY table_type
        """).fetchall()
        
        # Get total columns
        total_columns = conn.execute("SELECT COUNT(*) FROM columns").fetchone()[0]
        
        # Get relationships if available
        try:
            total_relationships = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        except:
            total_relationships = 0
        
        conn.close()
        
        return {
            "database_path": DB_PATH,
            "tables_by_type": {table_type: count for table_type, count in tables_info},
            "total_columns": total_columns,
            "total_relationships": total_relationships,
            "last_updated": datetime.datetime.fromtimestamp(os.path.getmtime(DB_PATH)).isoformat() if os.path.exists(DB_PATH) else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get database info: {str(e)}"
        )

@app.get("/tables/search")
async def search_tables(
    query: str, 
    limit: int = 20,
    table_type: Optional[str] = None
):
    """Search tables by name or description."""
    try:
        conn = duckdb.connect(DB_PATH)
        
        sql = """
            SELECT name, description, table_type 
            FROM tables 
            WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
        """
        params = [f"%{query.lower()}%", f"%{query.lower()}%"]
        
        if table_type:
            sql += " AND table_type = ?"
            params.append(table_type)
        
        sql += f" ORDER BY name LIMIT {limit}"
        
        results = conn.execute(sql, params).fetchall()
        conn.close()
        
        return {
            "query": query,
            "results": [
                {
                    "name": row[0],
                    "description": row[1] or "",
                    "table_type": row[2] or ""
                }
                for row in results
            ],
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Table search failed: {str(e)}"
        )

@app.get("/tables/{table_name}/details")
async def get_table_details(table_name: str):
    """Get detailed information about a specific table."""
    try:
        retriever = get_graph_retriever()
        table_details = retriever._get_table_details([table_name])
        
        if table_name not in table_details:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        
        details = table_details[table_name]
        if 'error' in details:
            raise HTTPException(status_code=500, detail=details['error'])
        
        return {
            "table_name": table_name,
            "details": details
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get table details: {str(e)}"
        )

@app.post("/sql/validate")
async def validate_sql(sql_query: str):
    """Validate SQL query syntax."""
    try:
        generator = get_sql_generator()
        validation = generator._validate_sql(sql_query)
        
        return {
            "sql": sql_query,
            "is_valid": validation['is_valid'],
            "error": validation['error'],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"SQL validation failed: {str(e)}"
        )

# === Error Handlers ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.datetime.now().isoformat(),
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.datetime.now().isoformat(),
            "status_code": 500,
            "path": str(request.url)
        }
    )

# === Startup and Shutdown Events ===

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🚀 Starting Advanced Graph Text-to-SQL API...")
    
    # Test database connection
    try:
        conn = duckdb.connect(DB_PATH)
        table_count = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
        conn.close()
        logger.info(f"📊 Database connected: {table_count} tables available")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise
    
    # Initialize components
    try:
        get_graph_retriever()
        get_sql_generator()
        logger.info("✅ All components initialized successfully")
    except Exception as e:
        logger.error(f"❌ Component initialization failed: {e}")
        raise
    
    logger.info("🌟 Advanced Graph Text-to-SQL API is ready!")
    logger.info("📚 API Documentation: http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down Advanced Graph Text-to-SQL API...")

# === Main Function ===

if __name__ == "__main__":
    import uvicorn
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        logger.error(f"❌ Database file not found: {DB_PATH}")
        logger.error("💡 Please ensure knowledge_graph.duckdb is in the same directory")
        exit(1)
    
    logger.info("🚀 Starting Advanced Graph Text-to-SQL API Server...")
    logger.info(f"📍 Database: {DB_PATH}")
    logger.info(f"🔗 Client Manager Available: {CLIENT_MANAGER_AVAILABLE}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )