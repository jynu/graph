#!/usr/bin/env python3
"""
Enhanced Text-to-SQL Graph Traversal with Improved Embeddings and Attention

Key Improvements:
1. Comprehensive Multi-Modal Table Embeddings
2. Sophisticated Multi-Head Attention Mechanism
3. Dynamic Query-Aware Graph Reasoning
4. Hierarchical Embedding Fusion
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import re
from collections import defaultdict
import duckdb

logger = logging.getLogger(__name__)

class EnhancedTableEmbedding:
    """
    Comprehensive table embedding that incorporates multiple information sources
    with weighted fusion and hierarchical representation.
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.component_weights = {
            'semantic': 0.35,      # Table/column descriptions
            'structural': 0.25,    # Data types, constraints, relationships
            'statistical': 0.20,   # Distinct values, frequent patterns
            'contextual': 0.20     # Usage patterns, query history
        }
        
    def create_comprehensive_embedding(self, table_name: str, conn: duckdb.DuckDBPyConnection) -> np.ndarray:
        """Create multi-modal table embedding from all available information sources."""
        try:
            # 1. Semantic Component: Descriptions and textual information
            semantic_embedding = self._create_semantic_embedding(table_name, conn)
            
            # 2. Structural Component: Schema and relationship information
            structural_embedding = self._create_structural_embedding(table_name, conn)
            
            # 3. Statistical Component: Data distribution and patterns
            statistical_embedding = self._create_statistical_embedding(table_name, conn)
            
            # 4. Contextual Component: Usage patterns and query context
            contextual_embedding = self._create_contextual_embedding(table_name, conn)
            
            # Weighted fusion of all components
            final_embedding = (
                semantic_embedding * self.component_weights['semantic'] +
                structural_embedding * self.component_weights['structural'] +
                statistical_embedding * self.component_weights['statistical'] +
                contextual_embedding * self.component_weights['contextual']
            )
            
            # L2 normalization for consistent similarity computation
            norm = np.linalg.norm(final_embedding)
            return final_embedding / (norm + 1e-8)
            
        except Exception as e:
            logger.error(f"Failed to create comprehensive embedding for {table_name}: {e}")
            return np.random.normal(0, 0.1, self.embedding_dim)
    
    def _create_semantic_embedding(self, table_name: str, conn: duckdb.DuckDBPyConnection) -> np.ndarray:
        """Create embedding from semantic textual information."""
        try:
            # Get table and column descriptions
            semantic_sql = """
            SELECT 
                t.description as table_desc,
                t.business_purpose,
                STRING_AGG(c.name || ': ' || COALESCE(c.description, ''), '; ') as column_info,
                STRING_AGG(c.business_meaning, '; ') as business_meanings
            FROM tables t
            LEFT JOIN columns c ON t.name = c.table_name
            WHERE t.name = ?
            GROUP BY t.description, t.business_purpose
            """
            
            result = conn.execute(semantic_sql, [table_name]).fetchone()
            if result:
                table_desc, business_purpose, column_info, business_meanings = result
                
                # Combine all semantic information
                semantic_text = " ".join(filter(None, [
                    table_desc or "",
                    business_purpose or "",
                    column_info or "",
                    business_meanings or ""
                ]))
                
                # Use your existing embedding function
                if semantic_text.strip():
                    from app.rag.embedding import embedding
                    return np.array(embedding.embed_text(semantic_text))
            
            # Fallback: table name only
            from app.rag.embedding import embedding
            return np.array(embedding.embed_text(table_name))
            
        except Exception as e:
            logger.warning(f"Semantic embedding failed for {table_name}: {e}")
            return np.random.normal(0, 0.1, self.embedding_dim)
    
    def _create_structural_embedding(self, table_name: str, conn: duckdb.DuckDBPyConnection) -> np.ndarray:
        """Create embedding from structural schema information."""
        try:
            # Get structural information
            structural_sql = """
            SELECT 
                c.data_type,
                c.is_nullable,
                c.is_primary_key,
                c.is_foreign_key,
                c.column_constraints
            FROM columns c
            WHERE c.table_name = ?
            """
            
            columns = conn.execute(structural_sql, [table_name]).fetchall()
            
            # Create structural signature
            data_types = []
            constraints = []
            key_info = []
            
            for col in columns:
                data_type, is_nullable, is_pk, is_fk, col_constraints = col
                data_types.append(data_type or "unknown")
                
                if is_pk:
                    key_info.append("PK")
                if is_fk:
                    key_info.append("FK")
                if col_constraints:
                    constraints.extend(col_constraints.split(','))
            
            # Get relationship information
            relationship_sql = """
            SELECT relationship_type, COUNT(*) as count
            FROM relationships
            WHERE from_table = ? OR to_table = ?
            GROUP BY relationship_type
            """
            relationships = conn.execute(relationship_sql, [table_name, table_name]).fetchall()
            
            # Create structural text representation
            structural_text = " ".join([
                f"types: {' '.join(set(data_types))}",
                f"keys: {' '.join(key_info)}",
                f"constraints: {' '.join(set(constraints))}",
                f"relationships: {' '.join([f'{rel}:{count}' for rel, count in relationships])}"
            ])
            
            # Embed structural information
            from app.rag.embedding import embedding
            return np.array(embedding.embed_text(structural_text))
            
        except Exception as e:
            logger.warning(f"Structural embedding failed for {table_name}: {e}")
            return np.random.normal(0, 0.1, self.embedding_dim)
    
    def _create_statistical_embedding(self, table_name: str, conn: duckdb.DuckDBPyConnection) -> np.ndarray:
        """Create embedding from statistical data patterns."""
        try:
            # Get statistical information
            statistical_sql = """
            SELECT 
                c.name,
                c.distinct_value_count,
                c.null_percentage,
                c.frequent_values,
                c.value_distribution_type
            FROM columns c
            WHERE c.table_name = ?
            """
            
            stats = conn.execute(statistical_sql, [table_name]).fetchall()
            
            # Create statistical signature
            stat_features = []
            value_patterns = []
            
            for col_name, distinct_count, null_pct, freq_values, dist_type in stats:
                # Cardinality patterns
                if distinct_count:
                    if distinct_count < 10:
                        stat_features.append("low_cardinality")
                    elif distinct_count < 1000:
                        stat_features.append("medium_cardinality")
                    else:
                        stat_features.append("high_cardinality")
                
                # Null patterns
                if null_pct and null_pct > 0.1:
                    stat_features.append("has_nulls")
                
                # Distribution patterns
                if dist_type:
                    stat_features.append(f"dist_{dist_type}")
                
                # Frequent value patterns
                if freq_values:
                    try:
                        values = json.loads(freq_values) if isinstance(freq_values, str) else freq_values
                        if isinstance(values, list):
                            value_patterns.extend([str(v)[:20] for v in values[:5]])  # Top 5 values, truncated
                    except:
                        pass
            
            # Create statistical text representation
            statistical_text = " ".join([
                f"patterns: {' '.join(set(stat_features))}",
                f"values: {' '.join(value_patterns[:10])}"  # Limit to 10 values
            ])
            
            # Embed statistical information
            from app.rag.embedding import embedding
            return np.array(embedding.embed_text(statistical_text))
            
        except Exception as e:
            logger.warning(f"Statistical embedding failed for {table_name}: {e}")
            return np.random.normal(0, 0.1, self.embedding_dim)
    
    def _create_contextual_embedding(self, table_name: str, conn: duckdb.DuckDBPyConnection) -> np.ndarray:
        """Create embedding from contextual usage patterns."""
        try:
            # Get contextual information (query history, join patterns, etc.)
            contextual_sql = """
            SELECT 
                t.usage_frequency,
                t.common_join_partners,
                t.typical_query_patterns,
                t.business_domain
            FROM tables t
            WHERE t.name = ?
            """
            
            result = conn.execute(contextual_sql, [table_name]).fetchone()
            
            contextual_features = []
            
            if result:
                usage_freq, join_partners, query_patterns, business_domain = result
                
                if usage_freq:
                    if usage_freq > 1000:
                        contextual_features.append("high_usage")
                    elif usage_freq > 100:
                        contextual_features.append("medium_usage")
                    else:
                        contextual_features.append("low_usage")
                
                if join_partners:
                    contextual_features.extend(join_partners.split(',')[:5])  # Top 5 join partners
                
                if query_patterns:
                    contextual_features.extend(query_patterns.split(',')[:3])  # Top 3 patterns
                
                if business_domain:
                    contextual_features.append(f"domain_{business_domain}")
            
            # Create contextual text representation
            contextual_text = " ".join(contextual_features) if contextual_features else f"table_{table_name}"
            
            # Embed contextual information
            from app.rag.embedding import embedding
            return np.array(embedding.embed_text(contextual_text))
            
        except Exception as e:
            logger.warning(f"Contextual embedding failed for {table_name}: {e}")
            return np.random.normal(0, 0.1, self.embedding_dim)


class MultiHeadQueryTableAttention(nn.Module):
    """
    Advanced multi-head attention mechanism for query-table matching
    with dynamic key-value computation and positional encoding.
    """
    
    def __init__(self, embedding_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Additional components
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Learnable parameters for different attention types
        self.semantic_weight = nn.Parameter(torch.tensor(0.4))
        self.structural_weight = nn.Parameter(torch.tensor(0.3))
        self.contextual_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, query_embedding: torch.Tensor, table_embeddings: torch.Tensor,
                table_metadata: Dict[str, Any] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention between query and table embeddings.
        
        Args:
            query_embedding: [1, embedding_dim] - User query embedding
            table_embeddings: [num_tables, embedding_dim] - Table embeddings
            table_metadata: Additional metadata for enhanced attention
            
        Returns:
            attention_scores: [num_tables] - Attention scores for each table
            attention_weights: [num_heads, num_tables] - Detailed attention weights
        """
        batch_size = 1
        num_tables = table_embeddings.size(0)
        
        # Expand query for broadcasting
        query_expanded = query_embedding.unsqueeze(0).expand(num_tables, -1)  # [num_tables, embedding_dim]
        
        # Project to Q, K, V
        Q = self.query_projection(query_expanded)  # [num_tables, embedding_dim]
        K = self.key_projection(table_embeddings)   # [num_tables, embedding_dim]
        V = self.value_projection(table_embeddings) # [num_tables, embedding_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(num_tables, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, num_tables, head_dim]
        K = K.view(num_tables, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, num_tables, head_dim]
        V = V.view(num_tables, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, num_tables, head_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [num_heads, num_tables, num_tables]
        
        # Apply metadata-based attention bias if available
        if table_metadata:
            attention_bias = self._compute_metadata_bias(table_metadata, num_tables)
            attention_scores = attention_scores + attention_bias.unsqueeze(0)  # Broadcast to all heads
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [num_heads, num_tables, num_tables]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # [num_heads, num_tables, head_dim]
        
        # Concatenate heads
        attended_values = attended_values.transpose(0, 1).contiguous().view(num_tables, self.embedding_dim)
        
        # Final projection
        output = self.output_projection(attended_values)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + table_embeddings)
        
        # Compute final attention scores (diagonal of attention matrix averaged across heads)
        final_scores = torch.diagonal(attention_weights, dim1=-2, dim2=-1).mean(dim=0)  # [num_tables]
        
        return final_scores, attention_weights
    
    def _compute_metadata_bias(self, metadata: Dict[str, Any], num_tables: int) -> torch.Tensor:
        """Compute attention bias based on table metadata."""
        bias = torch.zeros(num_tables, num_tables)
        
        for i, table_meta in enumerate(metadata.get('tables', [])):
            # Boost attention for tables with high relevance scores
            relevance_score = table_meta.get('relevance_score', 0.0)
            bias[i, i] += relevance_score * 0.1
            
            # Boost attention for frequently used tables
            usage_frequency = table_meta.get('usage_frequency', 0)
            if usage_frequency > 1000:
                bias[i, i] += 0.05
            
            # Boost attention for tables with matching business domain
            query_domain = metadata.get('query_domain', '')
            table_domain = table_meta.get('business_domain', '')
            if query_domain and query_domain == table_domain:
                bias[i, i] += 0.1
        
        return bias


class EnhancedGraphTraversalRetriever:
    """
    Enhanced graph traversal with improved embeddings and attention mechanism.
    """
    
    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path)
        self.embedding_creator = EnhancedTableEmbedding()
        
        # Initialize attention mechanism
        self.attention_model = MultiHeadQueryTableAttention()
        self.attention_model.eval()  # Set to evaluation mode
        
        # Build enhanced graph and embeddings
        self.graph_structure = self._build_enhanced_graph()
        self.enhanced_embeddings = self._compute_enhanced_embeddings()
        
        # Configuration
        self.config = {
            'embedding_fusion_strategy': 'weighted',  # 'weighted', 'concatenated', 'learned'
            'attention_temperature': 1.0,
            'diversity_penalty': 0.1,
            'structural_boost': 0.2,
            'semantic_threshold': 0.3
        }
    
    def get_tables_with_enhanced_attention(self, query: str, max_tables: int = 10, 
                                         similarity_threshold: float = 0.2) -> Tuple[List[str], Dict]:
        """Enhanced table retrieval with sophisticated attention mechanism."""
        try:
            # 1. Encode query using comprehensive embedding
            query_embedding = self._encode_query_comprehensively(query)
            
            # 2. Prepare table embeddings and metadata
            table_names = list(self.enhanced_embeddings.keys())
            table_embeddings_matrix = np.stack([self.enhanced_embeddings[name] for name in table_names])
            
            # Convert to tensors
            query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)
            table_tensor = torch.tensor(table_embeddings_matrix, dtype=torch.float32)
            
            # 3. Prepare metadata for attention bias
            table_metadata = self._prepare_table_metadata(table_names, query)
            
            # 4. Compute attention scores
            with torch.no_grad():
                attention_scores, attention_weights = self.attention_model(
                    query_tensor, table_tensor, table_metadata
                )
            
            # 5. Convert back to numpy and combine with graph structure
            attention_scores_np = attention_scores.numpy()
            
            # 6. Apply graph-based enhancements
            enhanced_scores = self._apply_graph_enhancements(
                table_names, attention_scores_np, query
            )
            
            # 7. Rank and filter tables
            table_score_pairs = list(zip(table_names, enhanced_scores))
            table_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by threshold and limit
            filtered_tables = [
                table for table, score in table_score_pairs 
                if score >= similarity_threshold
            ][:max_tables]
            
            # 8. Get detailed information
            table_details = self._get_enhanced_table_details(filtered_tables, query, table_score_pairs)
            
            return filtered_tables, table_details
            
        except Exception as e:
            logger.error(f"Enhanced attention retrieval failed: {e}")
            # Fallback to basic method
            return self._fallback_retrieval(query, max_tables)
    
    def _encode_query_comprehensively(self, query: str) -> np.ndarray:
        """Encode query using multiple strategies and combine."""
        try:
            # 1. Basic semantic encoding
            from app.rag.embedding import embedding
            semantic_embedding = np.array(embedding.embed_text(query))
            
            # 2. Extract query intent and entities
            query_entities = self._extract_query_entities(query)
            query_intent = self._classify_query_intent(query)
            
            # 3. Create intent-aware embedding
            intent_text = f"intent: {query_intent} entities: {' '.join(query_entities)}"
            intent_embedding = np.array(embedding.embed_text(intent_text))
            
            # 4. Combine embeddings
            comprehensive_embedding = 0.7 * semantic_embedding + 0.3 * intent_embedding
            
            # Normalize
            norm = np.linalg.norm(comprehensive_embedding)
            return comprehensive_embedding / (norm + 1e-8)
            
        except Exception as e:
            logger.warning(f"Comprehensive query encoding failed: {e}")
            # Fallback to basic encoding
            from app.rag.embedding import embedding
            return np.array(embedding.embed_text(query))
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract key entities from the query."""
        # Simple entity extraction (in production, use NER)
        entities = []
        
        # Common financial/data entities
        entity_patterns = [
            r'\b(trade|trading|trader|execution)\w*\b',
            r'\b(price|amount|volume|notional)\w*\b',
            r'\b(date|time|yesterday|today|week|month)\w*\b',
            r'\b(currency|USD|EUR|GBP)\b',
            r'\b(equity|bond|derivative|option)\w*\b',
            r'\b(counterparty|client|entity)\w*\b'
        ]
        
        query_lower = query.lower()
        for pattern in entity_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _classify_query_intent(self, query: str) -> str:
        """Classify the intent of the query."""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['show', 'list', 'get', 'fetch', 'find']):
            return 'retrieval'
        elif any(keyword in query_lower for keyword in ['count', 'sum', 'average', 'total']):
            return 'aggregation'
        elif any(keyword in query_lower for keyword in ['compare', 'difference', 'versus']):
            return 'comparison'
        elif any(keyword in query_lower for keyword in ['filter', 'where', 'only', 'specific']):
            return 'filtering'
        elif any(keyword in query_lower for keyword in ['rank', 'top', 'highest', 'lowest']):
            return 'ranking'
        else:
            return 'general'
    
    def _prepare_table_metadata(self, table_names: List[str], query: str) -> Dict[str, Any]:
        """Prepare comprehensive metadata for attention mechanism."""
        metadata = {
            'query_domain': self._infer_query_domain(query),
            'query_complexity': self._assess_query_complexity(query),
            'tables': []
        }
        
        for table_name in table_names:
            try:
                table_meta_sql = """
                SELECT 
                    usage_frequency,
                    business_domain,
                    table_importance_score,
                    avg_query_performance
                FROM tables
                WHERE name = ?
                """
                result = self.conn.execute(table_meta_sql, [table_name]).fetchone()
                
                table_meta = {
                    'name': table_name,
                    'usage_frequency': result[0] if result else 0,
                    'business_domain': result[1] if result else 'unknown',
                    'importance_score': result[2] if result else 0.5,
                    'query_performance': result[3] if result else 1.0,
                    'relevance_score': self._compute_basic_relevance(table_name, query)
                }
                
                metadata['tables'].append(table_meta)
                
            except Exception as e:
                logger.warning(f"Failed to get metadata for {table_name}: {e}")
                metadata['tables'].append({
                    'name': table_name,
                    'usage_frequency': 0,
                    'business_domain': 'unknown',
                    'importance_score': 0.5,
                    'query_performance': 1.0,
                    'relevance_score': 0.0
                })
        
        return metadata
    
    def _infer_query_domain(self, query: str) -> str:
        """Infer the business domain from the query."""
        query_lower = query.lower()
        
        domain_keywords = {
            'trading': ['trade', 'trading', 'execution', 'order', 'fill'],
            'risk': ['risk', 'exposure', 'limit', 'var', 'stress'],
            'finance': ['price', 'pnl', 'profit', 'loss', 'revenue'],
            'compliance': ['compliance', 'regulatory', 'audit', 'breach'],
            'operations': ['settlement', 'clearing', 'processing', 'workflow']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess the complexity of the query."""
        complexity_indicators = [
            len(query.split()) > 10,  # Long query
            'and' in query.lower() or 'or' in query.lower(),  # Logical operators
            any(word in query.lower() for word in ['group', 'aggregate', 'sum', 'count']),  # Aggregation
            any(word in query.lower() for word in ['join', 'combine', 'merge']),  # Multi-table
            any(word in query.lower() for word in ['where', 'filter', 'condition'])  # Filtering
        ]
        
        complexity_score = sum(complexity_indicators)
        
        if complexity_score >= 3:
            return 'high'
        elif complexity_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _compute_basic_relevance(self, table_name: str, query: str) -> float:
        """Compute basic relevance score between table and query."""
        try:
            # Get table description
            desc_sql = "SELECT description FROM tables WHERE name = ?"
            result = self.conn.execute(desc_sql, [table_name]).fetchone()
            
            if result and result[0]:
                description = result[0].lower()
                query_lower = query.lower()
                
                # Simple word overlap
                query_words = set(query_lower.split())
                desc_words = set(description.split())
                
                overlap = len(query_words.intersection(desc_words))
                total_words = len(query_words.union(desc_words))
                
                return overlap / total_words if total_words > 0 else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Basic relevance computation failed: {e}")
            return 0.0
    
    def _apply_graph_enhancements(self, table_names: List[str], attention_scores: np.ndarray, 
                                query: str) -> np.ndarray:
        """Apply graph structure enhancements to attention scores."""
        enhanced_scores = attention_scores.copy()
        
        try:
            # 1. Centrality boost
            for i, table_name in enumerate(table_names):
                centrality = self._get_table_centrality(table_name)
                enhanced_scores[i] += centrality * self.config['structural_boost']
            
            # 2. Query-specific boost
            for i, table_name in enumerate(table_names):
                query_relevance = self._compute_query_table_relevance(table_name, query)
                enhanced_scores[i] += query_relevance * 0.1
            
            # 3. Diversity penalty (avoid selecting too similar tables)
            enhanced_scores = self._apply_diversity_penalty(
                table_names, enhanced_scores, self.config['diversity_penalty']
            )
            
            return enhanced_scores
            
        except Exception as e:
            logger.warning(f"Graph enhancement failed: {e}")
            return attention_scores
    
    def _apply_diversity_penalty(self, table_names: List[str], scores: np.ndarray, 
                               penalty_weight: float) -> np.ndarray:
        """Apply diversity penalty to encourage selecting diverse tables."""
        try:
            penalized_scores = scores.copy()
            
            # Compute pairwise similarities between tables
            for i, table1 in enumerate(table_names):
                for j, table2 in enumerate(table_names):
                    if i != j and i < j:  # Avoid double counting
                        # Compute similarity between tables
                        similarity = self._compute_table_similarity(table1, table2)
                        
                        # Apply penalty to both tables based on similarity
                        penalty = similarity * penalty_weight
                        penalized_scores[i] -= penalty
                        penalized_scores[j] -= penalty
            
            # Ensure scores remain non-negative
            penalized_scores = np.maximum(penalized_scores, 0.0)
            
            return penalized_scores
            
        except Exception as e:
            logger.warning(f"Diversity penalty failed: {e}")
            return scores
    
    def _compute_table_similarity(self, table1: str, table2: str) -> float:
        """Compute similarity between two tables."""
        try:
            if table1 in self.enhanced_embeddings and table2 in self.enhanced_embeddings:
                emb1 = self.enhanced_embeddings[table1]
                emb2 = self.enhanced_embeddings[table2]
                
                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
                return max(0.0, similarity)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Table similarity computation failed: {e}")
            return 0.0
    
    def _compute_enhanced_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute comprehensive embeddings for all tables."""
        logger.info("Computing enhanced embeddings for all tables...")
        embeddings = {}
        
        try:
            # Get all table names
            tables_sql = "SELECT name FROM tables"
            table_results = self.conn.execute(tables_sql).fetchall()
            
            for (table_name,) in table_results:
                try:
                    embedding = self.embedding_creator.create_comprehensive_embedding(table_name, self.conn)
                    embeddings[table_name] = embedding
                    
                except Exception as e:
                    logger.warning(f"Failed to create embedding for {table_name}: {e}")
                    # Fallback to random embedding
                    embeddings[table_name] = np.random.normal(0, 0.1, self.embedding_creator.embedding_dim)
            
            logger.info(f"Created enhanced embeddings for {len(embeddings)} tables")
            return embeddings
            
        except Exception as e:
            logger.error(f"Enhanced embedding computation failed: {e}")
            return {}
    
    def _get_enhanced_table_details(self, table_names: List[str], query: str, 
                                  all_scores: List[Tuple[str, float]]) -> Dict:
        """Get enhanced table details with attention analysis."""
        details = {}
        score_lookup = dict(all_scores)
        
        for table_name in table_names:
            try:
                # Get basic table information
                table_sql = """
                SELECT name, description, table_type, business_purpose, usage_frequency
                FROM tables 
                WHERE name = ?
                """
                table_info = self.conn.execute(table_sql, [table_name]).fetchone()
                
                # Get enhanced column information
                columns_sql = """
                SELECT 
                    name, 
                    data_type, 
                    description,
                    business_meaning,
                    distinct_value_count,
                    frequent_values
                FROM columns 
                WHERE table_name = ?
                ORDER BY column_importance_score DESC
                """
                columns = self.conn.execute(columns_sql, [table_name]).fetchall()
                
                if table_info:
                    details[table_name] = {
                        'name': table_info[0],
                        'description': table_info[1] or '',
                        'table_type': table_info[2] or '',
                        'business_purpose': table_info[3] or '',
                        'usage_frequency': table_info[4] or 0,
                        'attention_score': score_lookup.get(table_name, 0.0),
                        'rank': next((i+1 for i, (t, _) in enumerate(all_scores) if t == table_name), 999),
                        'confidence': self._compute_confidence_level(score_lookup.get(table_name, 0.0)),
                        'columns': [
                            {
                                'name': col[0],
                                'data_type': col[1],
                                'description': col[2] or '',
                                'business_meaning': col[3] or '',
                                'distinct_count': col[4] or 0,
                                'sample_values': self._parse_sample_values(col[5]) if col[5] else []
                            }
                            for col in columns
                        ],
                        'query_relevance_analysis': self._analyze_query_relevance(table_name, query)
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to get enhanced details for table {table_name}: {e}")
                details[table_name] = {'error': str(e)}
        
        return details
    
    def _compute_confidence_level(self, score: float) -> str:
        """Compute confidence level based on attention score."""
        if score >= 0.8:
            return "Very High"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _parse_sample_values(self, values_json: str) -> List[str]:
        """Parse sample values from JSON string."""
        try:
            if isinstance(values_json, str):
                values = json.loads(values_json)
            else:
                values = values_json
            
            if isinstance(values, list):
                return [str(v) for v in values[:5]]  # Top 5 values
            else:
                return [str(values)]
                
        except Exception as e:
            return []
    
    def _analyze_query_relevance(self, table_name: str, query: str) -> Dict[str, Any]:
        """Analyze why this table is relevant to the query."""
        analysis = {
            'semantic_matches': [],
            'structural_matches': [],
            'contextual_matches': [],
            'reasoning': ''
        }
        
        try:
            # Semantic analysis
            desc_sql = "SELECT description, business_purpose FROM tables WHERE name = ?"
            result = self.conn.execute(desc_sql, [table_name]).fetchone()
            
            if result:
                description, business_purpose = result
                query_lower = query.lower()
                
                # Find semantic matches
                if description:
                    desc_words = set(description.lower().split())
                    query_words = set(query_lower.split())
                    matches = desc_words.intersection(query_words)
                    analysis['semantic_matches'] = list(matches)
                
                # Find business purpose matches
                if business_purpose and any(word in business_purpose.lower() for word in query_words):
                    analysis['contextual_matches'].append('business_purpose_match')
            
            # Structural analysis - check column names
            col_sql = "SELECT name FROM columns WHERE table_name = ?"
            columns = self.conn.execute(col_sql, [table_name]).fetchall()
            
            query_terms = query.lower().split()
            for (col_name,) in columns:
                if any(term in col_name.lower() for term in query_terms):
                    analysis['structural_matches'].append(col_name)
            
            # Generate reasoning
            reasoning_parts = []
            if analysis['semantic_matches']:
                reasoning_parts.append(f"Semantic match: {', '.join(analysis['semantic_matches'])}")
            if analysis['structural_matches']:
                reasoning_parts.append(f"Column matches: {', '.join(analysis['structural_matches'][:3])}")
            if analysis['contextual_matches']:
                reasoning_parts.append(f"Contextual relevance: {', '.join(analysis['contextual_matches'])}")
            
            analysis['reasoning'] = '; '.join(reasoning_parts) if reasoning_parts else 'General relevance based on embedding similarity'
            
        except Exception as e:
            logger.warning(f"Query relevance analysis failed for {table_name}: {e}")
            analysis['reasoning'] = 'Analysis unavailable'
        
        return analysis
    
    # Utility methods from original implementation
    def _build_enhanced_graph(self):
        """Build enhanced graph structure - simplified version."""
        try:
            import networkx as nx
            G = nx.DiGraph()
            
            # Add tables as nodes
            tables_sql = "SELECT name, description, table_type FROM tables"
            tables = self.conn.execute(tables_sql).fetchall()
            
            for table_name, description, table_type in tables:
                G.add_node(table_name, 
                          description=description or "", 
                          table_type=table_type or "unknown")
            
            # Add relationships as edges if available
            try:
                rels_sql = """
                SELECT from_table, to_table, relationship_type 
                FROM relationships 
                WHERE from_table != to_table
                """
                relationships = self.conn.execute(rels_sql).fetchall()
                
                for from_table, to_table, rel_type in relationships:
                    if G.has_node(from_table) and G.has_node(to_table):
                        G.add_edge(from_table, to_table, rel_type=rel_type)
                
            except:
                logger.warning("No relationships table found")
            
            return G
            
        except Exception as e:
            logger.warning(f"Graph building failed: {e}")
            import networkx as nx
            return nx.DiGraph()
    
    def _get_table_centrality(self, table_name: str) -> float:
        """Get table centrality score."""
        try:
            if hasattr(self, '_centrality_cache'):
                return self._centrality_cache.get(table_name, 0.0)
            
            # Compute centrality for all tables
            import networkx as nx
            if self.graph_structure and self.graph_structure.nodes():
                degree_centrality = nx.degree_centrality(self.graph_structure)
                self._centrality_cache = degree_centrality
                return degree_centrality.get(table_name, 0.0)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Centrality computation failed: {e}")
            return 0.0
    
    def _compute_query_table_relevance(self, table_name: str, query: str) -> float:
        """Compute relevance between table and query."""
        try:
            query_terms = query.lower().split()
            table_lower = table_name.lower()
            
            # Check table name match
            name_score = sum(1 for term in query_terms if term in table_lower) / len(query_terms) if query_terms else 0.0
            
            # Check description match
            desc_sql = "SELECT description FROM tables WHERE name = ?"
            result = self.conn.execute(desc_sql, [table_name]).fetchone()
            
            desc_score = 0.0
            if result and result[0]:
                description = result[0].lower()
                desc_score = sum(1 for term in query_terms if term in description) / len(query_terms) if query_terms else 0.0
            
            return (name_score * 0.4 + desc_score * 0.6)
            
        except Exception as e:
            logger.warning(f"Query relevance computation failed: {e}")
            return 0.0
    
    def _fallback_retrieval(self, query: str, max_tables: int) -> Tuple[List[str], Dict]:
        """Fallback retrieval method."""
        try:
            # Simple keyword-based search
            query_terms = query.lower().split()
            tables_sql = "SELECT name FROM tables"
            all_tables = [row[0] for row in self.conn.execute(tables_sql).fetchall()]
            
            # Score tables by keyword matches
            scored_tables = []
            for table in all_tables:
                score = sum(1 for term in query_terms if term in table.lower())
                if score > 0:
                    scored_tables.append((table, score))
            
            # Sort and limit
            scored_tables.sort(key=lambda x: x[1], reverse=True)
            selected_tables = [table for table, _ in scored_tables[:max_tables]]
            
            # Get basic details
            details = {}
            for table in selected_tables:
                details[table] = {'name': table, 'fallback': True}
            
            return selected_tables, details
            
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {e}")
            return [], {}


# Usage example and integration
class AttentionAnalyzer:
    """Utility class for analyzing and debugging attention mechanisms."""
    
    def __init__(self, retriever: EnhancedGraphTraversalRetriever):
        self.retriever = retriever
    
    def analyze_attention_patterns(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Analyze attention patterns for a given query."""
        try:
            # Get attention computation details
            query_embedding = self.retriever._encode_query_comprehensively(query)
            table_names = list(self.retriever.enhanced_embeddings.keys())[:20]  # Limit for analysis
            
            analysis = {
                'query': query,
                'query_entities': self.retriever._extract_query_entities(query),
                'query_intent': self.retriever._classify_query_intent(query),
                'query_domain': self.retriever._infer_query_domain(query),
                'attention_breakdown': {},
                'top_tables': []
            }
            
            # Analyze each table's attention components
            for table_name in table_names[:top_k]:
                if table_name in self.retriever.enhanced_embeddings:
                    table_embedding = self.retriever.enhanced_embeddings[table_name]
                    
                    # Compute different similarity components
                    semantic_sim = np.dot(query_embedding, table_embedding)
                    centrality_score = self.retriever._get_table_centrality(table_name)
                    relevance_score = self.retriever._compute_query_table_relevance(table_name, query)
                    
                    analysis['attention_breakdown'][table_name] = {
                        'semantic_similarity': float(semantic_sim),
                        'centrality_score': float(centrality_score),
                        'relevance_score': float(relevance_score),
                        'combined_score': float(semantic_sim * 0.6 + centrality_score * 0.2 + relevance_score * 0.2)
                    }
            
            # Sort by combined score
            sorted_tables = sorted(
                analysis['attention_breakdown'].items(),
                key=lambda x: x[1]['combined_score'],
                reverse=True
            )
            
            analysis['top_tables'] = [(table, scores) for table, scores in sorted_tables[:top_k]]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Attention analysis failed: {e}")
            return {'error': str(e)}
    
    def compare_attention_methods(self, query: str, methods: List[str] = None) -> Dict[str, Any]:
        """Compare different attention computation methods."""
        if methods is None:
            methods = ['semantic_only', 'structural_only', 'combined', 'enhanced']
        
        comparison = {
            'query': query,
            'methods': {},
            'agreement_analysis': {}
        }
        
        try:
            for method in methods:
                if method == 'semantic_only':
                    # Pure embedding similarity
                    results = self._compute_semantic_only_attention(query)
                elif method == 'structural_only':
                    # Pure graph structure
                    results = self._compute_structural_only_attention(query)
                elif method == 'combined':
                    # Original combined approach
                    results = self._compute_combined_attention(query)
                elif method == 'enhanced':
                    # New enhanced attention
                    results = self._compute_enhanced_attention(query)
                
                comparison['methods'][method] = results
            
            # Analyze agreement between methods
            if len(methods) > 1:
                comparison['agreement_analysis'] = self._analyze_method_agreement(comparison['methods'])
            
            return comparison
            
        except Exception as e:
            logger.error(f"Attention method comparison failed: {e}")
            return {'error': str(e)}
    
    def _compute_semantic_only_attention(self, query: str) -> List[Tuple[str, float]]:
        """Compute attention using only semantic similarity."""
        try:
            query_embedding = self.retriever._encode_query_comprehensively(query)
            results = []
            
            for table_name, table_embedding in self.retriever.enhanced_embeddings.items():
                similarity = np.dot(query_embedding, table_embedding)
                results.append((table_name, float(similarity)))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:10]
            
        except Exception as e:
            logger.warning(f"Semantic-only attention failed: {e}")
            return []
    
    def _compute_structural_only_attention(self, query: str) -> List[Tuple[str, float]]:
        """Compute attention using only graph structure."""
        try:
            results = []
            
            for table_name in self.retriever.enhanced_embeddings.keys():
                centrality = self.retriever._get_table_centrality(table_name)
                relevance = self.retriever._compute_query_table_relevance(table_name, query)
                structural_score = centrality * 0.5 + relevance * 0.5
                results.append((table_name, float(structural_score)))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:10]
            
        except Exception as e:
            logger.warning(f"Structural-only attention failed: {e}")
            return []
    
    def _compute_combined_attention(self, query: str) -> List[Tuple[str, float]]:
        """Compute attention using original combined approach."""
        try:
            query_embedding = self.retriever._encode_query_comprehensively(query)
            results = []
            
            for table_name, table_embedding in self.retriever.enhanced_embeddings.items():
                semantic_sim = np.dot(query_embedding, table_embedding)
                centrality = self.retriever._get_table_centrality(table_name)
                relevance = self.retriever._compute_query_table_relevance(table_name, query)
                
                # Original weighting
                combined_score = semantic_sim * 0.5 + centrality * 0.3 + relevance * 0.2
                results.append((table_name, float(combined_score)))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:10]
            
        except Exception as e:
            logger.warning(f"Combined attention failed: {e}")
            return []
    
    def _compute_enhanced_attention(self, query: str) -> List[Tuple[str, float]]:
        """Compute attention using enhanced multi-head approach."""
        try:
            # Use the actual enhanced method
            tables, _ = self.retriever.get_tables_with_enhanced_attention(query, max_tables=10)
            
            # Get scores (simplified - in real implementation, extract from method)
            results = []
            for i, table in enumerate(tables):
                # Approximate score based on ranking
                score = 1.0 - (i / len(tables))
                results.append((table, score))
            
            return results
            
        except Exception as e:
            logger.warning(f"Enhanced attention failed: {e}")
            return []
    
    def _analyze_method_agreement(self, method_results: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Any]:
        """Analyze agreement between different attention methods."""
        try:
            # Extract top tables from each method
            method_top_tables = {}
            for method, results in method_results.items():
                method_top_tables[method] = [table for table, _ in results[:5]]
            
            # Compute pairwise overlaps
            overlaps = {}
            methods = list(method_top_tables.keys())
            
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    set1 = set(method_top_tables[method1])
                    set2 = set(method_top_tables[method2])
                    
                    overlap = len(set1.intersection(set2))
                    total = len(set1.union(set2))
                    
                    overlaps[f"{method1}_vs_{method2}"] = {
                        'overlap_count': overlap,
                        'overlap_ratio': overlap / total if total > 0 else 0.0,
                        'common_tables': list(set1.intersection(set2))
                    }
            
            return {
                'method_top_tables': method_top_tables,
                'pairwise_overlaps': overlaps,
                'consensus_tables': self._find_consensus_tables(method_top_tables)
            }
            
        except Exception as e:
            logger.warning(f"Method agreement analysis failed: {e}")
            return {}
    
    def _find_consensus_tables(self, method_top_tables: Dict[str, List[str]]) -> List[str]:
        """Find tables that appear in multiple methods' top results."""
        table_votes = defaultdict(int)
        
        for method, tables in method_top_tables.items():
            for table in tables:
                table_votes[table] += 1
        
        # Sort by vote count
        consensus_tables = sorted(table_votes.items(), key=lambda x: x[1], reverse=True)
        
        # Return tables that appear in at least 2 methods
        return [table for table, votes in consensus_tables if votes >= 2]


# Integration example showing how to use the enhanced system
def demonstrate_enhanced_system():
    """Demonstration of the enhanced text-to-SQL system."""
    
    # Initialize enhanced retriever
    retriever = EnhancedGraphTraversalRetriever("knowledge_graph.duckdb")
    
    # Initialize analyzer
    analyzer = AttentionAnalyzer(retriever)
    
    # Example query
    query = "show me all trades by government entities with notional amount greater than 1 million"
    
    print("🔍 Enhanced Table Retrieval Analysis")
    print("=" * 50)
    
    # 1. Get tables using enhanced method
    print(f"\n📋 Query: {query}")
    tables, details = retriever.get_tables_with_enhanced_attention(query, max_tables=5)
    
    print(f"\n🎯 Top Retrieved Tables:")
    for i, table in enumerate(tables, 1):
        detail = details.get(table, {})
        score = detail.get('attention_score', 0.0)
        confidence = detail.get('confidence', 'Unknown')
        reasoning = detail.get('query_relevance_analysis', {}).get('reasoning', 'N/A')
        
        print(f"{i}. {table}")
        print(f"   Score: {score:.3f} | Confidence: {confidence}")
        print(f"   Reasoning: {reasoning}")
    
    # 2. Analyze attention patterns
    print(f"\n🧠 Attention Pattern Analysis:")
    attention_analysis = analyzer.analyze_attention_patterns(query)
    
    print(f"Query Intent: {attention_analysis.get('query_intent', 'unknown')}")
    print(f"Query Domain: {attention_analysis.get('query_domain', 'unknown')}")
    print(f"Extracted Entities: {attention_analysis.get('query_entities', [])}")
    
    # 3. Compare different attention methods
    print(f"\n⚖️ Method Comparison:")
    comparison = analyzer.compare_attention_methods(query)
    
    for method, results in comparison.get('methods', {}).items():
        top_3 = [table for table, _ in results[:3]]
        print(f"{method}: {top_3}")
    
    # Show consensus
    agreement = comparison.get('agreement_analysis', {})
    consensus = agreement.get('consensus_tables', [])
    print(f"\nConsensus Tables (appear in multiple methods): {consensus[:5]}")
    
    return {
        'retrieved_tables': tables,
        'table_details': details,
        'attention_analysis': attention_analysis,
        'method_comparison': comparison
    }


if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_enhanced_system()
    print("\n✅ Enhanced text-to-SQL system demonstration completed!")